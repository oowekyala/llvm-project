//===- EmptyTensorElimination.cpp - tensor.empty op elimination -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace bufferization {
#define GEN_PASS_DEF_EMPTYTENSORELIMINATIONPASS
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"
} // namespace bufferization
} // namespace mlir

using namespace mlir;
using namespace mlir::bufferization;

/// Return true if all `neededValues` are in scope at the given
/// `insertionPoint`.
static bool
neededValuesDominateInsertionPoint(const DominanceInfo &domInfo,
                                   Operation *insertionPoint,
                                   const SmallVector<Value> &neededValues) {
  for (Value val : neededValues) {
    if (auto bbArg = dyn_cast<BlockArgument>(val)) {
      Block *owner = bbArg.getOwner();
      if (!owner->findAncestorOpInBlock(*insertionPoint))
        return false;
    } else {
      auto opResult = cast<OpResult>(val);
      if (!domInfo.properlyDominates(opResult.getOwner(), insertionPoint))
        return false;
    }
  }
  return true;
}

/// Find a valid insertion point for a replacement of `emptyTensorOp`'s
/// use of `user` operation, assuming that the replacement may use any
/// value from `neededValues`.
static Operation *
findValidInsertionPoint(Operation *emptyTensorOp, Operation *user,
                        const SmallVector<Value> &neededValues) {
  DominanceInfo domInfo;
  Operation *candidateInsertionPoint = emptyTensorOp;

  // Gather all possible insertion points: the location of
  // `candidateInsertionPoint` and right after the definition of each value in
  // `neededValues`.
  SmallVector<Operation *> insertionPointCandidates;
  insertionPointCandidates.push_back(candidateInsertionPoint);
  for (Value val : neededValues) {
    // Note: The anchor op is using all of `neededValues`, so:
    // * in case of a block argument: There must be at least one op in the block
    //                                (the anchor op or one of its parents).
    // * in case of an OpResult: There must be at least one op right after the
    //                           defining op (the anchor op or one of its
    //                           parents).
    if (auto bbArg = dyn_cast<BlockArgument>(val)) {
      insertionPointCandidates.push_back(
          &bbArg.getOwner()->getOperations().front());
    } else {
      insertionPointCandidates.push_back(val.getDefiningOp()->getNextNode());
    }
  }

  // Select first matching insertion point.
  for (Operation *insertionPoint : insertionPointCandidates) {
    // Check if all needed values are in scope.
    if (!neededValuesDominateInsertionPoint(domInfo, insertionPoint,
                                            neededValues))
      continue;
    // Check if the insertion point is before the use to be replaced.
    if (!domInfo.dominates(insertionPoint, user))
      continue;
    return insertionPoint;
  }

  // No suitable insertion point was found.
  return nullptr;
}

Value mlir::bufferization::buildSubsetExtraction(RewriterBase &rewriter,
                                                 SubsetInsertionOpInterface op,
                                                 tensor::EmptyOp emptyTensorOp,
                                                 Operation *user) {

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  // All values that are needed to create the replacement op.
  SmallVector<Value> neededValues = op.getValuesNeededToBuildSubsetExtraction();
  // Find a suitable insertion point. If no suitable insertion point
  // for the replacement can be found, return an empty value to skip
  // this replacement.
  Operation *insertionPoint =
      findValidInsertionPoint(emptyTensorOp, user, neededValues);
  if (!insertionPoint) {
    // If no already suitable insertion point was found, attempt to move all
    // needed values before the user.
    if (failed(moveValueDefinitions(rewriter, neededValues, user)))
      return {};
    insertionPoint = user;
  }

  rewriter.setInsertionPoint(insertionPoint);
  Value replacement =
      op.buildSubsetExtraction(rewriter, emptyTensorOp->getLoc());
  return replacement;
}

static bool
isEquivalentModuloLayoutPreservingTypeChange(OpOperand *opnd,
                                             const AnalysisState &state) {
  auto *owner = opnd->getOwner();
  return isa<CastOpInterface>(owner) || isa<tensor::CollapseShapeOp>(owner) ||
         isa<tensor::ExpandShapeOp>(owner) ||
         llvm::all_of(state.getAliasingValues(*opnd), [&](auto alias) {
           return alias.relation == BufferRelation::Equivalent &&
                  alias.isDefinite &&
                  alias.value.getType() == opnd->get().getType();
         });
}

LogicalResult mlir::bufferization::eliminateEmptyTensors(
    RewriterBase &rewriter, Operation *op, OneShotAnalysisState &state,
    ControlBuildSubsetExtractionFn subsetsExtractionFn) {
  OpBuilder::InsertionGuard g(rewriter);
  llvm::DenseSet<OpOperand *> visitedOpOperands;
  op->walk([&](SubsetInsertionOpInterface op) {
    visitedOpOperands.clear();
    OpOperand &source = op.getSourceOperand();
    // Skip operands that do not bufferize inplace. "tensor.empty" could still
    // be replaced, but the transformation may not be beneficial.
    if (!state.isInPlace(source))
      return WalkResult::skip();

    // Find tensor.empty ops on the reverse SSA use-def chain. Only follow
    // equivalent tensors. I.e., stop when there are ops such as extract_slice
    // on the path.
    TraversalConfig config;
    config.followEquivalentOnly = true;
    config.alwaysIncludeLeaves = false;
    // Allow crossing reassociative reshape ops (CollapseShapeOp/ExpandShapeOp)
    // which report BufferRelation::Equivalent. The replacement building below
    // wraps the extract_slice in the inverse reshape to recover the empty's
    // original type. followEquivalentOnly guards against following non-reshape
    // type changes (e.g. extract_slice which is Unknown, not Equivalent).
    SetVector<Value> emptyTensors = state.findValueInReverseUseDefChain(
        &source, /*condition=*/
        [&](Value val) { return val.getDefiningOp<tensor::EmptyOp>(); }, config,
        &visitedOpOperands);

    for (Value v : emptyTensors) {
      auto emptyTensorOp = v.getDefiningOp<tensor::EmptyOp>();
      assert(emptyTensorOp && "expected tensor.empty op");
      // Find the use to be replaced from the use-def chain.
      auto iter = llvm::find_if(
          visitedOpOperands, [&emptyTensorOp](OpOperand *opOperand) {
            return llvm::count(emptyTensorOp->getUses(), *opOperand);
          });

      assert(iter != visitedOpOperands.end() && "could not find use");
      OpOperand *useToBeReplaced = *iter;
      Operation *user = useToBeReplaced->getOwner();
      auto replacement = subsetsExtractionFn(rewriter, op, emptyTensorOp, user);
      if (!replacement)
        continue;
      if (emptyTensorOp == replacement.getDefiningOp())
        continue;
      if (replacement.getType() != v.getType()) {
        // - srcShaped is the type of the subview
        // - dstShaped is the type of the tensor.empty that we're replacing
        auto srcShaped = cast<ShapedType>(replacement.getType());
        auto dstShaped = cast<ShapedType>(v.getType());
        if (srcShaped.getElementType() != dstShaped.getElementType())
          continue;

        // We need to make sure that the path from the empty tensor to the
        // subset insertion preserved the layout of the empty tensor. This
        // means each operation on the path should either have same type for
        // the operand and equivalent result, or be one of the known
        // layout-preserving ops (cast and reassociative reshapes).
        if (llvm::any_of(visitedOpOperands, [&](auto *opnd) {
              return !isEquivalentModuloLayoutPreservingTypeChange(opnd, state);
            }))
          continue;

        rewriter.setInsertionPointAfterValue(replacement);
        if (tensor::CastOp::areCastCompatible(srcShaped, dstShaped)) {
          // Same rank and compatible - a cast is likely what we need.
          replacement = tensor::CastOp::create(rewriter, v.getLoc(),
                                               v.getType(), replacement);
        } else {
          // Not cast-compatible (different rank or incompatible shapes).
          // Emit a tensor.reshape. May be canonicalized later to tensor.expand/collapse_shape.
          SmallVector<Value> shapeVals;
          unsigned dynIdx = 0;
          for (int64_t dim : dstShaped.getShape()) {
            if (ShapedType::isDynamic(dim)) {
              shapeVals.push_back(emptyTensorOp.getDynamicSizes()[dynIdx++]);
            } else {
              shapeVals.push_back(
                  arith::ConstantIndexOp::create(rewriter, v.getLoc(), dim));
            }
          }
          auto shapeType =
              RankedTensorType::get({static_cast<int64_t>(dstShaped.getRank())},
                                    rewriter.getIndexType());
          Value shape = tensor::FromElementsOp::create(rewriter, v.getLoc(),
                                                       shapeType, shapeVals);
          replacement = tensor::ReshapeOp::create(
              rewriter, v.getLoc(), dstShaped, replacement, shape);
        }
      }
      // Replace the specific use of the tensor::EmptyOp.
      rewriter.modifyOpInPlace(user,
                               [&]() { useToBeReplaced->assign(replacement); });
      state.resetCache();
    }

    return WalkResult::advance();
  });

  return success();
}

namespace {
struct EmptyTensorElimination
    : public bufferization::impl::EmptyTensorEliminationPassBase<
          EmptyTensorElimination> {
  using Base::Base;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    tensor::TensorDialect>();
  }
};
} // namespace

LogicalResult mlir::bufferization::eliminateEmptyTensors(RewriterBase &rewriter,
                                                         Operation *op) {
  auto moduleOp = dyn_cast<ModuleOp>(op);
  OneShotBufferizationOptions options;
  options.allowReturnAllocsFromLoops = true;
  if (moduleOp)
    options.bufferizeFunctionBoundaries = true;
  OneShotAnalysisState state(op, options);
  if (moduleOp) {
    // Module analysis takes into account function boundaries.
    if (failed(analyzeModuleOp(moduleOp, state)))
      return failure();
  } else {
    // Regular One-Shot Bufferize ignores func.func block arguments, func.call,
    // func.return.
    if (failed(analyzeOp(op, state)))
      return failure();
  }

  return bufferization::eliminateEmptyTensors(rewriter, op, state);
}

void EmptyTensorElimination::runOnOperation() {
  IRRewriter rewriter(getOperation()->getContext());
  if (failed(bufferization::eliminateEmptyTensors(rewriter, getOperation())))
    signalPassFailure();
}
