//===- EmptyTensorElimination.cpp - tensor.empty op elimination -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::linalg;

namespace mlir {
// namespace linalg {
#define GEN_PASS_DEF_EMPTYTENSORELIMINATIONPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
// } // namespace linalg
} // namespace mlir

/// Get an output operand that matches the given input operand and can be used
/// to eliminate a tensor.empty op.
static OpOperand *getUnusedOperand(LinalgOp op, OpOperand *in,
                                   SmallVectorImpl<OpOperand *> &range,
                                   bool unused) {
  for (OpOperand *operand : range) {
    // Operand must be unused.
    if (!unused || op.payloadUsesValueFromOperand(operand))
      continue;
    // Types must match.
    if (operand->get().getType() != in->get().getType())
      continue;
    // Indexing maps must match.
    if (op.getMatchingIndexingMap(operand) != op.getMatchingIndexingMap(in))
      continue;
    return operand;
  }
  return nullptr;
}

/*
 * Forward inputs into outputs to eliminate output tensors
 * that are equivalent to an input. This is only possible if
 * the input tensor is consumed in this linalg op, writable
 */
static llvm::LogicalResult
emptyOutputTensorElimination(RewriterBase &rewriter, LinalgOp op,
                             OneShotAnalysisState &state) {

  // Only ops with all "parallel" iterator types are supported.
  if (op.getNumParallelLoops() != op.getNumLoops())
    return failure();

  for (OpOperand &out : op.getDpsInitsMutable()) {

    // output must be unused
    if (op.payloadUsesValueFromOperand(&out))
      continue;

    auto emptyOp =
        llvm::dyn_cast_or_null<tensor::EmptyOp>(out.get().getDefiningOp());
    if (!emptyOp)
      continue;

    auto inputs = op.getDpsInputOperands();
    // we have an output that is an empty, unused op

    // find an input
    auto *input = getUnusedOperand(op, &out, inputs, false);
    // make sure the input is
    //  1. writable
    if (!state.isWritable(input->get()))
      continue;
    // todo 2. not read from after this op (may be written to though)

    // todo convert the input into the output (need to update payload as well..)  

    // todo it feels like this logic is exactly what bufferization is supposed to do?
    //  does it really only work with equivalence (and require DPS)? BC the kind of logic
    //  I'm implementing is literally the way to implement temporary buffer reuse without
    //  a DPS representation...

  }
  return success();
}

LogicalResult linalg::linalgOpAnchoredEmptyTensorEliminationStep(
    RewriterBase &rewriter, Operation *op, OneShotAnalysisState &state) {
  OpBuilder::InsertionGuard g(rewriter);
  DominanceInfo domInfo;

  op->walk([&](LinalgOp op) {
    // Only ops with all "parallel" iterator types are supported.
    if (op.getNumParallelLoops() != op.getNumLoops())
      return WalkResult::skip();

    for (OpOperand *in : op.getDpsInputOperands()) {
      // Skip non-tensor operands.
      if (!isa<RankedTensorType>(in->get().getType()))
        continue;

      // Find tensor.empty ops on the reverse SSA use-def chain. Only follow
      // equivalent tensors. I.e., stop when there are ops such as extract_slice
      // on the path.
      TraversalConfig config;
      config.followEquivalentOnly = true;
      config.alwaysIncludeLeaves = false;
      SetVector<Value> emptyTensors = state.findValueInReverseUseDefChain(
          in, /*condition=*/
          [&](Value val) {
            return val.getDefiningOp<tensor::EmptyOp>() &&
                   val.getType() == in->get().getType();
          },
          config);
      if (emptyTensors.empty())
        continue;

      SmallVector<OpOperand *> inits;
      for (auto &init : op.getDpsInitsMutable())
        inits.push_back(&init);

      // Find matching out operand.
      OpOperand *out = getUnusedOperand(op, in, inits, true);
      if (!out)
        continue;

      // Check if this transform would violate dominance.
      if (!llvm::all_of(emptyTensors, [&](Value v) {
            return domInfo.properlyDominates(out->get(), v.getDefiningOp());
          }))
        continue;

      // Replace all uses of the tensor.empty, but do not delete it yet. It will
      // fold away later (to not invalidate DominanceInfo).
      for (Value v : emptyTensors) {
        assert(v.getDefiningOp<tensor::EmptyOp>() && "expected tensor.empty");
        rewriter.replaceAllUsesWith(v, out->get());
      }

      // Turn the "in" into an "out".
      rewriter.modifyOpInPlace(op, [&]() {
        out->set(in->get());
        // The original "in" could be removed entirely here (because it will no
        // longer have any uses in the payload), but we delegate this to
        // existing cleanup patterns that remove unused operands.
        in->set(emptyTensors.front());
        BlockArgument outArg = op.getMatchingBlockArgument(out);
        assert(outArg.getUses().empty() && "expected that out has no uses");
        BlockArgument inArg = op.getMatchingBlockArgument(in);
        rewriter.replaceAllUsesWith(inArg, outArg);
        assert(!op.payloadUsesValueFromOperand(in) &&
               "expected that the in operand is now unused");
      });

      state.resetCache();
    }

    return WalkResult::advance();
  });
  return success();
}

namespace {
struct EmptyTensorElimination
    : public impl::EmptyTensorEliminationPassBase<EmptyTensorElimination> {
  using Base::Base;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<bufferization::BufferizationDialect, tensor::TensorDialect>();
  }
};
} // namespace

void EmptyTensorElimination::runOnOperation() {
  auto *op = getOperation();
  IRRewriter rewriter(op->getContext());

  // Use bufferization elimination, follow up with linalg
  (void)bufferization::eliminateEmptyTensors(rewriter, op);

  auto moduleOp = dyn_cast<ModuleOp>(op);
  OneShotBufferizationOptions options;
  options.allowReturnAllocsFromLoops = true;
  if (moduleOp)
    options.bufferizeFunctionBoundaries = true;
  OneShotAnalysisState state(op, options);
  if (moduleOp) {
    // Module analysis takes into account function boundaries.
    if (failed(mlir::bufferization::analyzeModuleOp(moduleOp, state))) {
      signalPassFailure();
      return;
    }
  } else {
    // Regular One-Shot Bufferize ignores func.func block arguments, func.call,
    // func.return.
    if (failed(analyzeOp(op, state))) {

      signalPassFailure();
      return;
    }
  }

  (void)linalg::linalgOpAnchoredEmptyTensorEliminationStep(rewriter, op, state);
}