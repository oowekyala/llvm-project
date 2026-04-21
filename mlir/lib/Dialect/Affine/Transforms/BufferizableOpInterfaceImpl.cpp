//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/UnstructuredControlFlow.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::affine;

namespace mlir {
namespace affine {
namespace {

/// Helper function for loop bufferization. Cast the given buffer to the given
/// memref type.
static Value castBuffer(OpBuilder &b, Value buffer, Type type) {
  assert(isa<BaseMemRefType>(type) && "expected BaseMemRefType");
  assert(isa<BaseMemRefType>(buffer.getType()) && "expected BaseMemRefType");
  // If the buffer already has the correct type, no cast is needed.
  if (buffer.getType() == type)
    return buffer;
  // TODO: In case `type` has a layout map that is not the fully dynamic
  // one, we may not be able to cast the buffer. In that case, the loop
  // iter_arg's layout map must be changed (see uses of `castBuffer`).
  assert(memref::CastOp::areCastCompatible(buffer.getType(), type) &&
         "scf.while op bufferization: cast incompatible");
  return memref::CastOp::create(b, buffer.getLoc(), type, buffer).getResult();
}

/// Helper function for loop bufferization. Return "true" if the given value
/// is guaranteed to not alias with an external tensor apart from values in
/// `exceptions`. A value is external if it is defined outside of the given
/// region or if it is an entry block argument of the region.
static bool doesNotAliasExternalValue(Value value, Region *region,
                                      ValueRange exceptions,
                                      const OneShotAnalysisState &state) {
  assert(region->hasOneBlock() && "expected region with single block");
  bool result = true;
  state.applyOnAliases(value, [&](Value alias) {
    if (llvm::is_contained(exceptions, alias))
      return;
    Region *aliasRegion = alias.getParentRegion();
    if (isa<BlockArgument>(alias) && !region->isProperAncestor(aliasRegion))
      result = false;
    if (isa<OpResult>(alias) && !region->isAncestor(aliasRegion))
      result = false;
  });
  return result;
}

/// Helper function for loop bufferization. Return the indices of all values
/// that have a tensor type.
static DenseSet<int64_t> getTensorIndices(ValueRange values) {
  DenseSet<int64_t> result;
  for (const auto &it : llvm::enumerate(values))
    if (isa<TensorType>(it.value().getType()))
      result.insert(it.index());
  return result;
}

/// Helper function for loop bufferization. Return the indices of all
/// bbArg/yielded value pairs who's buffer relation is "Equivalent".
DenseSet<int64_t> getEquivalentBuffers(Block::BlockArgListType bbArgs,
                                       ValueRange yieldedValues,
                                       const AnalysisState &state) {
  unsigned int minSize = std::min(bbArgs.size(), yieldedValues.size());
  DenseSet<int64_t> result;
  for (unsigned int i = 0; i < minSize; ++i) {
    if (!isa<TensorType>(bbArgs[i].getType()) ||
        !isa<TensorType>(yieldedValues[i].getType()))
      continue;
    if (state.areEquivalentBufferizedValues(bbArgs[i], yieldedValues[i]))
      result.insert(i);
  }
  return result;
}

/// Helper function for loop bufferization. Return the bufferized values of the
/// given OpOperands. If an operand is not a tensor, return the original value.
static FailureOr<SmallVector<Value>>
getBuffers(RewriterBase &rewriter, const MutableOperandRange &operands,
           const BufferizationOptions &options, BufferizationState &state) {
  SmallVector<Value> result;
  for (OpOperand &opOperand : operands) {
    if (isa<TensorType>(opOperand.get().getType())) {
      FailureOr<Value> resultBuffer =
          getBuffer(rewriter, opOperand.get(), options, state);
      if (failed(resultBuffer))
        return failure();
      result.push_back(*resultBuffer);
    } else {
      result.push_back(opOperand.get());
    }
  }
  return result;
}

/// Helper function for loop bufferization. Given a list of bbArgs of the new
/// (bufferized) loop op, wrap the bufferized tensor args (now memrefs) into
/// ToTensorOps, so that the block body can be moved over to the new op.
static SmallVector<Value>
getBbArgReplacements(RewriterBase &rewriter, Block::BlockArgListType bbArgs,
                     Block::BlockArgListType oldBbArgs,
                     const DenseSet<int64_t> &tensorIndices) {
  SmallVector<Value> result;
  for (const auto &it : llvm::enumerate(bbArgs)) {
    size_t idx = it.index();
    Value val = it.value();
    if (tensorIndices.contains(idx)) {
      result.push_back(
          bufferization::ToTensorOp::create(rewriter, val.getLoc(),
                                            oldBbArgs[idx].getType(), val)
              .getResult());
    } else {
      result.push_back(val);
    }
  }
  return result;
}

/// Compute the bufferized type of a loop iter_arg. This type must be equal to
/// the bufferized type of the corresponding init_arg and the bufferized type
/// of the corresponding yielded value.
///
/// This function uses bufferization::getBufferType to compute the bufferized
/// type of the init_arg and of the yielded value. (The computation of the
/// bufferized yielded value type usually requires computing the bufferized type
/// of the iter_arg again; the implementation of getBufferType traces back the
/// use-def chain of the given value and computes a buffer type along the way.)
/// If both buffer types are equal, no casts are needed the computed buffer type
/// can be used directly. Otherwise, the buffer types can only differ in their
/// layout map and a cast must be inserted.
static FailureOr<BufferLikeType> computeLoopRegionIterArgBufferType(
    Operation *loopOp, BlockArgument iterArg, Value initArg, Value yieldedValue,
    const BufferizationOptions &options, const BufferizationState &state,
    SmallVector<Value> &invocationStack) {
  // Determine the buffer type of the init_arg.
  auto initArgBufferType =
      bufferization::getBufferType(initArg, options, state, invocationStack);
  if (failed(initArgBufferType))
    return failure();

  if (llvm::count(invocationStack, iterArg) >= 2) {
    // If the iter_arg is already twice on the invocation stack, just take the
    // type of the init_arg. This is to avoid infinite loops when calculating
    // the buffer type. This will most likely result in computing a memref type
    // with a fully dynamic layout map.

    // Note: For more precise layout map computation, a fixpoint iteration could
    // be done (i.e., re-computing the yielded buffer type until the bufferized
    // iter_arg type no longer changes). This current implementation immediately
    // switches to a fully dynamic layout map when a mismatch between bufferized
    // init_arg type and bufferized yield value type is detected.
    return *initArgBufferType;
  }

  // Compute the buffer type of the yielded value.
  BufferLikeType yieldedValueBufferType;
  if (isa<BaseMemRefType>(yieldedValue.getType())) {
    // scf.yield was already bufferized.
    yieldedValueBufferType = cast<BufferLikeType>(yieldedValue.getType());
  } else {
    // Note: This typically triggers a recursive call for the buffer type of
    // the iter_arg.
    auto maybeBufferType = bufferization::getBufferType(yieldedValue, options,
                                                        state, invocationStack);
    if (failed(maybeBufferType))
      return failure();
    yieldedValueBufferType = *maybeBufferType;
  }

  // If yielded type and init_arg type are the same, use that type directly.
  if (*initArgBufferType == yieldedValueBufferType)
    return yieldedValueBufferType;

  // If there is a mismatch between the yielded buffer type and the init_arg
  // buffer type, the buffer type must be promoted to a fully dynamic layout
  // map.
  auto yieldedBufferType = cast<BaseMemRefType>(yieldedValueBufferType);
  auto iterTensorType = cast<TensorType>(iterArg.getType());
  auto initBufferType = llvm::cast<BaseMemRefType>(*initArgBufferType);
  if (initBufferType.getMemorySpace() != yieldedBufferType.getMemorySpace())
    return loopOp->emitOpError(
        "init_arg and yielded value bufferize to inconsistent memory spaces");
#ifndef NDEBUG
  if (auto yieldedRankedBufferType = dyn_cast<MemRefType>(yieldedBufferType)) {
    assert(
        llvm::all_equal({yieldedRankedBufferType.getShape(),
                         cast<MemRefType>(initBufferType).getShape(),
                         cast<RankedTensorType>(iterTensorType).getShape()}) &&
        "expected same shape");
  }
#endif // NDEBUG
  return cast<BufferLikeType>(getMemRefTypeWithFullyDynamicLayout(
      iterTensorType, yieldedBufferType.getMemorySpace()));
}

/// Return `true` if the given loop may have 0 iterations.
bool mayHaveZeroIterations(affine::AffineForOp forOp) {
  if (!forOp.hasConstantBounds())
    return true;
  return forOp.getConstantUpperBound() <= forOp.getConstantLowerBound();
}

/// Bufferization of scf.for. Replace with a new scf.for that operates on
/// memrefs.
struct ForOpInterface
    : public BufferizableOpInterface::ExternalModel<ForOpInterface,
                                                    affine::AffineForOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    auto forOp = cast<affine::AffineForOp>(op);

    // If the loop has zero iterations, the results of the op are their
    // corresponding init_args, meaning that the init_args bufferize to a read.
    if (mayHaveZeroIterations(forOp))
      return true;

    // affine::AffineForOp alone doesn't bufferize to a memory read, one of the
    // uses of its matching bbArg may.
    return state.isValueRead(forOp.getTiedLoopRegionIterArg(&opOperand));
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Tensor iter_args of affine::AffineForOps are always considered as a
    // write.
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    auto forOp = cast<affine::AffineForOp>(op);
    OpResult opResult = forOp.getTiedLoopResult(&opOperand);
    BufferRelation relation = bufferRelation(op, opResult, state);
    return {{opResult, relation,
             /*isDefinite=*/relation == BufferRelation::Equivalent}};
  }

  BufferRelation bufferRelation(Operation *op, OpResult opResult,
                                const AnalysisState &state) const {
    // ForOp results are equivalent to their corresponding init_args if the
    // corresponding iter_args and yield values are equivalent.
    auto forOp = cast<affine::AffineForOp>(op);
    BlockArgument bbArg = forOp.getTiedLoopRegionIterArg(opResult);
    bool equivalentYield = state.areEquivalentBufferizedValues(
        bbArg, forOp.getTiedLoopYieldedValue(bbArg)->get());
    return equivalentYield ? BufferRelation::Equivalent
                           : BufferRelation::Unknown;
  }

  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    // Interestingly, affine::AffineForOp's bbArg can **always** be viewed
    // inplace from the perspective of ops nested under:
    //   1. Either the matching iter operand is not bufferized inplace and an
    //      alloc + optional copy makes the bbArg itself inplaceable.
    //   2. Or the matching iter operand is bufferized inplace and bbArg just
    //      bufferizes to that too.
    return true;
  }

  LogicalResult
  resolveConflicts(Operation *op, RewriterBase &rewriter,
                   const AnalysisState &analysisState,
                   const BufferizationState &bufferizationState) const {
    auto bufferizableOp = cast<BufferizableOpInterface>(op);
    if (failed(bufferizableOp.resolveTensorOpOperandConflicts(
            rewriter, analysisState, bufferizationState)))
      return failure();

    if (analysisState.getOptions().copyBeforeWrite)
      return success();

    // According to the `getAliasing...` implementations, a bufferized OpResult
    // may alias only with the corresponding bufferized init_arg (or with a
    // newly allocated buffer) and not with other buffers defined outside of the
    // loop. I.e., the i-th OpResult may alias with the i-th init_arg;
    // but not with any other OpOperand.
    auto forOp = cast<affine::AffineForOp>(op);
    auto yieldOp =
        cast<affine::AffineYieldOp>(forOp.getBody()->getTerminator());
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(yieldOp);

    // Indices of all iter_args that have tensor type. These are the ones that
    // are bufferized.
    DenseSet<int64_t> indices = getTensorIndices(forOp.getInits());
    // For every yielded value, does it alias with something defined outside of
    // the loop?
    SmallVector<Value> yieldValues;
    for (const auto it : llvm::enumerate(yieldOp.getOperands())) {
      // Note: `state` is guaranteed to be a `OneShotAnalysisState`, but this
      // type cannot be used in the signature of `resolveConflicts` because the
      // op interface is in the "IR" build unit and the `OneShotAnalysisState`
      // is defined in the "Transforms" build unit.
      if (!indices.contains(it.index()) ||
          doesNotAliasExternalValue(
              it.value(), &forOp.getRegion(),
              /*exceptions=*/forOp.getRegionIterArgs()[it.index()],
              static_cast<const OneShotAnalysisState &>(analysisState))) {
        yieldValues.push_back(it.value());
        continue;
      }
      FailureOr<Value> alloc = allocateTensorForShapedValue(
          rewriter, yieldOp.getLoc(), it.value(), analysisState.getOptions(),
          bufferizationState);
      if (failed(alloc))
        return failure();
      yieldValues.push_back(*alloc);
    }

    rewriter.modifyOpInPlace(
        yieldOp, [&]() { yieldOp.getOperandsMutable().assign(yieldValues); });
    return success();
  }

  FailureOr<BufferLikeType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    auto forOp = cast<affine::AffineForOp>(op);
    assert(getOwnerOfValue(value) == op && "invalid value");
    assert(isa<TensorType>(value.getType()) && "expected tensor type");

    if (auto opResult = dyn_cast<OpResult>(value)) {
      // The type of an OpResult must match the corresponding iter_arg type.
      BlockArgument bbArg = forOp.getTiedLoopRegionIterArg(opResult);
      return bufferization::getBufferType(bbArg, options, state,
                                          invocationStack);
    }

    // Compute result/argument number.
    BlockArgument bbArg = cast<BlockArgument>(value);
    unsigned resultNum = forOp.getTiedLoopResult(bbArg).getResultNumber();

    // Compute the bufferized type.
    auto yieldOp =
        cast<affine::AffineYieldOp>(forOp.getBody()->getTerminator());
    Value yieldedValue = yieldOp.getOperand(resultNum);
    BlockArgument iterArg = forOp.getRegionIterArgs()[resultNum];
    Value initArg = forOp.getInits()[resultNum];
    return computeLoopRegionIterArgBufferType(
        op, iterArg, initArg, yieldedValue, options, state, invocationStack);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto forOp = cast<affine::AffineForOp>(op);
    Block *oldLoopBody = forOp.getBody();

    // Indices of all iter_args that have tensor type. These are the ones that
    // are bufferized.
    DenseSet<int64_t> indices = getTensorIndices(forOp.getInits());

    // The new memref init_args of the loop.
    FailureOr<SmallVector<Value>> maybeInitArgs =
        getBuffers(rewriter, forOp.getInitsMutable(), options, state);
    if (failed(maybeInitArgs))
      return failure();
    SmallVector<Value> initArgs = *maybeInitArgs;

    // Cast init_args if necessary.
    SmallVector<Value> castedInitArgs;
    for (const auto &it : llvm::enumerate(initArgs)) {
      Value initArg = it.value();
      Value result = forOp->getResult(it.index());
      // If the type is not a tensor, bufferization doesn't need to touch it.
      if (!isa<TensorType>(result.getType())) {
        castedInitArgs.push_back(initArg);
        continue;
      }
      auto targetType = bufferization::getBufferType(result, options, state);
      if (failed(targetType))
        return failure();
      castedInitArgs.push_back(castBuffer(rewriter, initArg, *targetType));
    }

    // Construct a new scf.for op with memref instead of tensor values.
    auto newForOp = affine::AffineForOp::create(
        rewriter, forOp.getLoc(), forOp.getLowerBoundOperands(),
        forOp.getLowerBoundMap(), forOp.getUpperBoundOperands(),
        forOp.getUpperBoundMap(), forOp.getStepAsInt(), castedInitArgs);
    newForOp->setAttrs(forOp->getAttrs());
    Block *loopBody = newForOp.getBody();

    // Set up new iter_args. The loop body uses tensors, so wrap the (memref)
    // iter_args of the new loop in ToTensorOps.
    rewriter.setInsertionPointToStart(loopBody);
    SmallVector<Value> iterArgs =
        getBbArgReplacements(rewriter, newForOp.getRegionIterArgs(),
                             forOp.getRegionIterArgs(), indices);
    iterArgs.insert(iterArgs.begin(), newForOp.getInductionVar());

    // Move loop body to new loop.
    rewriter.mergeBlocks(oldLoopBody, loopBody, iterArgs);

    // Replace loop results.
    replaceOpWithBufferizedValues(rewriter, op, newForOp->getResults());

    return success();
  }

  /// Assert that yielded values of an scf.for op are equivalent to their
  /// corresponding bbArgs. In that case, the buffer relations of the
  /// corresponding OpResults are "Equivalent".
  ///
  /// If this is not the case, an allocs+copies are inserted and yielded from
  /// the loop. This could be a performance problem, so it must be explicitly
  /// activated with `alloc-return-allocs`.
  LogicalResult verifyAnalysis(Operation *op,
                               const AnalysisState &state) const {
    const auto &options =
        static_cast<const OneShotBufferizationOptions &>(state.getOptions());
    if (options.allowReturnAllocsFromLoops)
      return success();

    auto forOp = cast<affine::AffineForOp>(op);
    auto yieldOp =
        cast<affine::AffineYieldOp>(forOp.getBody()->getTerminator());
    for (OpResult opResult : op->getOpResults()) {
      if (!isa<TensorType>(opResult.getType()))
        continue;

      // Note: This is overly strict. We should check for aliasing bufferized
      // values. But we don't have a "must-alias" analysis yet.
      if (bufferRelation(op, opResult, state) != BufferRelation::Equivalent)
        return yieldOp->emitError()
               << "Yield operand #" << opResult.getResultNumber()
               << " is not equivalent to the corresponding iter bbArg";
    }

    return success();
  }
};

/// Bufferization of scf.yield. Bufferized as part of their enclosing ops, so
/// this is for analysis only.
struct YieldOpInterface
    : public BufferizableOpInterface::ExternalModel<YieldOpInterface,
                                                    affine::AffineYieldOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    // Note: for affine.for (like scf.for), we do NOT alias the yielded value
    // with the for result here. The equivalence is established lazily by
    // equivalenceAnalysis after all in-place decisions are made. Doing it
    // eagerly would pre-populate alias sets in the constructor (via
    // mustBufferizeInPlace) and cause spurious RaW conflicts downstream.
    return {};
  }

  bool mustBufferizeInPlace(Operation *op, OpOperand &opOperand,
                            const AnalysisState &state) const {
    // Yield operands always bufferize inplace. Otherwise, an alloc + copy
    // may be generated inside the block. We should not return/yield allocations
    // when possible.
    return true;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto yieldOp = cast<affine::AffineYieldOp>(op);
    if (!isa<affine::AffineForOp>(yieldOp->getParentOp()))
      return yieldOp->emitError("unsupported affine::AffineYieldOp parent");

    SmallVector<Value> newResults;
    for (const auto &it : llvm::enumerate(yieldOp->getOperands())) {
      Value value = it.value();
      if (isa<TensorType>(value.getType())) {
        FailureOr<Value> maybeBuffer =
            getBuffer(rewriter, value, options, state);
        if (failed(maybeBuffer))
          return failure();
        Value buffer = *maybeBuffer;
        // We may have to cast the value before yielding it.
        if (isa<affine::AffineForOp>(yieldOp->getParentOp())) {
          FailureOr<BufferLikeType> resultType = bufferization::getBufferType(
              yieldOp->getParentOp()->getResult(it.index()), options, state);
          if (failed(resultType))
            return failure();
          buffer = castBuffer(rewriter, buffer, *resultType);
        }
        newResults.push_back(buffer);
      } else {
        newResults.push_back(value);
      }
    }

    replaceOpWithNewBufferizedOp<affine::AffineYieldOp>(rewriter, op,
                                                        newResults);
    return success();
  }
};

} // namespace
} // namespace affine
} // namespace mlir

void mlir::affine::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, affine::AffineDialect *dialect) {
    AffineForOp::attachInterface<ForOpInterface>(*ctx);
    AffineYieldOp::attachInterface<YieldOpInterface>(*ctx);
  });
}
