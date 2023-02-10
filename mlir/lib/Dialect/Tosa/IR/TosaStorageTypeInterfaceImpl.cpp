//===- TosaStorageTypeInterfaceImpl.cpp - Impl. of TosaStorageType ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaStorageTypeInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::tosa;
using namespace mlir::tosa::detail;

namespace {
using Concept = TosaStorageTypeInterfaceTraits::Concept;

Value lowerTosaToLinalgElementWiseOpDefault(Type elementTy,
                                            PatternRewriter &rewriter,
                                            Operation *op, ValueRange args,
                                            ArrayRef<Type> resultTypes);

/// Implementation of TosaStorageType for mlir's builtin IntegerType
struct IntegerStorageTypeItf
    : public TosaStorageTypeInterfaceTraits::ExternalModel<
          IntegerStorageTypeItf, IntegerType> {

  int getWidth(::mlir::Type t) const { return t.getIntOrFloatBitWidth(); }

  bool isSigned(::mlir::Type t) const { return t.isSignedInteger(); }
  bool isSignless(::mlir::Type t) const { return t.isSignlessInteger(); }
  bool isUnsigned(::mlir::Type t) const { return t.isUnsignedInteger(); }

  bool isIntegral(::mlir::Type t) const { return true; }
  bool isFloatingPoint(::mlir::Type t) const { return false; }

  Attribute materializeAttribute(::mlir::Type t, OpBuilder &rewriter,
                                 SpecialValueId value) const {
    APInt v;
    if (value == SpecialValueId::ZERO)
      v = APInt(getWidth(t), 0, isSigned(t));

    else if (value == SpecialValueId::ONE)
      v = APInt(getWidth(t), 1, isSigned(t));

    else if (value == SpecialValueId::RANGE_MIN)
      if (isUnsigned(t))
        v = APInt::getMinValue(getWidth(t));
      else
        v = APInt::getSignedMinValue(getWidth(t));

    else if (value == SpecialValueId::RANGE_MAX)
      if (isUnsigned(t))
        v = APInt::getMaxValue(getWidth(t));
      else
        v = APInt::getSignedMaxValue(getWidth(t));

    else if (value == SpecialValueId::ALL_ONES)
      v = APInt::getAllOnes(getWidth(t));

    return rewriter.getIntegerAttr(t, v);
  }

  Value materializeConstant(Type t, OpBuilder &rewriter, Location loc,
                            Attribute attr) const {
    return rewriter.create<arith::ConstantOp>(loc, attr);
  }

  Value lowerTosaElementWiseOp(Type elementTy, PatternRewriter &rewriter,
                               Operation *op, ValueRange args,
                               ArrayRef<Type> resultTypes) const {
    return lowerTosaToLinalgElementWiseOpDefault(elementTy, rewriter, op, args,
                                                 resultTypes);
  }

  Value lowerTosaReductionKernel(Type t, PatternRewriter &rewriter,
                                 Operation *op, ValueRange args) const {
    Location loc = op->getLoc();
    if (isa<tosa::ReduceSumOp>(op)) {
      return rewriter.create<arith::AddIOp>(loc, args);
    }

    if (isa<tosa::ReduceProdOp>(op)) {
      return rewriter.create<arith::MulIOp>(loc, args);
    }

    if (isa<tosa::ReduceMinOp>(op)) {
      auto predicate = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, args[0], args[1]);
      return rewriter.create<arith::SelectOp>(loc, predicate, args[0], args[1]);
    }

    if (isa<tosa::ReduceMaxOp>(op)) {
      auto predicate = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sgt, args[0], args[1]);
      return rewriter.create<arith::SelectOp>(loc, predicate, args[0], args[1]);
    }

    if (isa<tosa::ReduceAllOp>(op))
      return rewriter.create<arith::AndIOp>(loc, args);

    if (isa<tosa::ReduceAnyOp>(op))
      return rewriter.create<arith::OrIOp>(loc, args);

    return {};
  }
};

/// Implementation of TosaStorageType for mlir's builtin IntegerType
struct FloatStorageTypeItf
    : public TosaStorageTypeInterfaceTraits::ExternalModel<FloatStorageTypeItf,
                                                           FloatType> {

  int getWidth(::mlir::Type t) const { return t.getIntOrFloatBitWidth(); }

  bool isSigned(::mlir::Type t) const { return true; }
  bool isSignless(::mlir::Type t) const { return false; }
  bool isUnsigned(::mlir::Type t) const { return false; }

  bool isIntegral(::mlir::Type t) const { return false; }
  bool isFloatingPoint(::mlir::Type t) const { return true; }

  Attribute materializeAttribute(::mlir::Type t, OpBuilder &rewriter,
                                 SpecialValueId value) const {
    APFloat v(0.0);
    if (value == SpecialValueId::ZERO)
      v = APFloat(0.0);
    else if (value == SpecialValueId::ONE)
      v = APFloat(1.0);
    else if (value == SpecialValueId::RANGE_MIN)
      v = APFloat::getLargest(t.cast<FloatType>().getFloatSemantics(), true);
    else if (value == SpecialValueId::RANGE_MAX)
      v = APFloat::getLargest(t.cast<FloatType>().getFloatSemantics(), false);
    else if (value == SpecialValueId::ALL_ONES)
      v = APFloat::getAllOnesValue(t.cast<FloatType>().getFloatSemantics());

    return rewriter.getFloatAttr(t, v);
  }

  Value materializeConstant(Type t, OpBuilder &rewriter, Location loc,
                            Attribute attr) const {
    return rewriter.create<arith::ConstantOp>(loc, attr);
  }
  Value lowerTosaElementWiseOp(Type elementTy, PatternRewriter &rewriter,
                               Operation *op, ValueRange args,
                               ArrayRef<Type> resultTypes) const {
    return lowerTosaToLinalgElementWiseOpDefault(elementTy, rewriter, op, args,
                                                 resultTypes);
  }

  Value lowerTosaReductionKernel(Type t, PatternRewriter &rewriter,
                                 Operation *op, ValueRange args) const {
    Location loc = op->getLoc();
    if (isa<tosa::ReduceSumOp>(op)) {
      return rewriter.create<arith::AddFOp>(loc, args);
    } else if (isa<tosa::ReduceProdOp>(op)) {
      return rewriter.create<arith::MulFOp>(loc, args);
    } else if (isa<tosa::ReduceMinOp>(op)) {
      return rewriter.create<arith::MinFOp>(loc, args[0], args[1]);
    } else if (isa<tosa::ReduceMaxOp>(op)) {
      return rewriter.create<arith::MaxFOp>(loc, args[0], args[1]);
    }
    return {};
  }
};

/// @brief default lowering that should work for most patterns
Value lowerTosaToLinalgElementWiseOpDefault(Type elementType0,
                                            PatternRewriter &rewriter,
                                            Operation *op, ValueRange args,
                                            ArrayRef<Type> resultTypes) {
  Location loc = op->getLoc();
  auto elementTy = elementType0.cast<TosaStorageType>();

  // tosa::AbsOp
  if (isa<tosa::AbsOp>(op) && elementTy.isFloatingPoint())
    return rewriter.create<math::AbsFOp>(loc, resultTypes, args);

  if (isa<tosa::AbsOp>(op) && elementTy.isIntegral()) {
    auto zero =
        elementTy.materializeSpecialValue(rewriter, loc, SpecialValueId::ZERO);
    auto cmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                              args[0], zero);
    auto neg = rewriter.create<arith::SubIOp>(loc, zero, args[0]);
    return rewriter.create<arith::SelectOp>(loc, cmp, args[0], neg);
  }

  // tosa::AddOp
  if (isa<tosa::AddOp>(op) && elementTy.isFloatingPoint())
    return rewriter.create<arith::AddFOp>(loc, resultTypes, args);

  if (isa<tosa::AddOp>(op) && elementTy.isIntegral())
    return rewriter.create<arith::AddIOp>(loc, resultTypes, args);

  // tosa::SubOp
  if (isa<tosa::SubOp>(op) && elementTy.isFloatingPoint())
    return rewriter.create<arith::SubFOp>(loc, resultTypes, args);

  if (isa<tosa::SubOp>(op) && elementTy.isIntegral())
    return rewriter.create<arith::SubIOp>(loc, resultTypes, args);

  // tosa::MulOp
  if (isa<tosa::MulOp>(op)) {
    if (elementTy.isFloatingPoint()) {
      if (dyn_cast<tosa::MulOp>(op).getShift() != 0) {
        (void)rewriter.notifyMatchFailure(op,
                                          "Cannot have shift value for float");
        return nullptr;
      }
      return rewriter.create<arith::MulFOp>(loc, resultTypes, args);

    } else if (elementTy.isIntegral()) {

      Value a = args[0];
      Value b = args[1];
      auto shift =
          op->getAttr("shift").cast<IntegerAttr>().getValue().getSExtValue();
      if (shift > 0) {
        auto shiftConst =
            rewriter.create<arith::ConstantIntOp>(loc, shift, /*bitwidth=*/8);
        if (!a.getType().isInteger(32))
          a = rewriter.create<arith::ExtSIOp>(loc, rewriter.getI32Type(), a);

        if (!b.getType().isInteger(32))
          b = rewriter.create<arith::ExtSIOp>(loc, rewriter.getI32Type(), b);

        auto result = rewriter.create<tosa::ApplyScaleOp>(
            loc, rewriter.getI32Type(), a, b, shiftConst,
            rewriter.getBoolAttr(false));

        if (elementTy.isInteger(32))
          return result;

        return rewriter.create<arith::TruncIOp>(loc, elementTy, result);
      }

      int aWidth = a.getType().cast<TosaStorageType>().getWidth();
      int bWidth = b.getType().cast<TosaStorageType>().getWidth();
      int cWidth = resultTypes[0].cast<TosaStorageType>().getWidth();

      if (aWidth < cWidth)
        a = rewriter.create<arith::ExtSIOp>(loc, resultTypes[0], a);
      if (bWidth < cWidth)
        b = rewriter.create<arith::ExtSIOp>(loc, resultTypes[0], b);

      return rewriter.create<arith::MulIOp>(loc, resultTypes, a, b);
    }
  }

  // tosa::DivOp
  if (isa<tosa::DivOp>(op) && elementTy.isIntegral())
    return rewriter.create<arith::DivSIOp>(loc, resultTypes, args);

  // tosa::ReciprocalOp
  if (isa<tosa::ReciprocalOp>(op) && elementTy.isFloatingPoint()) {
    auto one =
        elementTy.materializeSpecialValue(rewriter, loc, SpecialValueId::ONE);
    return rewriter.create<arith::DivFOp>(loc, resultTypes, one, args[0]);
  }

  // tosa::NegateOp
  if (isa<tosa::NegateOp>(op) && elementTy.isFloatingPoint())
    return rewriter.create<arith::NegFOp>(loc, resultTypes, args);

  if (isa<tosa::NegateOp>(op) && elementTy.isIntegral() &&
      !cast<tosa::NegateOp>(op).getQuantizationInfo()) {
    auto constant =
        elementTy.materializeSpecialValue(rewriter, loc, SpecialValueId::ZERO);
    return rewriter.create<arith::SubIOp>(loc, resultTypes, constant, args[0]);
  }

  if (isa<tosa::NegateOp>(op) && elementTy.isIntegral() &&
      cast<tosa::NegateOp>(op).getQuantizationInfo()) {
    // todo quantized negate
    auto quantizationInfo = cast<tosa::NegateOp>(op).getQuantizationInfo();
    int32_t inputBitWidth = elementTy.getWidth();
    int64_t inZp = quantizationInfo.value().getInputZp();
    int64_t outZp = quantizationInfo.value().getOutputZp();

    // Compute the maximum value that can occur in the intermediate buffer.
    int64_t zpAdd = inZp + outZp;
    int64_t maxValue = APInt::getSignedMaxValue(inputBitWidth).getSExtValue() +
                       std::abs(zpAdd) + 1;

    // Convert that maximum value into the maximum bitwidth needed to represent
    // it. We assume 48-bit numbers may be supported further in the pipeline.
    int intermediateBitWidth = 64;
    if (maxValue <= APInt::getSignedMaxValue(16).getSExtValue()) {
      intermediateBitWidth = 16;
    } else if (maxValue <= APInt::getSignedMaxValue(32).getSExtValue()) {
      intermediateBitWidth = 32;
    } else if (maxValue <= APInt::getSignedMaxValue(48).getSExtValue()) {
      intermediateBitWidth = 48;
    }

    Type intermediateType = rewriter.getIntegerType(intermediateBitWidth);
    Value zpAddValue = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(intermediateType, zpAdd));

    // The negation can be applied by doing:
    //  outputValue = inZp + outZp - inputValue
    auto ext = rewriter.create<arith::ExtSIOp>(loc, intermediateType, args[0]);
    auto sub = rewriter.create<arith::SubIOp>(loc, zpAddValue, ext);

    // Clamp to the negation range.
    Value min = rewriter.create<arith::ConstantIntOp>(
        loc, APInt::getSignedMinValue(inputBitWidth).getSExtValue(),
        intermediateType);
    Value max = rewriter.create<arith::ConstantIntOp>(
        loc, APInt::getSignedMaxValue(inputBitWidth).getSExtValue(),
        intermediateType);
    auto clamp = clampIntHelper(loc, sub, min, max, rewriter);

    // Truncate to the final value.
    return rewriter.create<arith::TruncIOp>(loc, elementTy, clamp);
  }

  // tosa::BitwiseAndOp
  if (isa<tosa::BitwiseAndOp>(op) && elementTy.isIntegral())
    return rewriter.create<arith::AndIOp>(loc, resultTypes, args);

  // tosa::BitwiseOrOp
  if (isa<tosa::BitwiseOrOp>(op) && elementTy.isIntegral())
    return rewriter.create<arith::OrIOp>(loc, resultTypes, args);

  // tosa::BitwiseNotOp
  if (isa<tosa::BitwiseNotOp>(op) && elementTy.isIntegral()) {
    auto allOnes = elementTy.materializeSpecialValue(rewriter, loc,
                                                     SpecialValueId::ALL_ONES);
    return rewriter.create<arith::XOrIOp>(loc, resultTypes, args[0], allOnes);
  }

  // tosa::BitwiseXOrOp
  if (isa<tosa::BitwiseXorOp>(op) && elementTy.isIntegral())
    return rewriter.create<arith::XOrIOp>(loc, resultTypes, args);

  // tosa::LogicalLeftShiftOp
  if (isa<tosa::LogicalLeftShiftOp>(op) && elementTy.isIntegral())
    return rewriter.create<arith::ShLIOp>(loc, resultTypes, args);

  // tosa::LogicalRightShiftOp
  if (isa<tosa::LogicalRightShiftOp>(op) && elementTy.isIntegral())
    return rewriter.create<arith::ShRUIOp>(loc, resultTypes, args);

  // tosa::ArithmeticRightShiftOp
  if (isa<tosa::ArithmeticRightShiftOp>(op) && elementTy.isIntegral()) {
    auto result = rewriter.create<arith::ShRSIOp>(loc, resultTypes, args);
    auto round = op->getAttr("round").cast<BoolAttr>().getValue();
    if (!round) {
      return result;
    }

    Type i1Ty = IntegerType::get(rewriter.getContext(), /*width=*/1);
    auto one =
        elementTy.materializeSpecialValue(rewriter, loc, SpecialValueId::ONE);
    auto zero =
        elementTy.materializeSpecialValue(rewriter, loc, SpecialValueId::ZERO);
    auto i1one =
        rewriter.create<arith::ConstantOp>(loc, IntegerAttr::get(i1Ty, 1));

    // Checking that input2 != 0
    auto shiftValueGreaterThanZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, args[1], zero);

    // Checking for the last bit of input1 to be 1
    auto subtract =
        rewriter.create<arith::SubIOp>(loc, resultTypes, args[1], one);
    auto shifted =
        rewriter.create<arith::ShRSIOp>(loc, resultTypes, args[0], subtract)
            ->getResults();
    auto truncated =
        rewriter.create<arith::TruncIOp>(loc, i1Ty, shifted, std::nullopt);
    auto isInputOdd =
        rewriter.create<arith::AndIOp>(loc, i1Ty, truncated, i1one);

    auto shouldRound = rewriter.create<arith::AndIOp>(
        loc, i1Ty, shiftValueGreaterThanZero, isInputOdd);
    auto extended =
        rewriter.create<arith::ExtUIOp>(loc, resultTypes, shouldRound);
    return rewriter.create<arith::AddIOp>(loc, resultTypes, result, extended);
  }

  // tosa::ClzOp
  if (isa<tosa::ClzOp>(op) && elementTy.isIntegral()) {
    return rewriter.create<math::CountLeadingZerosOp>(loc, elementTy, args[0]);
  }

  // tosa::LogicalAnd
  if (isa<tosa::LogicalAndOp>(op) && elementTy.isBool())
    return rewriter.create<arith::AndIOp>(loc, resultTypes, args);

  // tosa::LogicalNot
  if (isa<tosa::LogicalNotOp>(op) && elementTy.isBool()) {
    auto one =
        elementTy.materializeSpecialValue(rewriter, loc, SpecialValueId::ONE);
    return rewriter.create<arith::XOrIOp>(loc, resultTypes, args[0], one);
  }

  // tosa::LogicalOr
  if (isa<tosa::LogicalOrOp>(op) && elementTy.isBool())
    return rewriter.create<arith::OrIOp>(loc, resultTypes, args);

  // tosa::LogicalXor
  if (isa<tosa::LogicalXorOp>(op) && elementTy.isBool())
    return rewriter.create<arith::XOrIOp>(loc, resultTypes, args);

  // tosa::PowOp
  if (isa<tosa::PowOp>(op) && elementTy.isFloatingPoint())
    return rewriter.create<mlir::math::PowFOp>(loc, resultTypes, args);

  // tosa::RsqrtOp
  if (isa<tosa::RsqrtOp>(op) && elementTy.isFloatingPoint())
    return rewriter.create<mlir::math::RsqrtOp>(loc, resultTypes, args);

  // tosa::LogOp
  if (isa<tosa::LogOp>(op) && elementTy.isFloatingPoint())
    return rewriter.create<mlir::math::LogOp>(loc, resultTypes, args);

  // tosa::ExpOp
  if (isa<tosa::ExpOp>(op) && elementTy.isFloatingPoint())
    return rewriter.create<mlir::math::ExpOp>(loc, resultTypes, args);

  // tosa::TanhOp
  if (isa<tosa::TanhOp>(op) && elementTy.isFloatingPoint())
    return rewriter.create<mlir::math::TanhOp>(loc, resultTypes, args);

  // tosa::GreaterOp
  if (isa<tosa::GreaterOp>(op) && elementTy.isFloatingPoint())
    return rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT,
                                          args[0], args[1]);

  if (isa<tosa::GreaterOp>(op) && elementTy.isSignlessInteger())
    return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                          args[0], args[1]);

  // tosa::GreaterEqualOp
  if (isa<tosa::GreaterEqualOp>(op) && elementTy.isFloatingPoint())
    return rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE,
                                          args[0], args[1]);

  if (isa<tosa::GreaterEqualOp>(op) && elementTy.isSignlessInteger())
    return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                          args[0], args[1]);

  // tosa::EqualOp
  if (isa<tosa::EqualOp>(op) && elementTy.isFloatingPoint())
    return rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                          args[0], args[1]);

  if (isa<tosa::EqualOp>(op) && elementTy.isSignlessInteger())
    return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                          args[0], args[1]);

  // tosa::SelectOp
  if (isa<tosa::SelectOp>(op)) {
    elementTy = op->getOperand(1)
                    .getType()
                    .cast<ShapedType>()
                    .getElementType()
                    .cast<TosaStorageType>();
    if (elementTy.isFloatingPoint() || elementTy.isIntegral())
      return rewriter.create<arith::SelectOp>(loc, args[0], args[1], args[2]);
  }

  // tosa::MaximumOp
  if (isa<tosa::MaximumOp>(op) && elementTy.isFloatingPoint()) {
    return rewriter.create<arith::MaxFOp>(loc, args[0], args[1]);
  }

  if (isa<tosa::MaximumOp>(op) && elementTy.isSignlessInteger()) {
    auto predicate = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sgt, args[0], args[1]);
    return rewriter.create<arith::SelectOp>(loc, predicate, args[0], args[1]);
  }

  // tosa::MinimumOp
  if (isa<tosa::MinimumOp>(op) && elementTy.isFloatingPoint()) {
    return rewriter.create<arith::MinFOp>(loc, args[0], args[1]);
  }

  if (isa<tosa::MinimumOp>(op) && elementTy.isSignlessInteger()) {
    auto predicate = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, args[0], args[1]);
    return rewriter.create<arith::SelectOp>(loc, predicate, args[0], args[1]);
  }

  // tosa::CeilOp
  if (isa<tosa::CeilOp>(op) && elementTy.isFloatingPoint())
    return rewriter.create<math::CeilOp>(loc, resultTypes, args);

  // tosa::FloorOp
  if (isa<tosa::FloorOp>(op) && elementTy.isFloatingPoint())
    return rewriter.create<math::FloorOp>(loc, resultTypes, args);

  // tosa::ClampOp
  if (isa<tosa::ClampOp>(op)) {
    if (elementTy.isFloatingPoint()) {
      bool losesInfo = false;
      APFloat minApf = op->getAttr("min_fp").cast<FloatAttr>().getValue();
      APFloat maxApf = op->getAttr("max_fp").cast<FloatAttr>().getValue();
      minApf.convert(elementTy.cast<FloatType>().getFloatSemantics(),
                     APFloat::rmNearestTiesToEven, &losesInfo);
      maxApf.convert(elementTy.cast<FloatType>().getFloatSemantics(),
                     APFloat::rmNearestTiesToEven, &losesInfo);
      auto min = elementTy.materializeConstant(
          rewriter, loc, rewriter.getFloatAttr(elementTy, minApf));
      auto max = elementTy.materializeConstant(
          rewriter, loc, rewriter.getFloatAttr(elementTy, maxApf));

      return clampFloatHelper(loc, args[0], min, max, rewriter);

    } else if (elementTy.isIntegral()) {

      int32_t min = static_cast<int32_t>(
          op->getAttr("min_int").cast<IntegerAttr>().getValue().getSExtValue());
      int32_t max = static_cast<int32_t>(
          op->getAttr("max_int").cast<IntegerAttr>().getValue().getSExtValue());

      if (elementTy.isUnsignedInteger()) {
        min = std::max<int32_t>(min, 0);
        max = std::min<int32_t>(
            max,
            APInt::getMaxValue(elementTy.getWidth()).getSExtValue());
      } else {
        min = std::max<int32_t>(
            min, APInt::getSignedMinValue(elementTy.getWidth())
                     .getSExtValue());
        max = std::min<int32_t>(
            max, APInt::getSignedMaxValue(elementTy.getWidth())
                     .getSExtValue());
      }

      auto minVal = rewriter.create<arith::ConstantIntOp>(
          loc, min, elementTy.getWidth());
      auto maxVal = rewriter.create<arith::ConstantIntOp>(
          loc, max, elementTy.getWidth());
      return clampIntHelper(loc, args[0], minVal, maxVal, rewriter);
    }
  }

  // tosa::SigmoidOp
  if (isa<tosa::SigmoidOp>(op) && elementTy.isFloatingPoint()) {
    auto one =
        elementTy.materializeSpecialValue(rewriter, loc, SpecialValueId::ONE);
    auto negate = rewriter.create<arith::NegFOp>(loc, resultTypes, args[0]);
    auto exp = rewriter.create<mlir::math::ExpOp>(loc, resultTypes, negate);
    auto added = rewriter.create<arith::AddFOp>(loc, resultTypes, exp, one);
    return rewriter.create<arith::DivFOp>(loc, resultTypes, one, added);
  }

  // tosa::CastOp
  if (isa<tosa::CastOp>(op)) {
    auto srcTy = elementTy.cast<TosaStorageType>();
    auto dstTy = resultTypes.front().cast<TosaStorageType>();
    bool bitExtend = srcTy.getWidth() < dstTy.getWidth();

    if (srcTy == dstTy)
      return args.front();

    if (srcTy.isFloatingPoint() && dstTy.isFloatingPoint() && bitExtend)
      return rewriter.create<arith::ExtFOp>(loc, resultTypes, args,
                                            std::nullopt);

    if (srcTy.isFloatingPoint() && dstTy.isFloatingPoint() && !bitExtend)
      return rewriter.create<arith::TruncFOp>(loc, resultTypes, args,
                                              std::nullopt);

    // 1-bit integers need to be treated as signless.
    if (srcTy.isBool() && arith::UIToFPOp::areCastCompatible(srcTy, dstTy))
      return rewriter.create<arith::UIToFPOp>(loc, resultTypes, args,
                                              std::nullopt);

    if (srcTy.isBool() && dstTy.isIntegral() && bitExtend)
      return rewriter.create<arith::ExtUIOp>(loc, resultTypes, args,
                                             std::nullopt);

    // Unsigned integers need an unrealized cast so that they can be passed
    // to UIToFP. UItoFP requires a signless integer input.
    if (srcTy.isUnsignedInteger() && dstTy.isFloatingPoint()) {
      auto unrealizedCast =
          rewriter
              .create<UnrealizedConversionCastOp>(
                  loc, rewriter.getIntegerType(srcTy.getWidth()), args[0])
              .getResult(0);
      return rewriter.create<arith::UIToFPOp>(loc, resultTypes[0],
                                              unrealizedCast);
    }

    // All other si-to-fp conversions should be handled by SIToFP.
    if (arith::SIToFPOp::areCastCompatible(srcTy, dstTy))
      return rewriter.create<arith::SIToFPOp>(loc, resultTypes, args,
                                              std::nullopt);

    // Casting to boolean, floats need to only be checked as not-equal to zero.
    if (srcTy.isFloatingPoint() && dstTy.isBool()) {
      Value zero = elementTy.materializeSpecialValue(rewriter, loc,
                                                     SpecialValueId::ZERO);
      return rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE,
                                            args.front(), zero);
    }

    if (arith::FPToSIOp::areCastCompatible(srcTy, dstTy)) {
      // todo tricky because of this "half" value.
      auto zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32FloatAttr(0.0f));
      auto half = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32FloatAttr(0.5f));

      auto intMin = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32FloatAttr(
                   APInt::getSignedMinValue(dstTy.getWidth()).getSExtValue()));

      auto intMax = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getF32FloatAttr(
                   APInt::getSignedMaxValue(dstTy.getWidth()).getSExtValue()));

      auto added = rewriter.create<arith::AddFOp>(loc, args[0], half);
      auto subbed = rewriter.create<arith::SubFOp>(loc, args[0], half);
      auto negative = rewriter.create<arith::CmpFOp>(
          loc, arith::CmpFPredicate::OLT, args[0], zero);
      auto rounded =
          rewriter.create<arith::SelectOp>(loc, negative, subbed, added);

      auto clamped = clampFloatHelper(loc, rounded, intMin, intMax, rewriter);

      return rewriter.create<arith::FPToSIOp>(loc, dstTy, clamped);
    }

    // Casting to boolean, integers need to only be checked as not-equal to
    // zero.
    if (srcTy.isIntegral() && dstTy.isBool()) {
      Value zero = elementTy.materializeSpecialValue(rewriter, loc,
                                                     SpecialValueId::ZERO);
      return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                            args.front(), zero);
    }

    if (srcTy.isIntegral() && dstTy.isIntegral() && bitExtend)
      return rewriter.create<arith::ExtSIOp>(loc, resultTypes, args,
                                             std::nullopt);

    if (srcTy.isIntegral() && dstTy.isIntegral() && !bitExtend) {
      return rewriter.create<arith::TruncIOp>(loc, dstTy, args[0]);
    }
  }

  (void)rewriter.notifyMatchFailure(
      op, "unhandled op for linalg body calculation for elementwise op");
  return nullptr;
}

} // namespace

namespace mlir {

namespace tosa {

#include "mlir/Dialect/Tosa/IR/TosaStorageTypeInterface.cpp.inc"

void registerStorageTypeInterfaceImpls(mlir::MLIRContext *ctx) {
  // works for all integers
  IntegerType::attachInterface<IntegerStorageTypeItf>(*ctx);

  // TOSA floats, need to attach them to concrete types
  Float32Type::attachInterface<FloatStorageTypeItf>(*ctx);
  Float16Type::attachInterface<FloatStorageTypeItf>(*ctx);
  BFloat16Type::attachInterface<FloatStorageTypeItf>(*ctx);
}

} // namespace tosa

} // namespace mlir
