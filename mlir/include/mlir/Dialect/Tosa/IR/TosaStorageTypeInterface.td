//===-- TosaStorageTypeInterface.td ------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the storage type interface for the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TOSA_STORAGE_TYPE_INTERFACE
#define TOSA_STORAGE_TYPE_INTERFACE

include "mlir/IR/OpBase.td"

def TosaStorageTypeInterface : TypeInterface<"TosaStorageType"> {
  let description = [{
     Implemented by types that can be used as element types for tensors manipulated by TOSA ops.
     TOSA quantized types must use a Tosa_StorageType as the underlying carrier type.
  }];

  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Return the width of this type in bits.
        }],
        /*retType=*/"int",
        /*methodName=*/"getWidth"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return whether this type is an integer type.
        }],
        /*retType=*/"bool",
        /*methodName=*/"isIntegral"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return whether this type is a floating-point number type.
        }],
        /*retType=*/"bool",
        /*methodName=*/"isFloatingPoint"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return whether this type is signed.
        }],
        /*retType=*/"bool",
        /*methodName=*/"isSigned"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return whether this type is signless.
        }],
        /*retType=*/"bool",
        /*methodName=*/"isSignless"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return whether this type is unsigned.
        }],
        /*retType=*/"bool",
        /*methodName=*/"isUnsigned"
      >,
      InterfaceMethod<
        /*desc=*/[{
          Create an attribute that holds the given value.
        }],
        /*retType=*/"mlir::Attribute",
        /*methodName=*/"materializeAttribute",
        /*args=*/(ins "::mlir::OpBuilder&":$rewriter, "::mlir::tosa::SpecialValueId":$value)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Materialize a constant of this type from an attribute. This may use eg arith.constant or so.
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"materializeConstant",
        /*args=*/(ins "::mlir::OpBuilder&":$rewriter, "::mlir::Location":$loc, "::mlir::Attribute":$constValue)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Create a DAG that performs the given operation in the lower-level dialect, eg arith.
        }],
        /*retType=*/"::mlir::Value",
        /*methodName=*/"lowerTosaElementWiseOp",
        /*args=*/(ins "::mlir::PatternRewriter&":$rewriter, 
                      "::mlir::Operation*":$op,
                      "::mlir::ValueRange":$args,
                      "::llvm::ArrayRef<::mlir::Type>":$resultTypes
                 )
      >,
      InterfaceMethod<
        /*desc=*/[{
          Create a DAG that performs the given operation in the lower-level dialect, eg arith.
        }],
        /*retType=*/"::mlir::Value",
        /*methodName=*/"lowerTosaReductionKernel",
        /*args=*/(ins "::mlir::PatternRewriter&":$rewriter, 
                      "::mlir::Operation*":$op,
                      "::mlir::ValueRange":$args
                 )
      >
  ];

  code extraClassDeclaration = [{
    public:
     [[nodiscard]] ::mlir::Value materializeSpecialValue(::mlir::OpBuilder& rewriter, ::mlir::Location loc, ::mlir::tosa::SpecialValueId id) const {
      return materializeConstant(rewriter, loc, materializeAttribute(rewriter, id));
     }

     /// Return whether this is an integer type with a width of 1.
     [[nodiscard]] bool isBool() const {
      return isIntegral() && getWidth() == 1;
     }

     [[nodiscard]] bool isSignlessInteger() const {
      return isIntegral() && isSignless();
     }

     [[nodiscard]] std::optional<::mlir::Attribute> fitIntoAttributeOrFail(::mlir::OpBuilder& rewriter, int64_t value) const {
        if (isIntegral()) {
          auto width = getWidth();

          int64_t intMin =
              APInt::getSignedMinValue(width).getSExtValue();
          int64_t intMax =
              APInt::getSignedMaxValue(width).getSExtValue();

          if (intMin <= value && value <= intMax)
            return rewriter.getIntegerAttr(IntegerType::get(getContext(), getWidth()), value);
        }
        return {};
      }

  }];

}

#endif
