//===-- TosaOps.h - TOSA dialect operation definitions ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the TOSA Dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TOSA_IR_TOSASTORAGETYPEINTERFACE_H
#define MLIR_DIALECT_TOSA_IR_TOSASTORAGETYPEINTERFACE_H

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

//===----------------------------------------------------------------------===//
// TOSA dialect and structs includes.
//===----------------------------------------------------------------------===//

namespace mlir {

namespace tosa {

/// Identifier for a remarkable value in the value range of a TosaStorageType.
enum class SpecialValueId {
  /// Additive neuter element.
  ZERO,
  /// Multiplicative neuter element.
  ONE,
  /// Smallest value representable by this type.
  RANGE_MIN,
  /// Largest value representable by this type.
  RANGE_MAX,
  /// Value where all bits are set to one.
  ALL_ONES
};
} // namespace tosa
} // namespace mlir

#include "mlir/Dialect/Tosa/IR/TosaStorageTypeInterface.h.inc"

#endif // MLIR_DIALECT_TOSA_IR_TOSASTORAGETYPEINTERFACE_H
