//===- TosaStorageTypeInterfaceImpl.h - Impl. of TosaStorageType ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// note: TosaStorageType is exported by TosaOps.h

#ifndef TOSA_STORAGE_ITF_IMPL
#define TOSA_STORAGE_ITF_IMPL


namespace mlir {
class MLIRContext;
namespace tosa {
void registerStorageTypeInterfaceImpls(MLIRContext *ctx);
} // namespace tosa
} // namespace mlir
#endif