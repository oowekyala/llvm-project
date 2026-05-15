//===- TensorToLinalg.cpp - Tensor to Linalg Patterns ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert Tensor dialect to Linalg dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TensorToLinalg/TensorToLinalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#define DEBUG_TYPE "tensor-to-linalg-pattern"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//
namespace {

struct TensorSplatToLinalgFill : public OpRewritePattern<tensor::SplatOp> {
  using OpRewritePattern<tensor::SplatOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tensor::SplatOp splatOp,
                                PatternRewriter &rewriter) const override {

    auto empty = tensor::EmptyOp::create(rewriter, splatOp->getLoc(),
                                         splatOp.getResult().getType(),
                                         splatOp.getDynamicSizes());
    auto fill = linalg::FillOp::create(rewriter, splatOp->getLoc(),
                                       splatOp.getInput(), empty.getResult());
    rewriter.replaceOp(splatOp, fill);
    return llvm::success();
  }
};
} // namespace

void mlir::populateTensorToLinalgPatterns(RewritePatternSet &patterns) {
  // TODO: Add the remaining patterns, e.g. to decompose Pack/Unpack Ops.
  // Alternatively, delete this file.
  patterns.add<mlir::linalg::DecomposePadOpPattern, TensorSplatToLinalgFill>(patterns.getContext());
}
