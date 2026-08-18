// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=8 vectorize-reductions=true" -split-input-file | FileCheck %s

// The reduction runs along the minor dimension, so its successive values are
// adjacent in memory and widening the accumulator reads them with one load.
// The reduction loop itself is vectorized.
func.func @contiguous_reduction(%in: memref<128xf32>) -> f32 {
  %cst = arith.constant 0.000000e+00 : f32
  %red = affine.for %i = 0 to 128 iter_args(%acc = %cst) -> (f32) {
    %v = affine.load %in[%i] : memref<128xf32>
    %sum = arith.addf %acc, %v : f32
    affine.yield %sum : f32
  }
  return %red : f32
}

// CHECK-LABEL: func.func @contiguous_reduction
// CHECK:         affine.for %{{.*}} = 0 to 128 step 8 iter_args({{.*}}) -> (vector<8xf32>)
// CHECK:           vector.transfer_read
// CHECK:           arith.addf {{.*}} : vector<8xf32>
// CHECK:         vector.reduction <add>

// -----

// The reduction runs along the major dimension, so its successive values lie a
// whole row apart: widening the accumulator would need one scalar load per
// lane. The enclosing parallel loop, whose accesses *are* adjacent, is
// vectorized instead and the reduction stays a loop over a vector accumulator.
func.func @strided_reduction(%in: memref<4x64xf32>, %out: memref<64xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %j = 0 to 64 {
    %red = affine.for %k = 0 to 4 iter_args(%acc = %cst) -> (f32) {
      %v = affine.load %in[%k, %j] : memref<4x64xf32>
      %sum = arith.addf %acc, %v : f32
      affine.yield %sum : f32
    }
    affine.store %red, %out[%j] : memref<64xf32>
  }
  return
}

// CHECK-LABEL: func.func @strided_reduction
// CHECK:         affine.for %{{.*}} = 0 to 64 step 8 {
// CHECK:           affine.for %{{.*}} = 0 to 4 iter_args({{.*}}) -> (vector<8xf32>)
// CHECK:             vector.transfer_read {{.*}} {in_bounds = [true]} : memref<4x64xf32>, vector<8xf32>
// CHECK:             arith.addf {{.*}} : vector<8xf32>
// CHECK:           vector.transfer_write {{.*}} {in_bounds = [true]} : vector<8xf32>, memref<64xf32>
// CHECK-NOT:       vector.reduction
