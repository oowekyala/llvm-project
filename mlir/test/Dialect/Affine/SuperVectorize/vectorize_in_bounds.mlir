// RUN: mlir-opt %s -affine-super-vectorize="virtual-vector-size=8 vectorize-reductions=true" -split-input-file | FileCheck %s

// A trip count that is a multiple of the vector size, filling the dimension
// exactly: every transfer stays inside the memref.
func.func @exact_bounds(%in: memref<32xf32>, %out: memref<32xf32>) {
  affine.for %i = 0 to 32 {
    %v = affine.load %in[%i] : memref<32xf32>
    affine.store %v, %out[%i] : memref<32xf32>
  }
  return
}

// CHECK-LABEL: func.func @exact_bounds
// CHECK:         vector.transfer_read {{.*}} {in_bounds = [true]}
// CHECK:         vector.transfer_write {{.*}} {in_bounds = [true]}

// -----

// The trip count stops short of the dimension, so the last vector still lands
// inside it.
func.func @short_of_dim(%in: memref<30xf32>, %out: memref<30xf32>) {
  affine.for %i = 0 to 24 {
    %v = affine.load %in[%i] : memref<30xf32>
    affine.store %v, %out[%i] : memref<30xf32>
  }
  return
}

// CHECK-LABEL: func.func @short_of_dim
// CHECK:         vector.transfer_read {{.*}} {in_bounds = [true]}
// CHECK:         vector.transfer_write {{.*}} {in_bounds = [true]}

// -----

// 30 is not a multiple of 8: the last iteration starts at 24 and would read
// past the end, so the transfers stay possibly-out-of-bounds.
func.func @ragged_bounds(%in: memref<30xf32>, %out: memref<30xf32>) {
  affine.for %i = 0 to 30 {
    %v = affine.load %in[%i] : memref<30xf32>
    affine.store %v, %out[%i] : memref<30xf32>
  }
  return
}

// CHECK-LABEL: func.func @ragged_bounds
// CHECK-NOT:     in_bounds

// -----

// A dynamic dimension bounds nothing statically.
func.func @dynamic_dim(%in: memref<?xf32>, %out: memref<?xf32>) {
  affine.for %i = 0 to 32 {
    %v = affine.load %in[%i] : memref<?xf32>
    affine.store %v, %out[%i] : memref<?xf32>
  }
  return
}

// CHECK-LABEL: func.func @dynamic_dim
// CHECK-NOT:     in_bounds
