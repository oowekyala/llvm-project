// RUN: mlir-opt %s -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries test-analysis-only" -split-input-file | FileCheck %s

// CHECK-LABEL: func @buffer_forwarding_conflict
func.func @buffer_forwarding_conflict(%arg0: tensor<?xf32> {bufferization.writable = true}, %arg1: index) -> (tensor<?xf32>, tensor<?xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["false", "none"]
  // Instead of allocating, share buffer with some inplace bufferization?
  %0 = tensor.empty(%arg1) : tensor<?xf32>

  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "false", "none"]
  %2 = tensor.insert_slice %1 into %arg0[0] [%arg1] [1] : tensor<?xf32> into tensor<?xf32>

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none"]
  %3 = tensor.insert_slice %1 into %arg0[42] [%arg1] [1] : tensor<?xf32> into tensor<?xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [-1, 0]
  return %2, %3 : tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @buffer_forwarding_no_conflict
func.func @buffer_forwarding_no_conflict(%arg0: tensor<?xf32> {bufferization.writable = true}, %arg1: index) -> (tensor<?xf32>, tensor<?xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none"]
  // Instead of allocating, share buffer with some inplace bufferization?
  %0 = tensor.empty(%arg1) : tensor<?xf32>

  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none"]
  %2 = tensor.insert_slice %1 into %arg0[42] [%arg1] [1] : tensor<?xf32> into tensor<?xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0, 0]
  return %2, %2 : tensor<?xf32>, tensor<?xf32>
}

// -----

// Verify that eliminate-empty-tensors crosses a collapse_shape on the backward
// path: tensor.empty (ND) → fill → collapse_shape → insert_slice (flat).
// The empty should be replaced by reshape(extract_slice(dest)).
// CHECK-LABEL: func @buffer_forwarding_through_collapse_shape
func.func @buffer_forwarding_through_collapse_shape(
    %arg0: tensor<256xi32> {bufferization.writable = true},
    %arg1: index) -> tensor<256xi32> {
  %c0 = arith.constant 0 : i32
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none"]
  //      CHECK: tensor.reshape
  %0 = tensor.empty() : tensor<16x8xi32>

  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]
  %1 = linalg.fill ins(%c0 : i32) outs(%0 : tensor<16x8xi32>) -> tensor<16x8xi32>

  %2 = tensor.collapse_shape %1 [[0, 1]] : tensor<16x8xi32> into tensor<128xi32>

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none"]
  %3 = tensor.insert_slice %2 into %arg0[%arg1] [128] [1]
         : tensor<128xi32> into tensor<256xi32>
  return %3 : tensor<256xi32>
}

// -----

// Verify that eliminate-empty-tensors crosses an expand_shape on the backward
// path: tensor.empty (flat) → fill → expand_shape → insert_slice (ND).
// The empty should be replaced by reshape(extract_slice(dest)).
// CHECK-LABEL: func @buffer_forwarding_through_expand_shape
func.func @buffer_forwarding_through_expand_shape(
    %arg0: tensor<16x8xi32> {bufferization.writable = true},
    %arg1: index) -> tensor<16x8xi32> {
  %c0 = arith.constant 0 : i32
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none"]
  //      CHECK: tensor.reshape
  %0 = tensor.empty() : tensor<128xi32>

  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]
  %1 = linalg.fill ins(%c0 : i32) outs(%0 : tensor<128xi32>) -> tensor<128xi32>

  %2 = tensor.expand_shape %1 [[0, 1]] output_shape [16, 8]
         : tensor<128xi32> into tensor<16x8xi32>

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none"]
  %3 = tensor.insert_slice %2 into %arg0[%arg1, 0] [16, 8] [1, 1]
         : tensor<16x8xi32> into tensor<16x8xi32>
  return %3 : tensor<16x8xi32>
}

// -----

// CHECK-LABEL: func @buffer_forwarding_through_two_collapse_shapes
func.func @buffer_forwarding_through_two_collapse_shapes_full_tensor(
    %arg0: tensor<128xi32> {bufferization.writable = true}) -> tensor<128xi32> {
  %c0 = arith.constant 0 : i32
  %idx = arith.constant 0 : index
  //      CHECK: tensor.reshape
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]
  %0 = tensor.empty() : tensor<4x8x4xi32>

  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]
  %1 = linalg.fill ins(%c0 : i32) outs(%0 : tensor<4x8x4xi32>) -> tensor<4x8x4xi32>

  %2 = tensor.collapse_shape %1 [[0, 1], [2]] : tensor<4x8x4xi32> into tensor<32x4xi32>
  %3 = tensor.collapse_shape %2 [[0, 1]] : tensor<32x4xi32> into tensor<128xi32>

  %4 = tensor.insert_slice %3 into %arg0[%idx] [128] [1]
         : tensor<128xi32> into tensor<128xi32>
  return %4 : tensor<128xi32>
}

// -----

// A collapse_shape followed by a cast (static→dynamic) on the path: both are
// crossed during the backward traversal. The rank change is resolved via
// getReassociationIndicesForReshape; the remaining dynamic→static mismatch on
// the extract_slice result is left for later (cast on the source side already
// handles it). The empty should be replaced by expand_shape(extract_slice(dest)).
// CHECK-LABEL: func @buffer_forwarding_through_collapse_and_cast
func.func @buffer_forwarding_through_collapse_and_cast(
    %arg0: tensor<?xi32> {bufferization.writable = true},
    %arg1: index, %arg2: index) -> tensor<?xi32> {
  %c0 = arith.constant 0 : i32
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none", "none"]
  //      CHECK: tensor.reshape
  %0 = tensor.empty() : tensor<16x8xi32>

  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]
  %1 = linalg.fill ins(%c0 : i32) outs(%0 : tensor<16x8xi32>) -> tensor<16x8xi32>

  %2 = tensor.collapse_shape %1 [[0, 1]] : tensor<16x8xi32> into tensor<128xi32>
  %3 = tensor.cast %2 : tensor<128xi32> to tensor<?xi32>

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none", "none"]
  %4 = tensor.insert_slice %3 into %arg0[%arg1] [%arg2] [1]
         : tensor<?xi32> into tensor<?xi32>
  return %4 : tensor<?xi32>
}

// -----

// CHECK-LABEL: func @buffer_forwarding_through_long_reshape_chain
func.func @buffer_forwarding_through_long_reshape_chain(
    %arg0: tensor<12x4xi32> {bufferization.writable = true},
    %arg1: index, %arg2: index) -> tensor<12x4xi32> {
  %c0 = arith.constant 0 : i32
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none"]
  //      CHECK: tensor.reshape
  %0 = tensor.empty() : tensor<3x8xi32>

  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]
  %1 = linalg.fill ins(%c0 : i32) outs(%0 : tensor<3x8xi32>) -> tensor<3x8xi32>

  %2 = tensor.collapse_shape %1 [[0, 1]] : tensor<3x8xi32> into tensor<24xi32>
  %3 = tensor.expand_shape %2 [[0, 1]] output_shape [6, 4]: tensor<24xi32> into tensor<6x4xi32>

  // %3 = tensor.cast %2 : tensor<128xi32> to tensor<?xi32>

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none"]
  %4 = tensor.insert_slice %3 into %arg0[%arg1, 0] [6, 4] [1, 1]
         : tensor<6x4xi32> into tensor<12x4xi32>
  return %4 : tensor<12x4xi32>
}

// -----

// CHECK-LABEL: func @buffer_forwarding_conflict_with_different_element_type
func.func @buffer_forwarding_conflict_with_different_element_type(%arg0: tensor<?xf32> {bufferization.writable = true}, %arg1: index) -> (tensor<?xf32>, tensor<?xf32>) {
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none"]
  %cst = arith.constant 0.000000e+00 : f32
  //      CHECK: bufferization.alloc_tensor(%arg1)
  %0 = tensor.empty(%arg1) : tensor<?xf32>

  //      CHECK: bufferization.alloc_tensor(%arg1)
  %1 = tensor.empty(%arg1) : tensor<?xbf16>

  //      CHECK: linalg.copy
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]
  %2 = linalg.copy ins(%0 : tensor<?xf32>) outs(%1 : tensor<?xbf16>) -> tensor<?xbf16>

  //      CHECK: linalg.copy
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]
  %3 = linalg.copy ins(%2 : tensor<?xbf16>) outs(%0 : tensor<?xf32>) -> tensor<?xf32>

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none"]
  %4 = tensor.insert_slice %3 into %arg0[42] [%arg1] [1] : tensor<?xf32> into tensor<?xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0, 0]
  return %4, %4 : tensor<?xf32>, tensor<?xf32>
}

