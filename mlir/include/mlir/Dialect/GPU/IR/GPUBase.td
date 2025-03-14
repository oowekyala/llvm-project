//===-- GPUBase.td - GPU dialect definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the GPU dialect
//
//===----------------------------------------------------------------------===//

#ifndef GPU_BASE
#define GPU_BASE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// GPU Dialect.
//===----------------------------------------------------------------------===//

def GPU_Dialect : Dialect {
  let name = "gpu";
  let cppNamespace = "::mlir::gpu";
  let hasOperationAttrVerify = 1;

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to annotate the modules that contain
    /// kernel modules.
    static StringRef getContainerModuleAttrName() {
      return "gpu.container_module";
    }
    /// Get the name of the attribute used to annotate external kernel
    /// functions.
    static StringRef getKernelFuncAttrName() { return "gpu.kernel"; }

    /// Returns whether the given function is a kernel function, i.e., has the
    /// 'gpu.kernel' attribute.
    static bool isKernel(Operation *op);

    /// Returns the number of workgroup (thread, block) dimensions supported in
    /// the GPU dialect.
    // TODO: consider generalizing this.
    static unsigned getNumWorkgroupDimensions() { return 3; }

    /// Returns the numeric value used to identify the workgroup memory address
    /// space.
    static AddressSpace getWorkgroupAddressSpace() { return AddressSpace::Workgroup; }

    /// Returns the numeric value used to identify the private memory address
    /// space.
    static AddressSpace getPrivateAddressSpace() { return AddressSpace::Private; }
  }];

  let dependentDialects = ["arith::ArithDialect"];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// GPU Enums.
//===----------------------------------------------------------------------===//

class GPU_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::gpu";
}
class GPU_I32EnumAttr<string mnemonic, GPU_I32Enum enumInfo> :
    EnumAttr<GPU_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

def GPU_AddressSpaceGlobal : I32EnumAttrCase<"Global", 1, "global">;
def GPU_AddressSpaceWorkgroup : I32EnumAttrCase<"Workgroup", 2, "workgroup">;
def GPU_AddressSpacePrivate : I32EnumAttrCase<"Private", 3, "private">;
def GPU_AddressSpaceEnum : GPU_I32Enum<
  "AddressSpace", "GPU address space", [
    GPU_AddressSpaceGlobal,
    GPU_AddressSpaceWorkgroup,
    GPU_AddressSpacePrivate
  ]>;

def GPU_AddressSpaceAttr :
  GPU_I32EnumAttr<"address_space", GPU_AddressSpaceEnum>;

//===----------------------------------------------------------------------===//
// GPU Types.
//===----------------------------------------------------------------------===//

def GPU_AsyncToken : DialectType<
  GPU_Dialect, CPred<"$_self.isa<::mlir::gpu::AsyncTokenType>()">, "async token type">,
             BuildableType<"mlir::gpu::AsyncTokenType::get($_builder.getContext())">;

// Predicat to check if type is gpu::MMAMatrixType.
def IsMMAMatrixTypePred : CPred<"$_self.isa<::mlir::gpu::MMAMatrixType>()">;

def GPU_MMAMatrix : DialectType<
  GPU_Dialect, IsMMAMatrixTypePred, "MMAMatrix type">;

// Memref type acceptable to gpu.subgroup_mma_{load|store}_matrix ops.
def GPU_MMAMemRef : MemRefOf<[F16, F32, VectorOfRankAndType<[1], [F16, F32]>]>;

class MMAMatrixOf<list<Type> allowedTypes> :
  ContainerType<AnyTypeOf<allowedTypes>, IsMMAMatrixTypePred,
  "$_self.cast<::mlir::gpu::MMAMatrixType>().getElementType()",
  "gpu.mma_matrix", "::mlir::gpu::MMAMatrixType">;

//===----------------------------------------------------------------------===//
// GPU Interfaces.
//===----------------------------------------------------------------------===//

def GPU_AsyncOpInterface : OpInterface<"AsyncOpInterface"> {
  let description = [{
    Interface for GPU operations that execute asynchronously on the device.

    GPU operations implementing this interface take a list of dependencies
    as `gpu.async.token` arguments and optionally return a `gpu.async.token`.

    The op doesn't start executing until all depent ops producing the async
    dependency tokens have finished executing.

    If the op returns a token, the op merely schedules the execution on the
    device and returns immediately, without waiting for the execution to
    complete. On the hand, if the op does not return a token, the op will wait
    for the execution to complete.
  }];
  let cppNamespace = "::mlir::gpu";

  let methods = [
    InterfaceMethod<[{
        Query the operands that represent async dependency tokens.
      }],
      "OperandRange", "getAsyncDependencies", (ins), [{}], [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getAsyncDependencies();
      }]
    >,
    InterfaceMethod<[{
        Adds a new token to the list of async dependencies if it is not already there.
      }],
      "void", "addAsyncDependency", (ins "Value":$token),
      [{}], [{
        if (!::llvm::is_contained(this->getAsyncDependencies(), token))
          ::mlir::gpu::addAsyncDependency(this->getOperation(), token);
      }]
    >,
    InterfaceMethod<[{
        Query the result that represents the async token to depend on.
      }],
      "Value", "getAsyncToken"
    >
  ];
}

//===----------------------------------------------------------------------===//
// GPU Attributes.
//===----------------------------------------------------------------------===//

class GPU_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<GPU_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

#endif // GPU_BASE
