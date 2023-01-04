add_mlir_dialect(TosaOps tosa)
add_mlir_doc(TosaOps TosaOps Dialects/ -gen-op-doc)
add_mlir_interface(TosaInterfaces)

set(interface TosaStorageTypeInterface)
  set(LLVM_TARGET_DEFINITIONS ${interface}.td)
  mlir_tablegen(${interface}.h.inc -gen-type-interface-decls)
  mlir_tablegen(${interface}.cpp.inc -gen-type-interface-defs)
  add_public_tablegen_target(MLIR${interface}IncGen)
  add_dependencies(mlir-generic-headers MLIR${interface}IncGen)


set(LLVM_TARGET_DEFINITIONS TosaOps.td)
mlir_tablegen(TosaAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(TosaAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRTosaAttributesIncGen)
