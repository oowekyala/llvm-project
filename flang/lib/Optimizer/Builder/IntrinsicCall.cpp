//===-- IntrinsicCall.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper routines for constructing the FIR dialect of MLIR. As FIR is a
// dialect of MLIR, it makes extensive use of MLIR interfaces and MLIR's coding
// style (https://mlir.llvm.org/getting_started/DeveloperGuide/) is used in this
// module.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/IntrinsicCall.h"
#include "flang/Common/static-multimap-view.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/Character.h"
#include "flang/Optimizer/Builder/Complex.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "flang/Optimizer/Builder/Runtime/Allocatable.h"
#include "flang/Optimizer/Builder/Runtime/Character.h"
#include "flang/Optimizer/Builder/Runtime/Command.h"
#include "flang/Optimizer/Builder/Runtime/Derived.h"
#include "flang/Optimizer/Builder/Runtime/Inquiry.h"
#include "flang/Optimizer/Builder/Runtime/Intrinsics.h"
#include "flang/Optimizer/Builder/Runtime/Numeric.h"
#include "flang/Optimizer/Builder/Runtime/RTBuilder.h"
#include "flang/Optimizer/Builder/Runtime/Reduction.h"
#include "flang/Optimizer/Builder/Runtime/Stop.h"
#include "flang/Optimizer/Builder/Runtime/Transformational.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIROpsSupport.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Runtime/entry-names.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "flang-lower-intrinsic"

/// This file implements lowering of Fortran intrinsic procedures and Fortran
/// intrinsic module procedures.  A call may be inlined with a mix of FIR and
/// MLIR operations, or as a call to a runtime function or LLVM intrinsic.

/// Lowering of intrinsic procedure calls is based on a map that associates
/// Fortran intrinsic generic names to FIR generator functions.
/// All generator functions are member functions of the IntrinsicLibrary class
/// and have the same interface.
/// If no generator is given for an intrinsic name, a math runtime library
/// is searched for an implementation and, if a runtime function is found,
/// a call is generated for it. LLVM intrinsics are handled as a math
/// runtime library here.

/// Enums used to templatize and share lowering of MIN and MAX.
enum class Extremum { Min, Max };

// There are different ways to deal with NaNs in MIN and MAX.
// Known existing behaviors are listed below and can be selected for
// f18 MIN/MAX implementation.
enum class ExtremumBehavior {
  // Note: the Signaling/quiet aspect of NaNs in the behaviors below are
  // not described because there is no way to control/observe such aspect in
  // MLIR/LLVM yet. The IEEE behaviors come with requirements regarding this
  // aspect that are therefore currently not enforced. In the descriptions
  // below, NaNs can be signaling or quite. Returned NaNs may be signaling
  // if one of the input NaN was signaling but it cannot be guaranteed either.
  // Existing compilers using an IEEE behavior (gfortran) also do not fulfill
  // signaling/quiet requirements.
  IeeeMinMaximumNumber,
  // IEEE minimumNumber/maximumNumber behavior (754-2019, section 9.6):
  // If one of the argument is and number and the other is NaN, return the
  // number. If both arguements are NaN, return NaN.
  // Compilers: gfortran.
  IeeeMinMaximum,
  // IEEE minimum/maximum behavior (754-2019, section 9.6):
  // If one of the argument is NaN, return NaN.
  MinMaxss,
  // x86 minss/maxss behavior:
  // If the second argument is a number and the other is NaN, return the number.
  // In all other cases where at least one operand is NaN, return NaN.
  // Compilers: xlf (only for MAX), ifort, pgfortran -nollvm, and nagfor.
  PgfortranLlvm,
  // "Opposite of" x86 minss/maxss behavior:
  // If the first argument is a number and the other is NaN, return the
  // number.
  // In all other cases where at least one operand is NaN, return NaN.
  // Compilers: xlf (only for MIN), and pgfortran (with llvm).
  IeeeMinMaxNum
  // IEEE minNum/maxNum behavior (754-2008, section 5.3.1):
  // TODO: Not implemented.
  // It is the only behavior where the signaling/quiet aspect of a NaN argument
  // impacts if the result should be NaN or the argument that is a number.
  // LLVM/MLIR do not provide ways to observe this aspect, so it is not
  // possible to implement it without some target dependent runtime.
};

fir::ExtendedValue fir::getAbsentIntrinsicArgument() {
  return fir::UnboxedValue{};
}

/// Test if an ExtendedValue is absent. This is used to test if an intrinsic
/// argument are absent at compile time.
static bool isStaticallyAbsent(const fir::ExtendedValue &exv) {
  return !fir::getBase(exv);
}
static bool isStaticallyAbsent(llvm::ArrayRef<fir::ExtendedValue> args,
                               size_t argIndex) {
  return args.size() <= argIndex || isStaticallyAbsent(args[argIndex]);
}
static bool isStaticallyAbsent(llvm::ArrayRef<mlir::Value> args,
                               size_t argIndex) {
  return args.size() <= argIndex || !args[argIndex];
}

/// Test if an ExtendedValue is present. This is used to test if an intrinsic
/// argument is present at compile time. This does not imply that the related
/// value may not be an absent dummy optional, disassociated pointer, or a
/// deallocated allocatable. See `handleDynamicOptional` to deal with these
/// cases when it makes sense.
static bool isStaticallyPresent(const fir::ExtendedValue &exv) {
  return !isStaticallyAbsent(exv);
}

// TODO error handling -> return a code or directly emit messages ?
struct IntrinsicLibrary {

  // Constructors.
  explicit IntrinsicLibrary(fir::FirOpBuilder &builder, mlir::Location loc)
      : builder{builder}, loc{loc} {}
  IntrinsicLibrary() = delete;
  IntrinsicLibrary(const IntrinsicLibrary &) = delete;

  /// Generate FIR for call to Fortran intrinsic \p name with arguments \p arg
  /// and expected result type \p resultType. Return the result and a boolean
  /// that, if true, indicates that the result must be freed after use.
  std::pair<fir::ExtendedValue, bool>
  genIntrinsicCall(llvm::StringRef name, std::optional<mlir::Type> resultType,
                   llvm::ArrayRef<fir::ExtendedValue> arg);

  /// Search a runtime function that is associated to the generic intrinsic name
  /// and whose signature matches the intrinsic arguments and result types.
  /// If no such runtime function is found but a runtime function associated
  /// with the Fortran generic exists and has the same number of arguments,
  /// conversions will be inserted before and/or after the call. This is to
  /// mainly to allow 16 bits float support even-though little or no math
  /// runtime is currently available for it.
  mlir::Value genRuntimeCall(llvm::StringRef name, mlir::Type,
                             llvm::ArrayRef<mlir::Value>);

  using RuntimeCallGenerator = std::function<mlir::Value(
      fir::FirOpBuilder &, mlir::Location, llvm::ArrayRef<mlir::Value>)>;
  RuntimeCallGenerator
  getRuntimeCallGenerator(llvm::StringRef name,
                          mlir::FunctionType soughtFuncType);

  void genAbort(llvm::ArrayRef<fir::ExtendedValue>);

  /// Lowering for the ABS intrinsic. The ABS intrinsic expects one argument in
  /// the llvm::ArrayRef. The ABS intrinsic is lowered into MLIR/FIR operation
  /// if the argument is an integer, into llvm intrinsics if the argument is
  /// real and to the `hypot` math routine if the argument is of complex type.
  mlir::Value genAbs(mlir::Type, llvm::ArrayRef<mlir::Value>);
  template <void (*CallRuntime)(fir::FirOpBuilder &, mlir::Location loc,
                                mlir::Value, mlir::Value)>
  fir::ExtendedValue genAdjustRtCall(mlir::Type,
                                     llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genAimag(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genAint(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genAll(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genAllocated(mlir::Type,
                                  llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genAnint(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genAny(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue
      genCommandArgumentCount(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genAssociated(mlir::Type,
                                   llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genBesselJn(mlir::Type,
                                 llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genBesselYn(mlir::Type,
                                 llvm::ArrayRef<fir::ExtendedValue>);
  /// Lower a bitwise comparison intrinsic using the given comparator.
  template <mlir::arith::CmpIPredicate pred>
  mlir::Value genBitwiseCompare(mlir::Type resultType,
                                llvm::ArrayRef<mlir::Value> args);

  mlir::Value genBtest(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genCeiling(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genChar(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  template <mlir::arith::CmpIPredicate pred>
  fir::ExtendedValue genCharacterCompare(mlir::Type,
                                         llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genCmplx(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genConjg(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genCount(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  void genCpuTime(llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genCshift(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genCAssociatedCFunPtr(mlir::Type,
                                           llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genCAssociatedCPtr(mlir::Type,
                                        llvm::ArrayRef<fir::ExtendedValue>);
  void genCFPointer(llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genCFunLoc(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genCLoc(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  void genDateAndTime(llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genDim(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genDotProduct(mlir::Type,
                                   llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genDprod(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genDshiftl(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genDshiftr(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genEoshift(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  void genExit(llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genExponent(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genExtendsTypeOf(mlir::Type,
                                      llvm::ArrayRef<fir::ExtendedValue>);
  template <Extremum, ExtremumBehavior>
  mlir::Value genExtremum(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genFloor(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genFraction(mlir::Type resultType,
                          mlir::ArrayRef<mlir::Value> args);
  void genGetCommand(mlir::ArrayRef<fir::ExtendedValue> args);
  void genGetCommandArgument(mlir::ArrayRef<fir::ExtendedValue> args);
  void genGetEnvironmentVariable(llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genIall(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  /// Lowering for the IAND intrinsic. The IAND intrinsic expects two arguments
  /// in the llvm::ArrayRef.
  mlir::Value genIand(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genIany(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genIbclr(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIbits(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIbset(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genIchar(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genFindloc(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genIeeeIsFinite(mlir::Type, llvm::ArrayRef<mlir::Value>);
  template <mlir::arith::CmpIPredicate pred>
  fir::ExtendedValue genIeeeTypeCompare(mlir::Type,
                                        llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genIeor(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genIndex(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genIor(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genIparity(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genIsContiguous(mlir::Type,
                                     llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genIshft(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genIshftc(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genLbound(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genLeadz(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genLen(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genLenTrim(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genLoc(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  template <typename Shift>
  mlir::Value genMask(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genMatmul(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genMaxloc(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genMaxval(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genMerge(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genMergeBits(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genMinloc(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genMinval(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genMod(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genModulo(mlir::Type, llvm::ArrayRef<mlir::Value>);
  void genMoveAlloc(llvm::ArrayRef<fir::ExtendedValue>);
  void genMvbits(llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genNearest(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genNint(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genNorm2(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genNot(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genNull(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genPack(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genParity(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genPopcnt(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genPoppar(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genPresent(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genProduct(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  void genRandomInit(llvm::ArrayRef<fir::ExtendedValue>);
  void genRandomNumber(llvm::ArrayRef<fir::ExtendedValue>);
  void genRandomSeed(llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genReduce(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genRepeat(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genReshape(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genRRSpacing(mlir::Type resultType,
                           llvm::ArrayRef<mlir::Value> args);
  fir::ExtendedValue genSameTypeAs(mlir::Type,
                                   llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genScale(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genScan(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genSelectedIntKind(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genSelectedRealKind(mlir::Type, llvm::ArrayRef<mlir::Value>);
  mlir::Value genSetExponent(mlir::Type resultType,
                             llvm::ArrayRef<mlir::Value> args);
  template <typename Shift>
  mlir::Value genShift(mlir::Type resultType, llvm::ArrayRef<mlir::Value>);
  mlir::Value genShiftA(mlir::Type resultType, llvm::ArrayRef<mlir::Value>);
  mlir::Value genSign(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genSize(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genSpacing(mlir::Type resultType,
                         llvm::ArrayRef<mlir::Value> args);
  fir::ExtendedValue genSpread(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genStorageSize(mlir::Type,
                                    llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genSum(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  void genSystemClock(llvm::ArrayRef<fir::ExtendedValue>);
  mlir::Value genTrailz(mlir::Type, llvm::ArrayRef<mlir::Value>);
  fir::ExtendedValue genTransfer(mlir::Type,
                                 llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genTranspose(mlir::Type,
                                  llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genTrim(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genUbound(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genUnpack(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  fir::ExtendedValue genVerify(mlir::Type, llvm::ArrayRef<fir::ExtendedValue>);
  /// Implement all conversion functions like DBLE, the first argument is
  /// the value to convert. There may be an additional KIND arguments that
  /// is ignored because this is already reflected in the result type.
  mlir::Value genConversion(mlir::Type, llvm::ArrayRef<mlir::Value>);

  /// In the template helper below:
  ///  - "FN func" is a callback to generate the related intrinsic runtime call.
  ///  - "FD funcDim" is a callback to generate the "dim" runtime call.
  ///  - "FC funcChar" is a callback to generate the character runtime call.
  /// Helper for MinLoc/MaxLoc.
  template <typename FN, typename FD>
  fir::ExtendedValue genExtremumloc(FN func, FD funcDim, llvm::StringRef errMsg,
                                    mlir::Type,
                                    llvm::ArrayRef<fir::ExtendedValue>);
  template <typename FN, typename FD, typename FC>
  /// Helper for MinVal/MaxVal.
  fir::ExtendedValue genExtremumVal(FN func, FD funcDim, FC funcChar,
                                    llvm::StringRef errMsg,
                                    mlir::Type resultType,
                                    llvm::ArrayRef<fir::ExtendedValue> args);
  /// Process calls to Product, Sum, IAll, IAny, IParity intrinsic functions
  template <typename FN, typename FD>
  fir::ExtendedValue genReduction(FN func, FD funcDim, llvm::StringRef errMsg,
                                  mlir::Type resultType,
                                  llvm::ArrayRef<fir::ExtendedValue> args);

  /// Define the different FIR generators that can be mapped to intrinsic to
  /// generate the related code.
  using ElementalGenerator = decltype(&IntrinsicLibrary::genAbs);
  using ExtendedGenerator = decltype(&IntrinsicLibrary::genLenTrim);
  using SubroutineGenerator = decltype(&IntrinsicLibrary::genDateAndTime);
  using Generator =
      std::variant<ElementalGenerator, ExtendedGenerator, SubroutineGenerator>;

  /// All generators can be outlined. This will build a function named
  /// "fir."+ <generic name> + "." + <result type code> and generate the
  /// intrinsic implementation inside instead of at the intrinsic call sites.
  /// This can be used to keep the FIR more readable. Only one function will
  /// be generated for all the similar calls in a program.
  /// If the Generator is nullptr, the wrapper uses genRuntimeCall.
  template <typename GeneratorType>
  mlir::Value outlineInWrapper(GeneratorType, llvm::StringRef name,
                               mlir::Type resultType,
                               llvm::ArrayRef<mlir::Value> args);
  template <typename GeneratorType>
  fir::ExtendedValue
  outlineInExtendedWrapper(GeneratorType, llvm::StringRef name,
                           std::optional<mlir::Type> resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args);

  template <typename GeneratorType>
  mlir::func::FuncOp getWrapper(GeneratorType, llvm::StringRef name,
                                mlir::FunctionType,
                                bool loadRefArguments = false);

  /// Generate calls to ElementalGenerator, handling the elemental aspects
  template <typename GeneratorType>
  fir::ExtendedValue
  genElementalCall(GeneratorType, llvm::StringRef name, mlir::Type resultType,
                   llvm::ArrayRef<fir::ExtendedValue> args, bool outline);

  /// Helper to invoke code generator for the intrinsics given arguments.
  mlir::Value invokeGenerator(ElementalGenerator generator,
                              mlir::Type resultType,
                              llvm::ArrayRef<mlir::Value> args);
  mlir::Value invokeGenerator(RuntimeCallGenerator generator,
                              mlir::Type resultType,
                              llvm::ArrayRef<mlir::Value> args);
  mlir::Value invokeGenerator(ExtendedGenerator generator,
                              mlir::Type resultType,
                              llvm::ArrayRef<mlir::Value> args);
  mlir::Value invokeGenerator(SubroutineGenerator generator,
                              llvm::ArrayRef<mlir::Value> args);

  /// Get pointer to unrestricted intrinsic. Generate the related unrestricted
  /// intrinsic if it is not defined yet.
  mlir::SymbolRefAttr
  getUnrestrictedIntrinsicSymbolRefAttr(llvm::StringRef name,
                                        mlir::FunctionType signature);

  /// Helper function for generating code clean-up for result descriptors
  fir::ExtendedValue readAndAddCleanUp(fir::MutableBoxValue resultMutableBox,
                                       mlir::Type resultType,
                                       llvm::StringRef errMsg);

  void setResultMustBeFreed() { resultMustBeFreed = true; }

  fir::FirOpBuilder &builder;
  mlir::Location loc;
  bool resultMustBeFreed = false;
};

struct IntrinsicDummyArgument {
  const char *name = nullptr;
  fir::LowerIntrinsicArgAs lowerAs = fir::LowerIntrinsicArgAs::Value;
  bool handleDynamicOptional = false;
};

/// This is shared by intrinsics and intrinsic module procedures.
struct fir::IntrinsicArgumentLoweringRules {
  /// There is no more than 7 non repeated arguments in Fortran intrinsics.
  IntrinsicDummyArgument args[7];
  constexpr bool hasDefaultRules() const { return args[0].name == nullptr; }
};

/// Structure describing what needs to be done to lower intrinsic or intrinsic
/// module procedure "name".
struct IntrinsicHandler {
  const char *name;
  IntrinsicLibrary::Generator generator;
  // The following may be omitted in the table below.
  fir::IntrinsicArgumentLoweringRules argLoweringRules = {};
  bool isElemental = true;
  /// Code heavy intrinsic can be outlined to make FIR
  /// more readable.
  bool outline = false;
};

constexpr auto asValue = fir::LowerIntrinsicArgAs::Value;
constexpr auto asAddr = fir::LowerIntrinsicArgAs::Addr;
constexpr auto asBox = fir::LowerIntrinsicArgAs::Box;
constexpr auto asInquired = fir::LowerIntrinsicArgAs::Inquired;
using I = IntrinsicLibrary;

/// Flag to indicate that an intrinsic argument has to be handled as
/// being dynamically optional (e.g. special handling when actual
/// argument is an optional variable in the current scope).
static constexpr bool handleDynamicOptional = true;

/// Table that drives the fir generation depending on the intrinsic or intrinsic
/// module procedure one to one mapping with Fortran arguments. If no mapping is
/// defined here for a generic intrinsic, genRuntimeCall will be called
/// to look for a match in the runtime a emit a call. Note that the argument
/// lowering rules for an intrinsic need to be provided only if at least one
/// argument must not be lowered by value. In which case, the lowering rules
/// should be provided for all the intrinsic arguments for completeness.
static constexpr IntrinsicHandler handlers[]{
    {"abort", &I::genAbort},
    {"abs", &I::genAbs},
    {"achar", &I::genChar},
    {"adjustl",
     &I::genAdjustRtCall<fir::runtime::genAdjustL>,
     {{{"string", asAddr}}},
     /*isElemental=*/true},
    {"adjustr",
     &I::genAdjustRtCall<fir::runtime::genAdjustR>,
     {{{"string", asAddr}}},
     /*isElemental=*/true},
    {"aimag", &I::genAimag},
    {"aint", &I::genAint},
    {"all",
     &I::genAll,
     {{{"mask", asAddr}, {"dim", asValue}}},
     /*isElemental=*/false},
    {"allocated",
     &I::genAllocated,
     {{{"array", asInquired}, {"scalar", asInquired}}},
     /*isElemental=*/false},
    {"anint", &I::genAnint},
    {"any",
     &I::genAny,
     {{{"mask", asAddr}, {"dim", asValue}}},
     /*isElemental=*/false},
    {"associated",
     &I::genAssociated,
     {{{"pointer", asInquired}, {"target", asInquired}}},
     /*isElemental=*/false},
    {"bessel_jn",
     &I::genBesselJn,
     {{{"n1", asValue}, {"n2", asValue}, {"x", asValue}}},
     /*isElemental=*/false},
    {"bessel_yn",
     &I::genBesselYn,
     {{{"n1", asValue}, {"n2", asValue}, {"x", asValue}}},
     /*isElemental=*/false},
    {"bge", &I::genBitwiseCompare<mlir::arith::CmpIPredicate::uge>},
    {"bgt", &I::genBitwiseCompare<mlir::arith::CmpIPredicate::ugt>},
    {"ble", &I::genBitwiseCompare<mlir::arith::CmpIPredicate::ule>},
    {"blt", &I::genBitwiseCompare<mlir::arith::CmpIPredicate::ult>},
    {"btest", &I::genBtest},
    {"c_associated_c_funptr",
     &I::genCAssociatedCFunPtr,
     {{{"c_ptr_1", asAddr}, {"c_ptr_2", asAddr, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"c_associated_c_ptr",
     &I::genCAssociatedCPtr,
     {{{"c_ptr_1", asAddr}, {"c_ptr_2", asAddr, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"c_f_pointer",
     &I::genCFPointer,
     {{{"cptr", asValue},
       {"fptr", asInquired},
       {"shape", asAddr, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"c_funloc", &I::genCFunLoc, {{{"x", asBox}}}, /*isElemental=*/false},
    {"c_loc", &I::genCLoc, {{{"x", asBox}}}, /*isElemental=*/false},
    {"ceiling", &I::genCeiling},
    {"char", &I::genChar},
    {"cmplx",
     &I::genCmplx,
     {{{"x", asValue}, {"y", asValue, handleDynamicOptional}}}},
    {"command_argument_count", &I::genCommandArgumentCount},
    {"conjg", &I::genConjg},
    {"count",
     &I::genCount,
     {{{"mask", asAddr}, {"dim", asValue}, {"kind", asValue}}},
     /*isElemental=*/false},
    {"cpu_time",
     &I::genCpuTime,
     {{{"time", asAddr}}},
     /*isElemental=*/false},
    {"cshift",
     &I::genCshift,
     {{{"array", asAddr}, {"shift", asAddr}, {"dim", asValue}}},
     /*isElemental=*/false},
    {"date_and_time",
     &I::genDateAndTime,
     {{{"date", asAddr, handleDynamicOptional},
       {"time", asAddr, handleDynamicOptional},
       {"zone", asAddr, handleDynamicOptional},
       {"values", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"dble", &I::genConversion},
    {"dim", &I::genDim},
    {"dot_product",
     &I::genDotProduct,
     {{{"vector_a", asBox}, {"vector_b", asBox}}},
     /*isElemental=*/false},
    {"dprod", &I::genDprod},
    {"dshiftl", &I::genDshiftl},
    {"dshiftr", &I::genDshiftr},
    {"eoshift",
     &I::genEoshift,
     {{{"array", asBox},
       {"shift", asAddr},
       {"boundary", asBox, handleDynamicOptional},
       {"dim", asValue}}},
     /*isElemental=*/false},
    {"exit",
     &I::genExit,
     {{{"status", asValue, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"exponent", &I::genExponent},
    {"extends_type_of",
     &I::genExtendsTypeOf,
     {{{"a", asBox}, {"mold", asBox}}},
     /*isElemental=*/false},
    {"findloc",
     &I::genFindloc,
     {{{"array", asBox},
       {"value", asAddr},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional},
       {"kind", asValue},
       {"back", asValue, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"floor", &I::genFloor},
    {"fraction", &I::genFraction},
    {"get_command",
     &I::genGetCommand,
     {{{"command", asBox, handleDynamicOptional},
       {"length", asBox, handleDynamicOptional},
       {"status", asAddr, handleDynamicOptional},
       {"errmsg", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"get_command_argument",
     &I::genGetCommandArgument,
     {{{"number", asValue},
       {"value", asBox, handleDynamicOptional},
       {"length", asBox, handleDynamicOptional},
       {"status", asAddr, handleDynamicOptional},
       {"errmsg", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"get_environment_variable",
     &I::genGetEnvironmentVariable,
     {{{"name", asBox},
       {"value", asBox, handleDynamicOptional},
       {"length", asBox, handleDynamicOptional},
       {"status", asAddr, handleDynamicOptional},
       {"trim_name", asAddr, handleDynamicOptional},
       {"errmsg", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"iachar", &I::genIchar},
    {"iall",
     &I::genIall,
     {{{"array", asBox},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"iand", &I::genIand},
    {"iany",
     &I::genIany,
     {{{"array", asBox},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"ibclr", &I::genIbclr},
    {"ibits", &I::genIbits},
    {"ibset", &I::genIbset},
    {"ichar", &I::genIchar},
    {"ieee_class_eq", &I::genIeeeTypeCompare<mlir::arith::CmpIPredicate::eq>},
    {"ieee_class_ne", &I::genIeeeTypeCompare<mlir::arith::CmpIPredicate::ne>},
    {"ieee_is_finite", &I::genIeeeIsFinite},
    {"ieee_round_eq", &I::genIeeeTypeCompare<mlir::arith::CmpIPredicate::eq>},
    {"ieee_round_ne", &I::genIeeeTypeCompare<mlir::arith::CmpIPredicate::ne>},
    {"ieor", &I::genIeor},
    {"index",
     &I::genIndex,
     {{{"string", asAddr},
       {"substring", asAddr},
       {"back", asValue, handleDynamicOptional},
       {"kind", asValue}}}},
    {"ior", &I::genIor},
    {"iparity",
     &I::genIparity,
     {{{"array", asBox},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"is_contiguous",
     &I::genIsContiguous,
     {{{"array", asBox}}},
     /*isElemental=*/false},
    {"ishft", &I::genIshft},
    {"ishftc", &I::genIshftc},
    {"lbound",
     &I::genLbound,
     {{{"array", asInquired}, {"dim", asValue}, {"kind", asValue}}},
     /*isElemental=*/false},
    {"leadz", &I::genLeadz},
    {"len",
     &I::genLen,
     {{{"string", asInquired}, {"kind", asValue}}},
     /*isElemental=*/false},
    {"len_trim", &I::genLenTrim},
    {"lge", &I::genCharacterCompare<mlir::arith::CmpIPredicate::sge>},
    {"lgt", &I::genCharacterCompare<mlir::arith::CmpIPredicate::sgt>},
    {"lle", &I::genCharacterCompare<mlir::arith::CmpIPredicate::sle>},
    {"llt", &I::genCharacterCompare<mlir::arith::CmpIPredicate::slt>},
    {"loc", &I::genLoc, {{{"x", asBox}}}, /*isElemental=*/false},
    {"maskl", &I::genMask<mlir::arith::ShLIOp>},
    {"maskr", &I::genMask<mlir::arith::ShRUIOp>},
    {"matmul",
     &I::genMatmul,
     {{{"matrix_a", asAddr}, {"matrix_b", asAddr}}},
     /*isElemental=*/false},
    {"max", &I::genExtremum<Extremum::Max, ExtremumBehavior::MinMaxss>},
    {"maxloc",
     &I::genMaxloc,
     {{{"array", asBox},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional},
       {"kind", asValue},
       {"back", asValue, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"maxval",
     &I::genMaxval,
     {{{"array", asBox},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"merge", &I::genMerge},
    {"merge_bits", &I::genMergeBits},
    {"min", &I::genExtremum<Extremum::Min, ExtremumBehavior::MinMaxss>},
    {"minloc",
     &I::genMinloc,
     {{{"array", asBox},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional},
       {"kind", asValue},
       {"back", asValue, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"minval",
     &I::genMinval,
     {{{"array", asBox},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"mod", &I::genMod},
    {"modulo", &I::genModulo},
    {"move_alloc",
     &I::genMoveAlloc,
     {{{"from", asInquired},
       {"to", asInquired},
       {"status", asAddr, handleDynamicOptional},
       {"errMsg", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"mvbits",
     &I::genMvbits,
     {{{"from", asValue},
       {"frompos", asValue},
       {"len", asValue},
       {"to", asAddr},
       {"topos", asValue}}}},
    {"nearest", &I::genNearest},
    {"nint", &I::genNint},
    {"norm2",
     &I::genNorm2,
     {{{"array", asBox}, {"dim", asValue}}},
     /*isElemental=*/false},
    {"not", &I::genNot},
    {"null", &I::genNull, {{{"mold", asInquired}}}, /*isElemental=*/false},
    {"pack",
     &I::genPack,
     {{{"array", asBox},
       {"mask", asBox},
       {"vector", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"parity",
     &I::genParity,
     {{{"mask", asBox}, {"dim", asValue}}},
     /*isElemental=*/false},
    {"popcnt", &I::genPopcnt},
    {"poppar", &I::genPoppar},
    {"present",
     &I::genPresent,
     {{{"a", asInquired}}},
     /*isElemental=*/false},
    {"product",
     &I::genProduct,
     {{{"array", asBox},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"random_init",
     &I::genRandomInit,
     {{{"repeatable", asValue}, {"image_distinct", asValue}}},
     /*isElemental=*/false},
    {"random_number",
     &I::genRandomNumber,
     {{{"harvest", asBox}}},
     /*isElemental=*/false},
    {"random_seed",
     &I::genRandomSeed,
     {{{"size", asBox, handleDynamicOptional},
       {"put", asBox, handleDynamicOptional},
       {"get", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"reduce",
     &I::genReduce,
     {{{"array", asBox},
       {"operation", asAddr},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional},
       {"identity", asValue},
       {"ordered", asValue}}},
     /*isElemental=*/false},
    {"repeat",
     &I::genRepeat,
     {{{"string", asAddr}, {"ncopies", asValue}}},
     /*isElemental=*/false},
    {"reshape",
     &I::genReshape,
     {{{"source", asBox},
       {"shape", asBox},
       {"pad", asBox, handleDynamicOptional},
       {"order", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"rrspacing", &I::genRRSpacing},
    {"same_type_as",
     &I::genSameTypeAs,
     {{{"a", asBox}, {"b", asBox}}},
     /*isElemental=*/false},
    {"scale",
     &I::genScale,
     {{{"x", asValue}, {"i", asValue}}},
     /*isElemental=*/true},
    {"scan",
     &I::genScan,
     {{{"string", asAddr},
       {"set", asAddr},
       {"back", asValue, handleDynamicOptional},
       {"kind", asValue}}},
     /*isElemental=*/true},
    {"selected_int_kind",
     &I::genSelectedIntKind,
     {{{"scalar", asAddr}}},
     /*isElemental=*/false},
    {"selected_real_kind",
     &I::genSelectedRealKind,
     {{{"precision", asAddr, handleDynamicOptional},
       {"range", asAddr, handleDynamicOptional},
       {"radix", asAddr, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"set_exponent", &I::genSetExponent},
    {"shifta", &I::genShiftA},
    {"shiftl", &I::genShift<mlir::arith::ShLIOp>},
    {"shiftr", &I::genShift<mlir::arith::ShRUIOp>},
    {"sign", &I::genSign},
    {"size",
     &I::genSize,
     {{{"array", asBox},
       {"dim", asAddr, handleDynamicOptional},
       {"kind", asValue}}},
     /*isElemental=*/false},
    {"spacing", &I::genSpacing},
    {"spread",
     &I::genSpread,
     {{{"source", asBox}, {"dim", asValue}, {"ncopies", asValue}}},
     /*isElemental=*/false},
    {"storage_size",
     &I::genStorageSize,
     {{{"a", asInquired}, {"kind", asValue}}},
     /*isElemental=*/false},
    {"sum",
     &I::genSum,
     {{{"array", asBox},
       {"dim", asValue},
       {"mask", asBox, handleDynamicOptional}}},
     /*isElemental=*/false},
    {"system_clock",
     &I::genSystemClock,
     {{{"count", asAddr}, {"count_rate", asAddr}, {"count_max", asAddr}}},
     /*isElemental=*/false},
    {"trailz", &I::genTrailz},
    {"transfer",
     &I::genTransfer,
     {{{"source", asAddr}, {"mold", asAddr}, {"size", asValue}}},
     /*isElemental=*/false},
    {"transpose",
     &I::genTranspose,
     {{{"matrix", asAddr}}},
     /*isElemental=*/false},
    {"trim", &I::genTrim, {{{"string", asAddr}}}, /*isElemental=*/false},
    {"ubound",
     &I::genUbound,
     {{{"array", asBox}, {"dim", asValue}, {"kind", asValue}}},
     /*isElemental=*/false},
    {"unpack",
     &I::genUnpack,
     {{{"vector", asBox}, {"mask", asBox}, {"field", asBox}}},
     /*isElemental=*/false},
    {"verify",
     &I::genVerify,
     {{{"string", asAddr},
       {"set", asAddr},
       {"back", asValue, handleDynamicOptional},
       {"kind", asValue}}},
     /*isElemental=*/true},
};

static const IntrinsicHandler *findIntrinsicHandler(llvm::StringRef name) {
  auto compare = [](const IntrinsicHandler &handler, llvm::StringRef name) {
    return name.compare(handler.name) > 0;
  };
  auto result = llvm::lower_bound(handlers, name, compare);
  return result != std::end(handlers) && result->name == name ? result
                                                              : nullptr;
}

/// To make fir output more readable for debug, one can outline all intrinsic
/// implementation in wrappers (overrides the IntrinsicHandler::outline flag).
static llvm::cl::opt<bool> outlineAllIntrinsics(
    "outline-intrinsics",
    llvm::cl::desc(
        "Lower all intrinsic procedure implementation in their own functions"),
    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// Math runtime description and matching utility
//===----------------------------------------------------------------------===//

/// Command line option to modify math runtime behavior used to implement
/// intrinsics. This option applies both to early and late math-lowering modes.
enum MathRuntimeVersion { fastVersion, relaxedVersion, preciseVersion };
llvm::cl::opt<MathRuntimeVersion> mathRuntimeVersion(
    "math-runtime", llvm::cl::desc("Select math operations' runtime behavior:"),
    llvm::cl::values(
        clEnumValN(fastVersion, "fast", "use fast runtime behavior"),
        clEnumValN(relaxedVersion, "relaxed", "use relaxed runtime behavior"),
        clEnumValN(preciseVersion, "precise", "use precise runtime behavior")),
    llvm::cl::init(fastVersion));

static llvm::cl::opt<bool>
    disableMlirComplex("disable-mlir-complex",
                       llvm::cl::desc("Use libm instead of the MLIR complex "
                                      "dialect to lower complex operations"),
                       llvm::cl::init(false));

struct RuntimeFunction {
  // llvm::StringRef comparison operator are not constexpr, so use string_view.
  using Key = std::string_view;
  // Needed for implicit compare with keys.
  constexpr operator Key() const { return key; }
  Key key; // intrinsic name

  // Name of a runtime function that implements the operation.
  llvm::StringRef symbol;
  fir::runtime::FuncTypeBuilderFunc typeGenerator;
};

static mlir::FunctionType genF32F32FuncType(mlir::MLIRContext *context) {
  mlir::Type t = mlir::FloatType::getF32(context);
  return mlir::FunctionType::get(context, {t}, {t});
}

static mlir::FunctionType genF64F64FuncType(mlir::MLIRContext *context) {
  mlir::Type t = mlir::FloatType::getF64(context);
  return mlir::FunctionType::get(context, {t}, {t});
}

static mlir::FunctionType genF80F80FuncType(mlir::MLIRContext *context) {
  mlir::Type t = mlir::FloatType::getF80(context);
  return mlir::FunctionType::get(context, {t}, {t});
}

static mlir::FunctionType genF128F128FuncType(mlir::MLIRContext *context) {
  mlir::Type t = mlir::FloatType::getF128(context);
  return mlir::FunctionType::get(context, {t}, {t});
}

static mlir::FunctionType genF32F32F32FuncType(mlir::MLIRContext *context) {
  auto t = mlir::FloatType::getF32(context);
  return mlir::FunctionType::get(context, {t, t}, {t});
}

static mlir::FunctionType genF64F64F64FuncType(mlir::MLIRContext *context) {
  auto t = mlir::FloatType::getF64(context);
  return mlir::FunctionType::get(context, {t, t}, {t});
}

static mlir::FunctionType genF80F80F80FuncType(mlir::MLIRContext *context) {
  auto t = mlir::FloatType::getF80(context);
  return mlir::FunctionType::get(context, {t, t}, {t});
}

static mlir::FunctionType genF128F128F128FuncType(mlir::MLIRContext *context) {
  auto t = mlir::FloatType::getF128(context);
  return mlir::FunctionType::get(context, {t, t}, {t});
}

template <int Bits>
static mlir::FunctionType genIntF64FuncType(mlir::MLIRContext *context) {
  auto t = mlir::FloatType::getF64(context);
  auto r = mlir::IntegerType::get(context, Bits);
  return mlir::FunctionType::get(context, {t}, {r});
}

template <int Bits>
static mlir::FunctionType genIntF32FuncType(mlir::MLIRContext *context) {
  auto t = mlir::FloatType::getF32(context);
  auto r = mlir::IntegerType::get(context, Bits);
  return mlir::FunctionType::get(context, {t}, {r});
}

template <int Bits>
static mlir::FunctionType genF64F64IntFuncType(mlir::MLIRContext *context) {
  auto ftype = mlir::FloatType::getF64(context);
  auto itype = mlir::IntegerType::get(context, Bits);
  return mlir::FunctionType::get(context, {ftype, itype}, {ftype});
}

template <int Bits>
static mlir::FunctionType genF32F32IntFuncType(mlir::MLIRContext *context) {
  auto ftype = mlir::FloatType::getF32(context);
  auto itype = mlir::IntegerType::get(context, Bits);
  return mlir::FunctionType::get(context, {ftype, itype}, {ftype});
}

template <int Bits>
static mlir::FunctionType genF64IntF64FuncType(mlir::MLIRContext *context) {
  auto ftype = mlir::FloatType::getF64(context);
  auto itype = mlir::IntegerType::get(context, Bits);
  return mlir::FunctionType::get(context, {itype, ftype}, {ftype});
}

template <int Bits>
static mlir::FunctionType genF32IntF32FuncType(mlir::MLIRContext *context) {
  auto ftype = mlir::FloatType::getF32(context);
  auto itype = mlir::IntegerType::get(context, Bits);
  return mlir::FunctionType::get(context, {itype, ftype}, {ftype});
}

template <int Bits>
static mlir::FunctionType genIntIntIntFuncType(mlir::MLIRContext *context) {
  auto itype = mlir::IntegerType::get(context, Bits);
  return mlir::FunctionType::get(context, {itype, itype}, {itype});
}

template <int Kind>
static mlir::FunctionType
genComplexComplexFuncType(mlir::MLIRContext *context) {
  auto ctype = fir::ComplexType::get(context, Kind);
  return mlir::FunctionType::get(context, {ctype}, {ctype});
}

template <int Kind>
static mlir::FunctionType
genComplexComplexComplexFuncType(mlir::MLIRContext *context) {
  auto ctype = fir::ComplexType::get(context, Kind);
  return mlir::FunctionType::get(context, {ctype, ctype}, {ctype});
}

static mlir::FunctionType genF32ComplexFuncType(mlir::MLIRContext *context) {
  auto ctype = fir::ComplexType::get(context, 4);
  auto ftype = mlir::FloatType::getF32(context);
  return mlir::FunctionType::get(context, {ctype}, {ftype});
}

static mlir::FunctionType genF64ComplexFuncType(mlir::MLIRContext *context) {
  auto ctype = fir::ComplexType::get(context, 8);
  auto ftype = mlir::FloatType::getF64(context);
  return mlir::FunctionType::get(context, {ctype}, {ftype});
}

template <int Kind, int Bits>
static mlir::FunctionType
genComplexComplexIntFuncType(mlir::MLIRContext *context) {
  auto ctype = fir::ComplexType::get(context, Kind);
  auto itype = mlir::IntegerType::get(context, Bits);
  return mlir::FunctionType::get(context, {ctype, itype}, {ctype});
}

/// Callback type for generating lowering for a math operation.
using MathGeneratorTy = mlir::Value (*)(fir::FirOpBuilder &, mlir::Location,
                                        llvm::StringRef, mlir::FunctionType,
                                        llvm::ArrayRef<mlir::Value>);

struct MathOperation {
  // llvm::StringRef comparison operator are not constexpr, so use string_view.
  using Key = std::string_view;
  // Needed for implicit compare with keys.
  constexpr operator Key() const { return key; }
  // Intrinsic name.
  Key key;

  // Name of a runtime function that implements the operation.
  llvm::StringRef runtimeFunc;
  fir::runtime::FuncTypeBuilderFunc typeGenerator;

  // A callback to generate FIR for the intrinsic defined by 'key'.
  // A callback may generate either dedicated MLIR operation(s) or
  // a function call to a runtime function with name defined by
  // 'runtimeFunc'.
  MathGeneratorTy funcGenerator;
};

static mlir::Value genLibCall(fir::FirOpBuilder &builder, mlir::Location loc,
                              llvm::StringRef libFuncName,
                              mlir::FunctionType libFuncType,
                              llvm::ArrayRef<mlir::Value> args) {
  LLVM_DEBUG(llvm::dbgs() << "Generating '" << libFuncName
                          << "' call with type ";
             libFuncType.dump(); llvm::dbgs() << "\n");
  mlir::func::FuncOp funcOp =
      builder.addNamedFunction(loc, libFuncName, libFuncType);
  // TODO: ensure 'strictfp' setting on the call for "precise/strict"
  //       FP mode. Set appropriate Fast-Math Flags otherwise.
  // TODO: we should also mark as many libm function as possible
  //       with 'pure' attribute (of course, not in strict FP mode).
  auto libCall = builder.create<fir::CallOp>(loc, funcOp, args);
  LLVM_DEBUG(libCall.dump(); llvm::dbgs() << "\n");
  return libCall.getResult(0);
}

template <typename T>
static mlir::Value genMathOp(fir::FirOpBuilder &builder, mlir::Location loc,
                             llvm::StringRef mathLibFuncName,
                             mlir::FunctionType mathLibFuncType,
                             llvm::ArrayRef<mlir::Value> args) {
  // TODO: we have to annotate the math operations with flags
  //       that will allow to define FP accuracy/exception
  //       behavior per operation, so that after early multi-module
  //       MLIR inlining we can distiguish operation that were
  //       compiled with different settings.
  //       Suggestion:
  //         * For "relaxed" FP mode set all Fast-Math Flags
  //           (see "[RFC] FastMath flags support in MLIR (arith dialect)"
  //           topic at discourse.llvm.org).
  //         * For "fast" FP mode set all Fast-Math Flags except 'afn'.
  //         * For "precise/strict" FP mode generate fir.calls to libm
  //           entries and annotate them with an attribute that will
  //           end up transformed into 'strictfp' LLVM attribute (TBD).
  //           Elsewhere, "precise/strict" FP mode should also set
  //           'strictfp' for all user functions and calls so that
  //           LLVM backend does the right job.
  //         * Operations that cannot be reasonably optimized in MLIR
  //           can be also lowered to libm calls for "fast" and "relaxed"
  //           modes.
  mlir::Value result;
  if (mathRuntimeVersion == preciseVersion &&
      // Some operations do not have to be lowered as conservative
      // calls, since they do not affect strict FP behavior.
      // For example, purely integer operations like exponentiation
      // with integer operands fall into this class.
      !mathLibFuncName.empty()) {
    result = genLibCall(builder, loc, mathLibFuncName, mathLibFuncType, args);
  } else {
    LLVM_DEBUG(llvm::dbgs() << "Generating '" << mathLibFuncName
                            << "' operation with type ";
               mathLibFuncType.dump(); llvm::dbgs() << "\n");
    result = builder.create<T>(loc, args);
  }
  LLVM_DEBUG(result.dump(); llvm::dbgs() << "\n");
  return result;
}

template <typename T>
static mlir::Value genComplexMathOp(fir::FirOpBuilder &builder,
                                    mlir::Location loc,
                                    llvm::StringRef mathLibFuncName,
                                    mlir::FunctionType mathLibFuncType,
                                    llvm::ArrayRef<mlir::Value> args) {
  mlir::Value result;
  if (disableMlirComplex ||
      (mathRuntimeVersion == preciseVersion && !mathLibFuncName.empty())) {
    result = genLibCall(builder, loc, mathLibFuncName, mathLibFuncType, args);
    LLVM_DEBUG(result.dump(); llvm::dbgs() << "\n");
    return result;
  }

  LLVM_DEBUG(llvm::dbgs() << "Generating '" << mathLibFuncName
                          << "' operation with type ";
             mathLibFuncType.dump(); llvm::dbgs() << "\n");
  auto type = mathLibFuncType.getInput(0).cast<fir::ComplexType>();
  auto kind = type.getElementType().cast<fir::RealType>().getFKind();
  auto realTy = builder.getRealType(kind);
  auto mComplexTy = mlir::ComplexType::get(realTy);

  llvm::SmallVector<mlir::Value, 2> cargs;
  for (const mlir::Value &arg : args) {
    // Convert the fir.complex to a mlir::complex
    cargs.push_back(builder.createConvert(loc, mComplexTy, arg));
  }

  // Builder expects an extra return type to be provided if different to
  // the argument types for an operation
  if constexpr (T::template hasTrait<
                    mlir::OpTrait::SameOperandsAndResultType>()) {
    result = builder.create<T>(loc, cargs);
    result = builder.createConvert(loc, mathLibFuncType.getResult(0), result);
  } else {
    result = builder.create<T>(loc, realTy, cargs);
    result = builder.createConvert(loc, mathLibFuncType.getResult(0), result);
  }

  LLVM_DEBUG(result.dump(); llvm::dbgs() << "\n");
  return result;
}

/// Mapping between mathematical intrinsic operations and MLIR operations
/// of some appropriate dialect (math, complex, etc.) or libm calls.
/// TODO: support remaining Fortran math intrinsics.
///       See https://gcc.gnu.org/onlinedocs/gcc-12.1.0/gfortran/\
///       Intrinsic-Procedures.html for a reference.
static constexpr MathOperation mathOperations[] = {
    {"abs", "fabsf", genF32F32FuncType, genMathOp<mlir::math::AbsFOp>},
    {"abs", "fabs", genF64F64FuncType, genMathOp<mlir::math::AbsFOp>},
    {"abs", "llvm.fabs.f128", genF128F128FuncType,
     genMathOp<mlir::math::AbsFOp>},
    {"abs", "cabsf", genF32ComplexFuncType,
     genComplexMathOp<mlir::complex::AbsOp>},
    {"abs", "cabs", genF64ComplexFuncType,
     genComplexMathOp<mlir::complex::AbsOp>},
    {"acos", "acosf", genF32F32FuncType, genLibCall},
    {"acos", "acos", genF64F64FuncType, genLibCall},
    {"acos", "cacosf", genComplexComplexFuncType<4>, genLibCall},
    {"acos", "cacos", genComplexComplexFuncType<8>, genLibCall},
    {"acosh", "acoshf", genF32F32FuncType, genLibCall},
    {"acosh", "acosh", genF64F64FuncType, genLibCall},
    {"acosh", "cacoshf", genComplexComplexFuncType<4>, genLibCall},
    {"acosh", "cacosh", genComplexComplexFuncType<8>, genLibCall},
    // llvm.trunc behaves the same way as libm's trunc.
    {"aint", "llvm.trunc.f32", genF32F32FuncType, genLibCall},
    {"aint", "llvm.trunc.f64", genF64F64FuncType, genLibCall},
    {"aint", "llvm.trunc.f80", genF80F80FuncType, genLibCall},
    // llvm.round behaves the same way as libm's round.
    {"anint", "llvm.round.f32", genF32F32FuncType,
     genMathOp<mlir::LLVM::RoundOp>},
    {"anint", "llvm.round.f64", genF64F64FuncType,
     genMathOp<mlir::LLVM::RoundOp>},
    {"anint", "llvm.round.f80", genF80F80FuncType,
     genMathOp<mlir::LLVM::RoundOp>},
    {"asin", "asinf", genF32F32FuncType, genLibCall},
    {"asin", "asin", genF64F64FuncType, genLibCall},
    {"asin", "casinf", genComplexComplexFuncType<4>, genLibCall},
    {"asin", "casin", genComplexComplexFuncType<8>, genLibCall},
    {"asinh", "asinhf", genF32F32FuncType, genLibCall},
    {"asinh", "asinh", genF64F64FuncType, genLibCall},
    {"asinh", "casinhf", genComplexComplexFuncType<4>, genLibCall},
    {"asinh", "casinh", genComplexComplexFuncType<8>, genLibCall},
    {"atan", "atanf", genF32F32FuncType, genMathOp<mlir::math::AtanOp>},
    {"atan", "atan", genF64F64FuncType, genMathOp<mlir::math::AtanOp>},
    {"atan", "catanf", genComplexComplexFuncType<4>, genLibCall},
    {"atan", "catan", genComplexComplexFuncType<8>, genLibCall},
    {"atan2", "atan2f", genF32F32F32FuncType, genMathOp<mlir::math::Atan2Op>},
    {"atan2", "atan2", genF64F64F64FuncType, genMathOp<mlir::math::Atan2Op>},
    {"atanh", "atanhf", genF32F32FuncType, genLibCall},
    {"atanh", "atanh", genF64F64FuncType, genLibCall},
    {"atanh", "catanhf", genComplexComplexFuncType<4>, genLibCall},
    {"atanh", "catanh", genComplexComplexFuncType<8>, genLibCall},
    {"bessel_j0", "j0f", genF32F32FuncType, genLibCall},
    {"bessel_j0", "j0", genF64F64FuncType, genLibCall},
    {"bessel_j1", "j1f", genF32F32FuncType, genLibCall},
    {"bessel_j1", "j1", genF64F64FuncType, genLibCall},
    {"bessel_jn", "jnf", genF32IntF32FuncType<32>, genLibCall},
    {"bessel_jn", "jn", genF64IntF64FuncType<32>, genLibCall},
    {"bessel_y0", "y0f", genF32F32FuncType, genLibCall},
    {"bessel_y0", "y0", genF64F64FuncType, genLibCall},
    {"bessel_y1", "y1f", genF32F32FuncType, genLibCall},
    {"bessel_y1", "y1", genF64F64FuncType, genLibCall},
    {"bessel_yn", "ynf", genF32IntF32FuncType<32>, genLibCall},
    {"bessel_yn", "yn", genF64IntF64FuncType<32>, genLibCall},
    // math::CeilOp returns a real, while Fortran CEILING returns integer.
    {"ceil", "ceilf", genF32F32FuncType, genMathOp<mlir::math::CeilOp>},
    {"ceil", "ceil", genF64F64FuncType, genMathOp<mlir::math::CeilOp>},
    {"cos", "cosf", genF32F32FuncType, genMathOp<mlir::math::CosOp>},
    {"cos", "cos", genF64F64FuncType, genMathOp<mlir::math::CosOp>},
    {"cos", "ccosf", genComplexComplexFuncType<4>,
     genComplexMathOp<mlir::complex::CosOp>},
    {"cos", "ccos", genComplexComplexFuncType<8>,
     genComplexMathOp<mlir::complex::CosOp>},
    {"cosh", "coshf", genF32F32FuncType, genLibCall},
    {"cosh", "cosh", genF64F64FuncType, genLibCall},
    {"cosh", "ccoshf", genComplexComplexFuncType<4>, genLibCall},
    {"cosh", "ccosh", genComplexComplexFuncType<8>, genLibCall},
    {"erf", "erff", genF32F32FuncType, genMathOp<mlir::math::ErfOp>},
    {"erf", "erf", genF64F64FuncType, genMathOp<mlir::math::ErfOp>},
    {"erfc", "erfcf", genF32F32FuncType, genLibCall},
    {"erfc", "erfc", genF64F64FuncType, genLibCall},
    {"exp", "expf", genF32F32FuncType, genMathOp<mlir::math::ExpOp>},
    {"exp", "exp", genF64F64FuncType, genMathOp<mlir::math::ExpOp>},
    {"exp", "cexpf", genComplexComplexFuncType<4>,
     genComplexMathOp<mlir::complex::ExpOp>},
    {"exp", "cexp", genComplexComplexFuncType<8>,
     genComplexMathOp<mlir::complex::ExpOp>},
    // math::FloorOp returns a real, while Fortran FLOOR returns integer.
    {"floor", "floorf", genF32F32FuncType, genMathOp<mlir::math::FloorOp>},
    {"floor", "floor", genF64F64FuncType, genMathOp<mlir::math::FloorOp>},
    {"gamma", "tgammaf", genF32F32FuncType, genLibCall},
    {"gamma", "tgamma", genF64F64FuncType, genLibCall},
    {"hypot", "hypotf", genF32F32F32FuncType, genLibCall},
    {"hypot", "hypot", genF64F64F64FuncType, genLibCall},
    {"log", "logf", genF32F32FuncType, genMathOp<mlir::math::LogOp>},
    {"log", "log", genF64F64FuncType, genMathOp<mlir::math::LogOp>},
    {"log", "clogf", genComplexComplexFuncType<4>,
     genComplexMathOp<mlir::complex::LogOp>},
    {"log", "clog", genComplexComplexFuncType<8>,
     genComplexMathOp<mlir::complex::LogOp>},
    {"log10", "log10f", genF32F32FuncType, genMathOp<mlir::math::Log10Op>},
    {"log10", "log10", genF64F64FuncType, genMathOp<mlir::math::Log10Op>},
    {"log_gamma", "lgammaf", genF32F32FuncType, genLibCall},
    {"log_gamma", "lgamma", genF64F64FuncType, genLibCall},
    // llvm.lround behaves the same way as libm's lround.
    {"nint", "llvm.lround.i64.f64", genIntF64FuncType<64>, genLibCall},
    {"nint", "llvm.lround.i64.f32", genIntF32FuncType<64>, genLibCall},
    {"nint", "llvm.lround.i32.f64", genIntF64FuncType<32>, genLibCall},
    {"nint", "llvm.lround.i32.f32", genIntF32FuncType<32>, genLibCall},
    {"pow", {}, genIntIntIntFuncType<8>, genMathOp<mlir::math::IPowIOp>},
    {"pow", {}, genIntIntIntFuncType<16>, genMathOp<mlir::math::IPowIOp>},
    {"pow", {}, genIntIntIntFuncType<32>, genMathOp<mlir::math::IPowIOp>},
    {"pow", {}, genIntIntIntFuncType<64>, genMathOp<mlir::math::IPowIOp>},
    {"pow", "powf", genF32F32F32FuncType, genMathOp<mlir::math::PowFOp>},
    {"pow", "pow", genF64F64F64FuncType, genMathOp<mlir::math::PowFOp>},
    {"pow", "cpowf", genComplexComplexComplexFuncType<4>,
     genComplexMathOp<mlir::complex::PowOp>},
    {"pow", "cpow", genComplexComplexComplexFuncType<8>,
     genComplexMathOp<mlir::complex::PowOp>},
    {"pow", RTNAME_STRING(FPow4i), genF32F32IntFuncType<32>,
     genMathOp<mlir::math::FPowIOp>},
    {"pow", RTNAME_STRING(FPow8i), genF64F64IntFuncType<32>,
     genMathOp<mlir::math::FPowIOp>},
    {"pow", RTNAME_STRING(FPow4k), genF32F32IntFuncType<64>,
     genMathOp<mlir::math::FPowIOp>},
    {"pow", RTNAME_STRING(FPow8k), genF64F64IntFuncType<64>,
     genMathOp<mlir::math::FPowIOp>},
    {"pow", RTNAME_STRING(cpowi), genComplexComplexIntFuncType<4, 32>,
     genLibCall},
    {"pow", RTNAME_STRING(zpowi), genComplexComplexIntFuncType<8, 32>,
     genLibCall},
    {"pow", RTNAME_STRING(cpowk), genComplexComplexIntFuncType<4, 64>,
     genLibCall},
    {"pow", RTNAME_STRING(zpowk), genComplexComplexIntFuncType<8, 64>,
     genLibCall},
    {"sign", "copysignf", genF32F32F32FuncType,
     genMathOp<mlir::math::CopySignOp>},
    {"sign", "copysign", genF64F64F64FuncType,
     genMathOp<mlir::math::CopySignOp>},
    {"sign", "copysignl", genF80F80F80FuncType,
     genMathOp<mlir::math::CopySignOp>},
    {"sign", "llvm.copysign.f128", genF128F128F128FuncType,
     genMathOp<mlir::math::CopySignOp>},
    {"sin", "sinf", genF32F32FuncType, genMathOp<mlir::math::SinOp>},
    {"sin", "sin", genF64F64FuncType, genMathOp<mlir::math::SinOp>},
    {"sin", "csinf", genComplexComplexFuncType<4>,
     genComplexMathOp<mlir::complex::SinOp>},
    {"sin", "csin", genComplexComplexFuncType<8>,
     genComplexMathOp<mlir::complex::SinOp>},
    {"sinh", "sinhf", genF32F32FuncType, genLibCall},
    {"sinh", "sinh", genF64F64FuncType, genLibCall},
    {"sinh", "csinhf", genComplexComplexFuncType<4>, genLibCall},
    {"sinh", "csinh", genComplexComplexFuncType<8>, genLibCall},
    {"sqrt", "sqrtf", genF32F32FuncType, genMathOp<mlir::math::SqrtOp>},
    {"sqrt", "sqrt", genF64F64FuncType, genMathOp<mlir::math::SqrtOp>},
    {"sqrt", "csqrtf", genComplexComplexFuncType<4>,
     genComplexMathOp<mlir::complex::SqrtOp>},
    {"sqrt", "csqrt", genComplexComplexFuncType<8>,
     genComplexMathOp<mlir::complex::SqrtOp>},
    {"tan", "tanf", genF32F32FuncType, genMathOp<mlir::math::TanOp>},
    {"tan", "tan", genF64F64FuncType, genMathOp<mlir::math::TanOp>},
    {"tan", "ctanf", genComplexComplexFuncType<4>,
     genComplexMathOp<mlir::complex::TanOp>},
    {"tan", "ctan", genComplexComplexFuncType<8>,
     genComplexMathOp<mlir::complex::TanOp>},
    {"tanh", "tanhf", genF32F32FuncType, genMathOp<mlir::math::TanhOp>},
    {"tanh", "tanh", genF64F64FuncType, genMathOp<mlir::math::TanhOp>},
    {"tanh", "ctanhf", genComplexComplexFuncType<4>,
     genComplexMathOp<mlir::complex::TanhOp>},
    {"tanh", "ctanh", genComplexComplexFuncType<8>,
     genComplexMathOp<mlir::complex::TanhOp>},
};

// This helper class computes a "distance" between two function types.
// The distance measures how many narrowing conversions of actual arguments
// and result of "from" must be made in order to use "to" instead of "from".
// For instance, the distance between ACOS(REAL(10)) and ACOS(REAL(8)) is
// greater than the one between ACOS(REAL(10)) and ACOS(REAL(16)). This means
// if no implementation of ACOS(REAL(10)) is available, it is better to use
// ACOS(REAL(16)) with casts rather than ACOS(REAL(8)).
// Note that this is not a symmetric distance and the order of "from" and "to"
// arguments matters, d(foo, bar) may not be the same as d(bar, foo) because it
// may be safe to replace foo by bar, but not the opposite.
class FunctionDistance {
public:
  FunctionDistance() : infinite{true} {}

  FunctionDistance(mlir::FunctionType from, mlir::FunctionType to) {
    unsigned nInputs = from.getNumInputs();
    unsigned nResults = from.getNumResults();
    if (nResults != to.getNumResults() || nInputs != to.getNumInputs()) {
      infinite = true;
    } else {
      for (decltype(nInputs) i = 0; i < nInputs && !infinite; ++i)
        addArgumentDistance(from.getInput(i), to.getInput(i));
      for (decltype(nResults) i = 0; i < nResults && !infinite; ++i)
        addResultDistance(to.getResult(i), from.getResult(i));
    }
  }

  /// Beware both d1.isSmallerThan(d2) *and* d2.isSmallerThan(d1) may be
  /// false if both d1 and d2 are infinite. This implies that
  ///  d1.isSmallerThan(d2) is not equivalent to !d2.isSmallerThan(d1)
  bool isSmallerThan(const FunctionDistance &d) const {
    return !infinite &&
           (d.infinite || std::lexicographical_compare(
                              conversions.begin(), conversions.end(),
                              d.conversions.begin(), d.conversions.end()));
  }

  bool isLosingPrecision() const {
    return conversions[narrowingArg] != 0 || conversions[extendingResult] != 0;
  }

  bool isInfinite() const { return infinite; }

private:
  enum class Conversion { Forbidden, None, Narrow, Extend };

  void addArgumentDistance(mlir::Type from, mlir::Type to) {
    switch (conversionBetweenTypes(from, to)) {
    case Conversion::Forbidden:
      infinite = true;
      break;
    case Conversion::None:
      break;
    case Conversion::Narrow:
      conversions[narrowingArg]++;
      break;
    case Conversion::Extend:
      conversions[nonNarrowingArg]++;
      break;
    }
  }

  void addResultDistance(mlir::Type from, mlir::Type to) {
    switch (conversionBetweenTypes(from, to)) {
    case Conversion::Forbidden:
      infinite = true;
      break;
    case Conversion::None:
      break;
    case Conversion::Narrow:
      conversions[nonExtendingResult]++;
      break;
    case Conversion::Extend:
      conversions[extendingResult]++;
      break;
    }
  }

  // Floating point can be mlir::FloatType or fir::real
  static unsigned getFloatingPointWidth(mlir::Type t) {
    if (auto f{t.dyn_cast<mlir::FloatType>()})
      return f.getWidth();
    // FIXME: Get width another way for fir.real/complex
    // - use fir/KindMapping.h and llvm::Type
    // - or use evaluate/type.h
    if (auto r{t.dyn_cast<fir::RealType>()})
      return r.getFKind() * 4;
    if (auto cplx{t.dyn_cast<fir::ComplexType>()})
      return cplx.getFKind() * 4;
    llvm_unreachable("not a floating-point type");
  }

  static Conversion conversionBetweenTypes(mlir::Type from, mlir::Type to) {
    if (from == to)
      return Conversion::None;

    if (auto fromIntTy{from.dyn_cast<mlir::IntegerType>()}) {
      if (auto toIntTy{to.dyn_cast<mlir::IntegerType>()}) {
        return fromIntTy.getWidth() > toIntTy.getWidth() ? Conversion::Narrow
                                                         : Conversion::Extend;
      }
    }

    if (fir::isa_real(from) && fir::isa_real(to)) {
      return getFloatingPointWidth(from) > getFloatingPointWidth(to)
                 ? Conversion::Narrow
                 : Conversion::Extend;
    }

    if (auto fromCplxTy{from.dyn_cast<fir::ComplexType>()}) {
      if (auto toCplxTy{to.dyn_cast<fir::ComplexType>()}) {
        return getFloatingPointWidth(fromCplxTy) >
                       getFloatingPointWidth(toCplxTy)
                   ? Conversion::Narrow
                   : Conversion::Extend;
      }
    }
    // Notes:
    // - No conversion between character types, specialization of runtime
    // functions should be made instead.
    // - It is not clear there is a use case for automatic conversions
    // around Logical and it may damage hidden information in the physical
    // storage so do not do it.
    return Conversion::Forbidden;
  }

  // Below are indexes to access data in conversions.
  // The order in data does matter for lexicographical_compare
  enum {
    narrowingArg = 0,   // usually bad
    extendingResult,    // usually bad
    nonExtendingResult, // usually ok
    nonNarrowingArg,    // usually ok
    dataSize
  };

  std::array<int, dataSize> conversions = {};
  bool infinite = false; // When forbidden conversion or wrong argument number
};

using RtMap = Fortran::common::StaticMultimapView<MathOperation>;
static constexpr RtMap mathOps(mathOperations);
static_assert(mathOps.Verify() && "map must be sorted");

/// Look for a MathOperation entry specifying how to lower a mathematical
/// operation defined by \p name with its result' and operands' types
/// specified in the form of a FunctionType \p funcType.
/// If exact match for the given types is found, then the function
/// returns a pointer to the corresponding MathOperation.
/// Otherwise, the function returns nullptr.
/// If there is a MathOperation that can be used with additional
/// type casts for the operands or/and result (non-exact match),
/// then it is returned via \p bestNearMatch argument, and
/// \p bestMatchDistance specifies the FunctionDistance between
/// the requested operation and the non-exact match.
static const MathOperation *
searchMathOperation(fir::FirOpBuilder &builder, llvm::StringRef name,
                    mlir::FunctionType funcType,
                    const MathOperation **bestNearMatch,
                    FunctionDistance &bestMatchDistance) {
  auto range = mathOps.equal_range(name);
  for (auto iter = range.first; iter != range.second && iter; ++iter) {
    const auto &impl = *iter;
    auto implType = impl.typeGenerator(builder.getContext());
    if (funcType == implType)
      return &impl; // exact match

    FunctionDistance distance(funcType, implType);
    if (distance.isSmallerThan(bestMatchDistance)) {
      *bestNearMatch = &impl;
      bestMatchDistance = std::move(distance);
    }
  }
  return nullptr;
}

/// Implementation of the operation defined by \p name with type
/// \p funcType is not precise, and the actual available implementation
/// is \p distance away from the requested. If using the available
/// implementation results in a precision loss, emit an error message
/// with the given code location \p loc.
static void checkPrecisionLoss(llvm::StringRef name,
                               mlir::FunctionType funcType,
                               const FunctionDistance &distance,
                               mlir::Location loc) {
  if (!distance.isLosingPrecision())
    return;

  // Using this runtime version requires narrowing the arguments
  // or extending the result. It is not numerically safe. There
  // is currently no quad math library that was described in
  // lowering and could be used here. Emit an error and continue
  // generating the code with the narrowing cast so that the user
  // can get a complete list of the problematic intrinsic calls.
  std::string message("not yet implemented: no math runtime available for '");
  llvm::raw_string_ostream sstream(message);
  if (name == "pow") {
    assert(funcType.getNumInputs() == 2 && "power operator has two arguments");
    sstream << funcType.getInput(0) << " ** " << funcType.getInput(1);
  } else {
    sstream << name << "(";
    if (funcType.getNumInputs() > 0)
      sstream << funcType.getInput(0);
    for (mlir::Type argType : funcType.getInputs().drop_front())
      sstream << ", " << argType;
    sstream << ")";
  }
  sstream << "'";
  mlir::emitError(loc, message);
}

/// Helpers to get function type from arguments and result type.
static mlir::FunctionType getFunctionType(std::optional<mlir::Type> resultType,
                                          llvm::ArrayRef<mlir::Value> arguments,
                                          fir::FirOpBuilder &builder) {
  llvm::SmallVector<mlir::Type> argTypes;
  for (mlir::Value arg : arguments)
    argTypes.push_back(arg.getType());
  llvm::SmallVector<mlir::Type> resTypes;
  if (resultType)
    resTypes.push_back(*resultType);
  return mlir::FunctionType::get(builder.getModule().getContext(), argTypes,
                                 resTypes);
}

/// fir::ExtendedValue to mlir::Value translation layer

fir::ExtendedValue toExtendedValue(mlir::Value val, fir::FirOpBuilder &builder,
                                   mlir::Location loc) {
  assert(val && "optional unhandled here");
  mlir::Type type = val.getType();
  mlir::Value base = val;
  mlir::IndexType indexType = builder.getIndexType();
  llvm::SmallVector<mlir::Value> extents;

  fir::factory::CharacterExprHelper charHelper{builder, loc};
  // FIXME: we may want to allow non character scalar here.
  if (charHelper.isCharacterScalar(type))
    return charHelper.toExtendedValue(val);

  if (auto refType = type.dyn_cast<fir::ReferenceType>())
    type = refType.getEleTy();

  if (auto arrayType = type.dyn_cast<fir::SequenceType>()) {
    type = arrayType.getEleTy();
    for (fir::SequenceType::Extent extent : arrayType.getShape()) {
      if (extent == fir::SequenceType::getUnknownExtent())
        break;
      extents.emplace_back(
          builder.createIntegerConstant(loc, indexType, extent));
    }
    // Last extent might be missing in case of assumed-size. If more extents
    // could not be deduced from type, that's an error (a fir.box should
    // have been used in the interface).
    if (extents.size() + 1 < arrayType.getShape().size())
      mlir::emitError(loc, "cannot retrieve array extents from type");
  } else if (type.isa<fir::BoxType>() || type.isa<fir::RecordType>()) {
    fir::emitFatalError(loc, "not yet implemented: descriptor or derived type");
  }

  if (!extents.empty())
    return fir::ArrayBoxValue{base, extents};
  return base;
}

mlir::Value toValue(const fir::ExtendedValue &val, fir::FirOpBuilder &builder,
                    mlir::Location loc) {
  if (const fir::CharBoxValue *charBox = val.getCharBox()) {
    mlir::Value buffer = charBox->getBuffer();
    auto buffTy = buffer.getType();
    if (buffTy.isa<mlir::FunctionType>())
      fir::emitFatalError(
          loc, "A character's buffer type cannot be a function type.");
    if (buffTy.isa<fir::BoxCharType>())
      return buffer;
    return fir::factory::CharacterExprHelper{builder, loc}.createEmboxChar(
        buffer, charBox->getLen());
  }

  // FIXME: need to access other ExtendedValue variants and handle them
  // properly.
  return fir::getBase(val);
}

//===----------------------------------------------------------------------===//
// IntrinsicLibrary
//===----------------------------------------------------------------------===//

static bool isIntrinsicModuleProcedure(llvm::StringRef name) {
  return name.startswith("c_") || name.startswith("compiler_") ||
         name.startswith("ieee_");
}

/// Return the generic name of an intrinsic module procedure specific name.
/// Remove any "__builtin_" prefix, and any specific suffix of the form
/// {_[ail]?[0-9]+}*, such as _1 or _a4.
llvm::StringRef genericName(llvm::StringRef specificName) {
  const std::string builtin = "__builtin_";
  llvm::StringRef name = specificName.startswith(builtin)
                             ? specificName.drop_front(builtin.size())
                             : specificName;
  size_t size = name.size();
  if (isIntrinsicModuleProcedure(name))
    while (isdigit(name[size - 1]))
      while (name[--size] != '_')
        ;
  return name.drop_back(name.size() - size);
}

/// Generate a TODO error message for an as yet unimplemented intrinsic.
void crashOnMissingIntrinsic(mlir::Location loc, llvm::StringRef name) {
  if (isIntrinsicModuleProcedure(name))
    TODO(loc, "intrinsic module procedure: " + llvm::Twine(name));
  else
    TODO(loc, "intrinsic: " + llvm::Twine(name));
}

template <typename GeneratorType>
fir::ExtendedValue IntrinsicLibrary::genElementalCall(
    GeneratorType generator, llvm::StringRef name, mlir::Type resultType,
    llvm::ArrayRef<fir::ExtendedValue> args, bool outline) {
  llvm::SmallVector<mlir::Value> scalarArgs;
  for (const fir::ExtendedValue &arg : args)
    if (arg.getUnboxed() || arg.getCharBox())
      scalarArgs.emplace_back(fir::getBase(arg));
    else
      fir::emitFatalError(loc, "nonscalar intrinsic argument");
  if (outline)
    return outlineInWrapper(generator, name, resultType, scalarArgs);
  return invokeGenerator(generator, resultType, scalarArgs);
}

template <>
fir::ExtendedValue
IntrinsicLibrary::genElementalCall<IntrinsicLibrary::ExtendedGenerator>(
    ExtendedGenerator generator, llvm::StringRef name, mlir::Type resultType,
    llvm::ArrayRef<fir::ExtendedValue> args, bool outline) {
  for (const fir::ExtendedValue &arg : args)
    if (!arg.getUnboxed() && !arg.getCharBox())
      fir::emitFatalError(loc, "nonscalar intrinsic argument");
  if (outline)
    return outlineInExtendedWrapper(generator, name, resultType, args);
  return std::invoke(generator, *this, resultType, args);
}

template <>
fir::ExtendedValue
IntrinsicLibrary::genElementalCall<IntrinsicLibrary::SubroutineGenerator>(
    SubroutineGenerator generator, llvm::StringRef name, mlir::Type resultType,
    llvm::ArrayRef<fir::ExtendedValue> args, bool outline) {
  for (const fir::ExtendedValue &arg : args)
    if (!arg.getUnboxed() && !arg.getCharBox())
      // fir::emitFatalError(loc, "nonscalar intrinsic argument");
      crashOnMissingIntrinsic(loc, name);
  if (outline)
    return outlineInExtendedWrapper(generator, name, resultType, args);
  std::invoke(generator, *this, args);
  return mlir::Value();
}

static fir::ExtendedValue
invokeHandler(IntrinsicLibrary::ElementalGenerator generator,
              const IntrinsicHandler &handler,
              std::optional<mlir::Type> resultType,
              llvm::ArrayRef<fir::ExtendedValue> args, bool outline,
              IntrinsicLibrary &lib) {
  assert(resultType && "expect elemental intrinsic to be functions");
  return lib.genElementalCall(generator, handler.name, *resultType, args,
                              outline);
}

static fir::ExtendedValue
invokeHandler(IntrinsicLibrary::ExtendedGenerator generator,
              const IntrinsicHandler &handler,
              std::optional<mlir::Type> resultType,
              llvm::ArrayRef<fir::ExtendedValue> args, bool outline,
              IntrinsicLibrary &lib) {
  assert(resultType && "expect intrinsic function");
  if (handler.isElemental)
    return lib.genElementalCall(generator, handler.name, *resultType, args,
                                outline);
  if (outline)
    return lib.outlineInExtendedWrapper(generator, handler.name, *resultType,
                                        args);
  return std::invoke(generator, lib, *resultType, args);
}

static fir::ExtendedValue
invokeHandler(IntrinsicLibrary::SubroutineGenerator generator,
              const IntrinsicHandler &handler,
              std::optional<mlir::Type> resultType,
              llvm::ArrayRef<fir::ExtendedValue> args, bool outline,
              IntrinsicLibrary &lib) {
  if (handler.isElemental)
    return lib.genElementalCall(generator, handler.name, mlir::Type{}, args,
                                outline);
  if (outline)
    return lib.outlineInExtendedWrapper(generator, handler.name, resultType,
                                        args);
  std::invoke(generator, lib, args);
  return mlir::Value{};
}

std::pair<fir::ExtendedValue, bool>
IntrinsicLibrary::genIntrinsicCall(llvm::StringRef specificName,
                                   std::optional<mlir::Type> resultType,
                                   llvm::ArrayRef<fir::ExtendedValue> args) {
  llvm::StringRef name = genericName(specificName);
  if (const IntrinsicHandler *handler = findIntrinsicHandler(name)) {
    bool outline = handler->outline || outlineAllIntrinsics;
    return {std::visit(
                [&](auto &generator) -> fir::ExtendedValue {
                  return invokeHandler(generator, *handler, resultType, args,
                                       outline, *this);
                },
                handler->generator),
            this->resultMustBeFreed};
  }

  if (!resultType)
    // Subroutine should have a handler, they are likely missing for now.
    crashOnMissingIntrinsic(loc, name);

  // Try the runtime if no special handler was defined for the
  // intrinsic being called. Maths runtime only has numerical elemental.
  // No optional arguments are expected at this point, the code will
  // crash if it gets absent optional.

  // FIXME: using toValue to get the type won't work with array arguments.
  llvm::SmallVector<mlir::Value> mlirArgs;
  for (const fir::ExtendedValue &extendedVal : args) {
    mlir::Value val = toValue(extendedVal, builder, loc);
    if (!val)
      // If an absent optional gets there, most likely its handler has just
      // not yet been defined.
      crashOnMissingIntrinsic(loc, name);
    mlirArgs.emplace_back(val);
  }
  mlir::FunctionType soughtFuncType =
      getFunctionType(*resultType, mlirArgs, builder);

  IntrinsicLibrary::RuntimeCallGenerator runtimeCallGenerator =
      getRuntimeCallGenerator(name, soughtFuncType);
  return {genElementalCall(runtimeCallGenerator, name, *resultType, args,
                           /*outline=*/outlineAllIntrinsics),
          resultMustBeFreed};
}

mlir::Value
IntrinsicLibrary::invokeGenerator(ElementalGenerator generator,
                                  mlir::Type resultType,
                                  llvm::ArrayRef<mlir::Value> args) {
  return std::invoke(generator, *this, resultType, args);
}

mlir::Value
IntrinsicLibrary::invokeGenerator(RuntimeCallGenerator generator,
                                  mlir::Type resultType,
                                  llvm::ArrayRef<mlir::Value> args) {
  return generator(builder, loc, args);
}

mlir::Value
IntrinsicLibrary::invokeGenerator(ExtendedGenerator generator,
                                  mlir::Type resultType,
                                  llvm::ArrayRef<mlir::Value> args) {
  llvm::SmallVector<fir::ExtendedValue> extendedArgs;
  for (mlir::Value arg : args)
    extendedArgs.emplace_back(toExtendedValue(arg, builder, loc));
  auto extendedResult = std::invoke(generator, *this, resultType, extendedArgs);
  return toValue(extendedResult, builder, loc);
}

mlir::Value
IntrinsicLibrary::invokeGenerator(SubroutineGenerator generator,
                                  llvm::ArrayRef<mlir::Value> args) {
  llvm::SmallVector<fir::ExtendedValue> extendedArgs;
  for (mlir::Value arg : args)
    extendedArgs.emplace_back(toExtendedValue(arg, builder, loc));
  std::invoke(generator, *this, extendedArgs);
  return {};
}

//===----------------------------------------------------------------------===//
// Intrinsic Procedure Mangling
//===----------------------------------------------------------------------===//

/// Helper to encode type into string for intrinsic procedure names.
/// Note: mlir has Type::dump(ostream) methods but it may add "!" that is not
/// suitable for function names.
static std::string typeToString(mlir::Type t) {
  if (auto refT{t.dyn_cast<fir::ReferenceType>()})
    return "ref_" + typeToString(refT.getEleTy());
  if (auto i{t.dyn_cast<mlir::IntegerType>()}) {
    return "i" + std::to_string(i.getWidth());
  }
  if (auto cplx{t.dyn_cast<fir::ComplexType>()}) {
    return "z" + std::to_string(cplx.getFKind());
  }
  if (auto real{t.dyn_cast<fir::RealType>()}) {
    return "r" + std::to_string(real.getFKind());
  }
  if (auto f{t.dyn_cast<mlir::FloatType>()}) {
    return "f" + std::to_string(f.getWidth());
  }
  if (auto logical{t.dyn_cast<fir::LogicalType>()}) {
    return "l" + std::to_string(logical.getFKind());
  }
  if (auto character{t.dyn_cast<fir::CharacterType>()}) {
    return "c" + std::to_string(character.getFKind());
  }
  if (auto boxCharacter{t.dyn_cast<fir::BoxCharType>()}) {
    return "bc" + std::to_string(boxCharacter.getEleTy().getFKind());
  }
  llvm_unreachable("no mangling for type");
}

/// Returns a name suitable to define mlir functions for Fortran intrinsic
/// Procedure. These names are guaranteed to not conflict with user defined
/// procedures. This is needed to implement Fortran generic intrinsics as
/// several mlir functions specialized for the argument types.
/// The result is guaranteed to be distinct for different mlir::FunctionType
/// arguments. The mangling pattern is:
///    fir.<generic name>.<result type>.<arg type>...
/// e.g ACOS(COMPLEX(4)) is mangled as fir.acos.z4.z4
static std::string mangleIntrinsicProcedure(llvm::StringRef intrinsic,
                                            mlir::FunctionType funTy) {
  std::string name = "fir.";
  name.append(intrinsic.str()).append(".");
  assert(funTy.getNumResults() == 1 && "only function mangling supported");
  name.append(typeToString(funTy.getResult(0)));
  unsigned e = funTy.getNumInputs();
  for (decltype(e) i = 0; i < e; ++i)
    name.append(".").append(typeToString(funTy.getInput(i)));
  return name;
}

template <typename GeneratorType>
mlir::func::FuncOp IntrinsicLibrary::getWrapper(GeneratorType generator,
                                                llvm::StringRef name,
                                                mlir::FunctionType funcType,
                                                bool loadRefArguments) {
  std::string wrapperName = mangleIntrinsicProcedure(name, funcType);
  mlir::func::FuncOp function = builder.getNamedFunction(wrapperName);
  if (!function) {
    // First time this wrapper is needed, build it.
    function = builder.createFunction(loc, wrapperName, funcType);
    function->setAttr("fir.intrinsic", builder.getUnitAttr());
    auto internalLinkage = mlir::LLVM::linkage::Linkage::Internal;
    auto linkage =
        mlir::LLVM::LinkageAttr::get(builder.getContext(), internalLinkage);
    function->setAttr("llvm.linkage", linkage);
    function.addEntryBlock();

    // Create local context to emit code into the newly created function
    // This new function is not linked to a source file location, only
    // its calls will be.
    auto localBuilder =
        std::make_unique<fir::FirOpBuilder>(function, builder.getKindMap());
    localBuilder->setInsertionPointToStart(&function.front());
    // Location of code inside wrapper of the wrapper is independent from
    // the location of the intrinsic call.
    mlir::Location localLoc = localBuilder->getUnknownLoc();
    llvm::SmallVector<mlir::Value> localArguments;
    for (mlir::BlockArgument bArg : function.front().getArguments()) {
      auto refType = bArg.getType().dyn_cast<fir::ReferenceType>();
      if (loadRefArguments && refType) {
        auto loaded = localBuilder->create<fir::LoadOp>(localLoc, bArg);
        localArguments.push_back(loaded);
      } else {
        localArguments.push_back(bArg);
      }
    }

    IntrinsicLibrary localLib{*localBuilder, localLoc};

    if constexpr (std::is_same_v<GeneratorType, SubroutineGenerator>) {
      localLib.invokeGenerator(generator, localArguments);
      localBuilder->create<mlir::func::ReturnOp>(localLoc);
    } else {
      assert(funcType.getNumResults() == 1 &&
             "expect one result for intrinsic function wrapper type");
      mlir::Type resultType = funcType.getResult(0);
      auto result =
          localLib.invokeGenerator(generator, resultType, localArguments);
      localBuilder->create<mlir::func::ReturnOp>(localLoc, result);
    }
  } else {
    // Wrapper was already built, ensure it has the sought type
    assert(function.getFunctionType() == funcType &&
           "conflict between intrinsic wrapper types");
  }
  return function;
}

/// Helpers to detect absent optional (not yet supported in outlining).
bool static hasAbsentOptional(llvm::ArrayRef<mlir::Value> args) {
  for (const mlir::Value &arg : args)
    if (!arg)
      return true;
  return false;
}
bool static hasAbsentOptional(llvm::ArrayRef<fir::ExtendedValue> args) {
  for (const fir::ExtendedValue &arg : args)
    if (!fir::getBase(arg))
      return true;
  return false;
}

template <typename GeneratorType>
mlir::Value
IntrinsicLibrary::outlineInWrapper(GeneratorType generator,
                                   llvm::StringRef name, mlir::Type resultType,
                                   llvm::ArrayRef<mlir::Value> args) {
  if (hasAbsentOptional(args)) {
    // TODO: absent optional in outlining is an issue: we cannot just ignore
    // them. Needs a better interface here. The issue is that we cannot easily
    // tell that a value is optional or not here if it is presents. And if it is
    // absent, we cannot tell what it type should be.
    TODO(loc, "cannot outline call to intrinsic " + llvm::Twine(name) +
                  " with absent optional argument");
  }

  mlir::FunctionType funcType = getFunctionType(resultType, args, builder);
  mlir::func::FuncOp wrapper = getWrapper(generator, name, funcType);
  return builder.create<fir::CallOp>(loc, wrapper, args).getResult(0);
}

template <typename GeneratorType>
fir::ExtendedValue IntrinsicLibrary::outlineInExtendedWrapper(
    GeneratorType generator, llvm::StringRef name,
    std::optional<mlir::Type> resultType,
    llvm::ArrayRef<fir::ExtendedValue> args) {
  if (hasAbsentOptional(args))
    TODO(loc, "cannot outline call to intrinsic " + llvm::Twine(name) +
                  " with absent optional argument");
  llvm::SmallVector<mlir::Value> mlirArgs;
  for (const auto &extendedVal : args)
    mlirArgs.emplace_back(toValue(extendedVal, builder, loc));
  mlir::FunctionType funcType = getFunctionType(resultType, mlirArgs, builder);
  mlir::func::FuncOp wrapper = getWrapper(generator, name, funcType);
  auto call = builder.create<fir::CallOp>(loc, wrapper, mlirArgs);
  if (resultType)
    return toExtendedValue(call.getResult(0), builder, loc);
  // Subroutine calls
  return mlir::Value{};
}

IntrinsicLibrary::RuntimeCallGenerator
IntrinsicLibrary::getRuntimeCallGenerator(llvm::StringRef name,
                                          mlir::FunctionType soughtFuncType) {
  mlir::FunctionType actualFuncType;
  const MathOperation *mathOp = nullptr;

  // Look for a dedicated math operation generator, which
  // normally produces a single MLIR operation implementing
  // the math operation.
  const MathOperation *bestNearMatch = nullptr;
  FunctionDistance bestMatchDistance;
  mathOp = searchMathOperation(builder, name, soughtFuncType, &bestNearMatch,
                               bestMatchDistance);
  if (!mathOp && bestNearMatch) {
    // Use the best near match, optionally issuing an error,
    // if types conversions cause precision loss.
    checkPrecisionLoss(name, soughtFuncType, bestMatchDistance, loc);
    mathOp = bestNearMatch;
  }

  if (!mathOp) {
    std::string nameAndType;
    llvm::raw_string_ostream sstream(nameAndType);
    sstream << name << "\nrequested type: " << soughtFuncType;
    crashOnMissingIntrinsic(loc, nameAndType);
  }

  actualFuncType = mathOp->typeGenerator(builder.getContext());

  assert(actualFuncType.getNumResults() == soughtFuncType.getNumResults() &&
         actualFuncType.getNumInputs() == soughtFuncType.getNumInputs() &&
         actualFuncType.getNumResults() == 1 && "Bad intrinsic match");

  return [actualFuncType, mathOp,
          soughtFuncType](fir::FirOpBuilder &builder, mlir::Location loc,
                          llvm::ArrayRef<mlir::Value> args) {
    llvm::SmallVector<mlir::Value> convertedArguments;
    for (auto [fst, snd] : llvm::zip(actualFuncType.getInputs(), args))
      convertedArguments.push_back(builder.createConvert(loc, fst, snd));
    mlir::Value result = mathOp->funcGenerator(
        builder, loc, mathOp->runtimeFunc, actualFuncType, convertedArguments);
    mlir::Type soughtType = soughtFuncType.getResult(0);
    return builder.createConvert(loc, soughtType, result);
  };
}

mlir::SymbolRefAttr IntrinsicLibrary::getUnrestrictedIntrinsicSymbolRefAttr(
    llvm::StringRef name, mlir::FunctionType signature) {
  // Unrestricted intrinsics signature follows implicit rules: argument
  // are passed by references. But the runtime versions expect values.
  // So instead of duplicating the runtime, just have the wrappers loading
  // this before calling the code generators.
  bool loadRefArguments = true;
  mlir::func::FuncOp funcOp;
  if (const IntrinsicHandler *handler = findIntrinsicHandler(name))
    funcOp = std::visit(
        [&](auto generator) {
          return getWrapper(generator, name, signature, loadRefArguments);
        },
        handler->generator);

  if (!funcOp) {
    llvm::SmallVector<mlir::Type> argTypes;
    for (mlir::Type type : signature.getInputs()) {
      if (auto refType = type.dyn_cast<fir::ReferenceType>())
        argTypes.push_back(refType.getEleTy());
      else
        argTypes.push_back(type);
    }
    mlir::FunctionType soughtFuncType =
        builder.getFunctionType(argTypes, signature.getResults());
    IntrinsicLibrary::RuntimeCallGenerator rtCallGenerator =
        getRuntimeCallGenerator(name, soughtFuncType);
    funcOp = getWrapper(rtCallGenerator, name, signature, loadRefArguments);
  }

  return mlir::SymbolRefAttr::get(funcOp);
}

fir::ExtendedValue
IntrinsicLibrary::readAndAddCleanUp(fir::MutableBoxValue resultMutableBox,
                                    mlir::Type resultType,
                                    llvm::StringRef intrinsicName) {
  fir::ExtendedValue res =
      fir::factory::genMutableBoxRead(builder, loc, resultMutableBox);
  return res.match(
      [&](const fir::ArrayBoxValue &box) -> fir::ExtendedValue {
        setResultMustBeFreed();
        return box;
      },
      [&](const fir::BoxValue &box) -> fir::ExtendedValue {
        setResultMustBeFreed();
        return box;
      },
      [&](const fir::CharArrayBoxValue &box) -> fir::ExtendedValue {
        setResultMustBeFreed();
        return box;
      },
      [&](const mlir::Value &tempAddr) -> fir::ExtendedValue {
        auto load = builder.create<fir::LoadOp>(loc, resultType, tempAddr);
        // Temp can be freed right away since it was loaded.
        builder.create<fir::FreeMemOp>(loc, tempAddr);
        return load;
      },
      [&](const fir::CharBoxValue &box) -> fir::ExtendedValue {
        setResultMustBeFreed();
        return box;
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(loc, "unexpected result for " + intrinsicName);
      });
}

//===----------------------------------------------------------------------===//
// Code generators for the intrinsic
//===----------------------------------------------------------------------===//

mlir::Value IntrinsicLibrary::genRuntimeCall(llvm::StringRef name,
                                             mlir::Type resultType,
                                             llvm::ArrayRef<mlir::Value> args) {
  mlir::FunctionType soughtFuncType =
      getFunctionType(resultType, args, builder);
  return getRuntimeCallGenerator(name, soughtFuncType)(builder, loc, args);
}

mlir::Value IntrinsicLibrary::genConversion(mlir::Type resultType,
                                            llvm::ArrayRef<mlir::Value> args) {
  // There can be an optional kind in second argument.
  assert(args.size() >= 1);
  return builder.convertWithSemantics(loc, resultType, args[0]);
}

// ABORT
void IntrinsicLibrary::genAbort(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 0);
  fir::runtime::genAbort(builder, loc);
}

// ABS
mlir::Value IntrinsicLibrary::genAbs(mlir::Type resultType,
                                     llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);
  mlir::Value arg = args[0];
  mlir::Type type = arg.getType();
  if (fir::isa_real(type) || fir::isa_complex(type)) {
    // Runtime call to fp abs. An alternative would be to use mlir
    // math::AbsFOp but it does not support all fir floating point types.
    return genRuntimeCall("abs", resultType, args);
  }
  if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
    // At the time of this implementation there is no abs op in mlir.
    // So, implement abs here without branching.
    mlir::Value shift =
        builder.createIntegerConstant(loc, intType, intType.getWidth() - 1);
    auto mask = builder.create<mlir::arith::ShRSIOp>(loc, arg, shift);
    auto xored = builder.create<mlir::arith::XOrIOp>(loc, arg, mask);
    return builder.create<mlir::arith::SubIOp>(loc, xored, mask);
  }
  llvm_unreachable("unexpected type in ABS argument");
}

// ADJUSTL & ADJUSTR
template <void (*CallRuntime)(fir::FirOpBuilder &, mlir::Location loc,
                              mlir::Value, mlir::Value)>
fir::ExtendedValue
IntrinsicLibrary::genAdjustRtCall(mlir::Type resultType,
                                  llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  mlir::Value string = builder.createBox(loc, args[0]);
  // Create a mutable fir.box to be passed to the runtime for the result.
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultType);
  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  // Call the runtime -- the runtime will allocate the result.
  CallRuntime(builder, loc, resultIrBox, string);
  // Read result from mutable fir.box and add it to the list of temps to be
  // finalized by the StatementContext.
  return readAndAddCleanUp(resultMutableBox, resultType, "ADJUSTL or ADJUSTR");
}

// AIMAG
mlir::Value IntrinsicLibrary::genAimag(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);
  return fir::factory::Complex{builder, loc}.extractComplexPart(
      args[0], /*isImagPart=*/true);
}

// AINT
mlir::Value IntrinsicLibrary::genAint(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() >= 1 && args.size() <= 2);
  // Skip optional kind argument to search the runtime; it is already reflected
  // in result type.
  return genRuntimeCall("aint", resultType, {args[0]});
}

// ALL
fir::ExtendedValue
IntrinsicLibrary::genAll(mlir::Type resultType,
                         llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 2);
  // Handle required mask argument
  mlir::Value mask = builder.createBox(loc, args[0]);

  fir::BoxValue maskArry = builder.createBox(loc, args[0]);
  int rank = maskArry.rank();
  assert(rank >= 1);

  // Handle optional dim argument
  bool absentDim = isStaticallyAbsent(args[1]);
  mlir::Value dim =
      absentDim ? builder.createIntegerConstant(loc, builder.getIndexType(), 1)
                : fir::getBase(args[1]);

  if (rank == 1 || absentDim)
    return builder.createConvert(loc, resultType,
                                 fir::runtime::genAll(builder, loc, mask, dim));

  // else use the result descriptor AllDim() intrinsic

  // Create mutable fir.box to be passed to the runtime for the result.

  mlir::Type resultArrayType = builder.getVarLenSeqTy(resultType, rank - 1);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genAllDescriptor(builder, loc, resultIrBox, mask, dim);
  return readAndAddCleanUp(resultMutableBox, resultType, "ALL");
}

// ALLOCATED
fir::ExtendedValue
IntrinsicLibrary::genAllocated(mlir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  return args[0].match(
      [&](const fir::MutableBoxValue &x) -> fir::ExtendedValue {
        return fir::factory::genIsAllocatedOrAssociatedTest(builder, loc, x);
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(loc,
                            "allocated arg not lowered to MutableBoxValue");
      });
}

// ANINT
mlir::Value IntrinsicLibrary::genAnint(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() >= 1 && args.size() <= 2);
  // Skip optional kind argument to search the runtime; it is already reflected
  // in result type.
  return genRuntimeCall("anint", resultType, {args[0]});
}

// ANY
fir::ExtendedValue
IntrinsicLibrary::genAny(mlir::Type resultType,
                         llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 2);
  // Handle required mask argument
  mlir::Value mask = builder.createBox(loc, args[0]);

  fir::BoxValue maskArry = builder.createBox(loc, args[0]);
  int rank = maskArry.rank();
  assert(rank >= 1);

  // Handle optional dim argument
  bool absentDim = isStaticallyAbsent(args[1]);
  mlir::Value dim =
      absentDim ? builder.createIntegerConstant(loc, builder.getIndexType(), 1)
                : fir::getBase(args[1]);

  if (rank == 1 || absentDim)
    return builder.createConvert(loc, resultType,
                                 fir::runtime::genAny(builder, loc, mask, dim));

  // else use the result descriptor AnyDim() intrinsic

  // Create mutable fir.box to be passed to the runtime for the result.

  mlir::Type resultArrayType = builder.getVarLenSeqTy(resultType, rank - 1);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genAnyDescriptor(builder, loc, resultIrBox, mask, dim);
  return readAndAddCleanUp(resultMutableBox, resultType, "ANY");
}

// ASSOCIATED
fir::ExtendedValue
IntrinsicLibrary::genAssociated(mlir::Type resultType,
                                llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  auto *pointer =
      args[0].match([&](const fir::MutableBoxValue &x) { return &x; },
                    [&](const auto &) -> const fir::MutableBoxValue * {
                      fir::emitFatalError(loc, "pointer not a MutableBoxValue");
                    });
  const fir::ExtendedValue &target = args[1];
  if (isStaticallyAbsent(target))
    return fir::factory::genIsAllocatedOrAssociatedTest(builder, loc, *pointer);

  mlir::Value targetBox;
  if (fir::valueHasFirAttribute(fir::getBase(target),
                                fir::getOptionalAttrName())) {
    // Subtle: contrary to other intrinsic optional arguments, disassociated
    // POINTER and unallocated ALLOCATABLE actual argument are not considered
    // absent here. This is because ASSOCIATED has special requirements for
    // TARGET actual arguments that are POINTERs. There is no precise
    // requirements for ALLOCATABLEs, but all existing Fortran compilers treat
    // them similarly to POINTERs. That is: unallocated TARGETs cause ASSOCIATED
    // to rerun false.  The runtime deals with the disassociated/unallocated
    // case. Simply ensures that TARGET that are OPTIONAL get conditionally
    // emboxed here to convey the optional aspect to the runtime.
    mlir::Type boxType = fir::BoxType::get(builder.getNoneType());
    auto isPresent = builder.create<fir::IsPresentOp>(loc, builder.getI1Type(),
                                                      fir::getBase(target));
    targetBox = builder
                    .genIfOp(loc, {boxType}, isPresent,
                             /*withElseRegion=*/true)
                    .genThen([&]() {
                      mlir::Value box = builder.createBox(loc, target);
                      mlir::Value cast =
                          builder.createConvert(loc, boxType, box);
                      builder.create<fir::ResultOp>(loc, cast);
                    })
                    .genElse([&]() {
                      mlir::Value absentBox =
                          builder.create<fir::AbsentOp>(loc, boxType);
                      builder.create<fir::ResultOp>(loc, absentBox);
                    })
                    .getResults()[0];
  } else {
    targetBox = builder.createBox(loc, target);
  }
  mlir::Value pointerBoxRef =
      fir::factory::getMutableIRBox(builder, loc, *pointer);
  auto pointerBox = builder.create<fir::LoadOp>(loc, pointerBoxRef);
  return fir::runtime::genAssociated(builder, loc, pointerBox, targetBox);
}

// BESSEL_JN
fir::ExtendedValue
IntrinsicLibrary::genBesselJn(mlir::Type resultType,
                              llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2 || args.size() == 3);

  mlir::Value x = fir::getBase(args.back());

  if (args.size() == 2) {
    mlir::Value n = fir::getBase(args[0]);

    return genRuntimeCall("bessel_jn", resultType, {n, x});
  } else {
    mlir::Value n1 = fir::getBase(args[0]);
    mlir::Value n2 = fir::getBase(args[1]);

    mlir::Type intTy = n1.getType();
    mlir::Type floatTy = x.getType();
    mlir::Value zero = builder.createRealZeroConstant(loc, floatTy);
    mlir::Value one = builder.createIntegerConstant(loc, intTy, 1);

    mlir::Type resultArrayType = builder.getVarLenSeqTy(resultType, 1);
    fir::MutableBoxValue resultMutableBox =
        fir::factory::createTempMutableBox(builder, loc, resultArrayType);
    mlir::Value resultBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

    mlir::Value cmpXEq0 = builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::UEQ, x, zero);
    mlir::Value cmpN1LtN2 = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, n1, n2);
    mlir::Value cmpN1EqN2 = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, n1, n2);

    auto genXEq0 = [&]() {
      fir::runtime::genBesselJnX0(builder, loc, floatTy, resultBox, n1, n2);
    };

    auto genN1LtN2 = [&]() {
      // The runtime generates the values in the range using a backward
      // recursion from n2 to n1. (see https://dlmf.nist.gov/10.74.iv and
      // https://dlmf.nist.gov/10.6.E1). When n1 < n2, this requires
      // the values of BESSEL_JN(n2) and BESSEL_JN(n2 - 1) since they
      // are the anchors of the recursion.
      mlir::Value n2_1 = builder.create<mlir::arith::SubIOp>(loc, n2, one);
      mlir::Value bn2 = genRuntimeCall("bessel_jn", resultType, {n2, x});
      mlir::Value bn2_1 = genRuntimeCall("bessel_jn", resultType, {n2_1, x});
      fir::runtime::genBesselJn(builder, loc, resultBox, n1, n2, x, bn2, bn2_1);
    };

    auto genN1EqN2 = [&]() {
      // When n1 == n2, only BESSEL_JN(n2) is needed.
      mlir::Value bn2 = genRuntimeCall("bessel_jn", resultType, {n2, x});
      fir::runtime::genBesselJn(builder, loc, resultBox, n1, n2, x, bn2, zero);
    };

    auto genN1GtN2 = [&]() {
      // The standard requires n1 <= n2. However, we still need to allocate
      // a zero-length array and return it when n1 > n2, so we do need to call
      // the runtime function.
      fir::runtime::genBesselJn(builder, loc, resultBox, n1, n2, x, zero, zero);
    };

    auto genN1GeN2 = [&] {
      builder.genIfThenElse(loc, cmpN1EqN2)
          .genThen(genN1EqN2)
          .genElse(genN1GtN2)
          .end();
    };

    auto genXNeq0 = [&]() {
      builder.genIfThenElse(loc, cmpN1LtN2)
          .genThen(genN1LtN2)
          .genElse(genN1GeN2)
          .end();
    };

    builder.genIfThenElse(loc, cmpXEq0)
        .genThen(genXEq0)
        .genElse(genXNeq0)
        .end();
    return readAndAddCleanUp(resultMutableBox, resultType, "BESSEL_JN");
  }
}

// BESSEL_YN
fir::ExtendedValue
IntrinsicLibrary::genBesselYn(mlir::Type resultType,
                              llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2 || args.size() == 3);

  mlir::Value x = fir::getBase(args.back());

  if (args.size() == 2) {
    mlir::Value n = fir::getBase(args[0]);

    return genRuntimeCall("bessel_yn", resultType, {n, x});
  } else {
    mlir::Value n1 = fir::getBase(args[0]);
    mlir::Value n2 = fir::getBase(args[1]);

    mlir::Type floatTy = x.getType();
    mlir::Type intTy = n1.getType();
    mlir::Value zero = builder.createRealZeroConstant(loc, floatTy);
    mlir::Value one = builder.createIntegerConstant(loc, intTy, 1);

    mlir::Type resultArrayType = builder.getVarLenSeqTy(resultType, 1);
    fir::MutableBoxValue resultMutableBox =
        fir::factory::createTempMutableBox(builder, loc, resultArrayType);
    mlir::Value resultBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

    mlir::Value cmpXEq0 = builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::UEQ, x, zero);
    mlir::Value cmpN1LtN2 = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, n1, n2);
    mlir::Value cmpN1EqN2 = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, n1, n2);

    auto genXEq0 = [&]() {
      fir::runtime::genBesselYnX0(builder, loc, floatTy, resultBox, n1, n2);
    };

    auto genN1LtN2 = [&]() {
      // The runtime generates the values in the range using a forward
      // recursion from n1 to n2. (see https://dlmf.nist.gov/10.74.iv and
      // https://dlmf.nist.gov/10.6.E1). When n1 < n2, this requires
      // the values of BESSEL_YN(n1) and BESSEL_YN(n1 + 1) since they
      // are the anchors of the recursion.
      mlir::Value n1_1 = builder.create<mlir::arith::AddIOp>(loc, n1, one);
      mlir::Value bn1 = genRuntimeCall("bessel_yn", resultType, {n1, x});
      mlir::Value bn1_1 = genRuntimeCall("bessel_yn", resultType, {n1_1, x});
      fir::runtime::genBesselYn(builder, loc, resultBox, n1, n2, x, bn1, bn1_1);
    };

    auto genN1EqN2 = [&]() {
      // When n1 == n2, only BESSEL_YN(n1) is needed.
      mlir::Value bn1 = genRuntimeCall("bessel_yn", resultType, {n1, x});
      fir::runtime::genBesselYn(builder, loc, resultBox, n1, n2, x, bn1, zero);
    };

    auto genN1GtN2 = [&]() {
      // The standard requires n1 <= n2. However, we still need to allocate
      // a zero-length array and return it when n1 > n2, so we do need to call
      // the runtime function.
      fir::runtime::genBesselYn(builder, loc, resultBox, n1, n2, x, zero, zero);
    };

    auto genN1GeN2 = [&] {
      builder.genIfThenElse(loc, cmpN1EqN2)
          .genThen(genN1EqN2)
          .genElse(genN1GtN2)
          .end();
    };

    auto genXNeq0 = [&]() {
      builder.genIfThenElse(loc, cmpN1LtN2)
          .genThen(genN1LtN2)
          .genElse(genN1GeN2)
          .end();
    };

    builder.genIfThenElse(loc, cmpXEq0)
        .genThen(genXEq0)
        .genElse(genXNeq0)
        .end();
    return readAndAddCleanUp(resultMutableBox, resultType, "BESSEL_YN");
  }
}

// BGE, BGT, BLE, BLT
template <mlir::arith::CmpIPredicate pred>
mlir::Value
IntrinsicLibrary::genBitwiseCompare(mlir::Type resultType,
                                    llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);

  mlir::Value arg0 = args[0];
  mlir::Value arg1 = args[1];
  mlir::Type arg0Ty = arg0.getType();
  mlir::Type arg1Ty = arg1.getType();
  unsigned bits0 = arg0Ty.getIntOrFloatBitWidth();
  unsigned bits1 = arg1Ty.getIntOrFloatBitWidth();

  // Arguments do not have to be of the same integer type. However, if neither
  // of the arguments is a BOZ literal, then the shorter of the two needs
  // to be converted to the longer by zero-extending (not sign-extending)
  // to the left [Fortran 2008, 13.3.2].
  //
  // In the case of BOZ literals, the standard describes zero-extension or
  // truncation depending on the kind of the result [Fortran 2008, 13.3.3].
  // However, that seems to be relevant for the case where the type of the
  // result must match the type of the BOZ literal. That is not the case for
  // these intrinsics, so, again, zero-extend to the larger type.
  //
  if (bits0 > bits1)
    arg1 = builder.create<mlir::arith::ExtUIOp>(loc, arg0Ty, arg1);
  else if (bits0 < bits1)
    arg0 = builder.create<mlir::arith::ExtUIOp>(loc, arg1Ty, arg0);

  return builder.create<mlir::arith::CmpIOp>(loc, pred, arg0, arg1);
}

// BTEST
mlir::Value IntrinsicLibrary::genBtest(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  // A conformant BTEST(I,POS) call satisfies:
  //     POS >= 0
  //     POS < BIT_SIZE(I)
  // Return:  (I >> POS) & 1
  assert(args.size() == 2);
  mlir::Type argType = args[0].getType();
  mlir::Value pos = builder.createConvert(loc, argType, args[1]);
  auto shift = builder.create<mlir::arith::ShRUIOp>(loc, args[0], pos);
  mlir::Value one = builder.createIntegerConstant(loc, argType, 1);
  auto res = builder.create<mlir::arith::AndIOp>(loc, shift, one);
  return builder.createConvert(loc, resultType, res);
}

static mlir::Value getAddrFromBox(fir::FirOpBuilder &builder,
                                  mlir::Location loc, fir::ExtendedValue arg,
                                  bool isFunc) {
  mlir::Value argValue = fir::getBase(arg);
  mlir::Value addr{nullptr};
  if (isFunc) {
    auto funcTy = argValue.getType().cast<fir::BoxProcType>().getEleTy();
    addr = builder.create<fir::BoxAddrOp>(loc, funcTy, argValue);
  } else {
    const auto *box = arg.getBoxOf<fir::BoxValue>();
    addr = builder.create<fir::BoxAddrOp>(loc, box->getMemTy(),
                                          fir::getBase(*box));
  }
  return addr;
}

static fir::ExtendedValue
genCLocOrCFunLoc(fir::FirOpBuilder &builder, mlir::Location loc,
                 mlir::Type resultType, llvm::ArrayRef<fir::ExtendedValue> args,
                 bool isFunc = false) {
  assert(args.size() == 1);
  mlir::Value res = builder.create<fir::AllocaOp>(loc, resultType);
  mlir::Value resAddr =
      fir::factory::genCPtrOrCFunptrAddr(builder, loc, res, resultType);
  assert(fir::isa_box_type(fir::getBase(args[0]).getType()) &&
         "argument must have been lowered to box type");
  mlir::Value argAddr = getAddrFromBox(builder, loc, args[0], isFunc);
  mlir::Value argAddrVal = builder.createConvert(
      loc, fir::unwrapRefType(resAddr.getType()), argAddr);
  builder.create<fir::StoreOp>(loc, argAddrVal, resAddr);
  return res;
}

/// C_ASSOCIATED
static fir::ExtendedValue
genCAssociated(fir::FirOpBuilder &builder, mlir::Location loc,
               mlir::Type resultType, llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  mlir::Value cPtr1 = fir::getBase(args[0]);
  mlir::Value cPtrVal1 =
      fir::factory::genCPtrOrCFunptrValue(builder, loc, cPtr1);
  mlir::Value zero = builder.createIntegerConstant(loc, cPtrVal1.getType(), 0);
  mlir::Value res = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, cPtrVal1, zero);

  if (isStaticallyPresent(args[1])) {
    mlir::Type i1Ty = builder.getI1Type();
    mlir::Value cPtr2 = fir::getBase(args[1]);
    mlir::Value isDynamicallyAbsent = builder.genIsNullAddr(loc, cPtr2);
    res =
        builder
            .genIfOp(loc, {i1Ty}, isDynamicallyAbsent, /*withElseRegion=*/true)
            .genThen([&]() { builder.create<fir::ResultOp>(loc, res); })
            .genElse([&]() {
              mlir::Value cPtrVal2 =
                  fir::factory::genCPtrOrCFunptrValue(builder, loc, cPtr2);
              mlir::Value cmpVal = builder.create<mlir::arith::CmpIOp>(
                  loc, mlir::arith::CmpIPredicate::eq, cPtrVal1, cPtrVal2);
              mlir::Value newRes =
                  builder.create<mlir::arith::AndIOp>(loc, res, cmpVal);
              builder.create<fir::ResultOp>(loc, newRes);
            })
            .getResults()[0];
  }
  return builder.createConvert(loc, resultType, res);
}

/// C_ASSOCIATED (C_FUNPTR [, C_FUNPTR])
fir::ExtendedValue IntrinsicLibrary::genCAssociatedCFunPtr(
    mlir::Type resultType, llvm::ArrayRef<fir::ExtendedValue> args) {
  return genCAssociated(builder, loc, resultType, args);
}

/// C_ASSOCIATED (C_PTR [, C_PTR])
fir::ExtendedValue
IntrinsicLibrary::genCAssociatedCPtr(mlir::Type resultType,
                                     llvm::ArrayRef<fir::ExtendedValue> args) {
  return genCAssociated(builder, loc, resultType, args);
}

// C_F_POINTER
void IntrinsicLibrary::genCFPointer(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  // Handle CPTR argument
  // Get the value of the C address or the result of a reference to C_LOC.
  mlir::Value cPtr = fir::getBase(args[0]);
  mlir::Value cPtrAddrVal =
      fir::factory::genCPtrOrCFunptrValue(builder, loc, cPtr);

  // Handle FPTR argument
  const auto *fPtr = args[1].getBoxOf<fir::MutableBoxValue>();
  assert(fPtr && "FPTR must be a pointer");

  auto getCPtrExtVal = [&](fir::MutableBoxValue box) -> fir::ExtendedValue {
    mlir::Value addr =
        builder.createConvert(loc, fPtr->getMemTy(), cPtrAddrVal);
    mlir::SmallVector<mlir::Value> extents;
    if (box.hasRank()) {
      assert(isStaticallyPresent(args[2]) &&
             "FPTR argument must be an array if SHAPE argument exists");
      mlir::Value shape = fir::getBase(args[2]);
      int arrayRank = box.rank();
      mlir::Type shapeElementType =
          fir::unwrapSequenceType(fir::unwrapPassByRefType(shape.getType()));
      mlir::Type idxType = builder.getIndexType();
      for (int i = 0; i < arrayRank; ++i) {
        mlir::Value index = builder.createIntegerConstant(loc, idxType, i);
        mlir::Value var = builder.create<fir::CoordinateOp>(
            loc, builder.getRefType(shapeElementType), shape, index);
        mlir::Value load = builder.create<fir::LoadOp>(loc, var);
        extents.push_back(builder.createConvert(loc, idxType, load));
      }
    }
    if (box.isCharacter()) {
      mlir::Value len = box.nonDeferredLenParams()[0];
      if (box.hasRank())
        return fir::CharArrayBoxValue{addr, len, extents};
      return fir::CharBoxValue{addr, len};
    }
    if (box.isDerivedWithLenParameters())
      TODO(loc, "get length parameters of derived type");
    if (box.hasRank())
      return fir::ArrayBoxValue{addr, extents};
    return addr;
  };

  fir::factory::associateMutableBox(builder, loc, *fPtr, getCPtrExtVal(*fPtr),
                                    /*lbounds=*/mlir::ValueRange{});
}

// C_FUNLOC
fir::ExtendedValue
IntrinsicLibrary::genCFunLoc(mlir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  return genCLocOrCFunLoc(builder, loc, resultType, args, /*isFunc=*/true);
}

// C_LOC
fir::ExtendedValue
IntrinsicLibrary::genCLoc(mlir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  return genCLocOrCFunLoc(builder, loc, resultType, args);
}

// CEILING
mlir::Value IntrinsicLibrary::genCeiling(mlir::Type resultType,
                                         llvm::ArrayRef<mlir::Value> args) {
  // Optional KIND argument.
  assert(args.size() >= 1);
  mlir::Value arg = args[0];
  // Use ceil that is not an actual Fortran intrinsic but that is
  // an llvm intrinsic that does the same, but return a floating
  // point.
  mlir::Value ceil = genRuntimeCall("ceil", arg.getType(), {arg});
  return builder.createConvert(loc, resultType, ceil);
}

// CHAR
fir::ExtendedValue
IntrinsicLibrary::genChar(mlir::Type type,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  // Optional KIND argument.
  assert(args.size() >= 1);
  const mlir::Value *arg = args[0].getUnboxed();
  // expect argument to be a scalar integer
  if (!arg)
    mlir::emitError(loc, "CHAR intrinsic argument not unboxed");
  fir::factory::CharacterExprHelper helper{builder, loc};
  fir::CharacterType::KindTy kind = helper.getCharacterType(type).getFKind();
  mlir::Value cast = helper.createSingletonFromCode(*arg, kind);
  mlir::Value len =
      builder.createIntegerConstant(loc, builder.getCharacterLengthType(), 1);
  return fir::CharBoxValue{cast, len};
}

// CMPLX
mlir::Value IntrinsicLibrary::genCmplx(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() >= 1);
  fir::factory::Complex complexHelper(builder, loc);
  mlir::Type partType = complexHelper.getComplexPartType(resultType);
  mlir::Value real = builder.createConvert(loc, partType, args[0]);
  mlir::Value imag = isStaticallyAbsent(args, 1)
                         ? builder.createRealZeroConstant(loc, partType)
                         : builder.createConvert(loc, partType, args[1]);
  return fir::factory::Complex{builder, loc}.createComplex(resultType, real,
                                                           imag);
}

// COMMAND_ARGUMENT_COUNT
fir::ExtendedValue IntrinsicLibrary::genCommandArgumentCount(
    mlir::Type resultType, llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 0);
  assert(resultType == builder.getDefaultIntegerType() &&
         "result type is not default integer kind type");
  return builder.createConvert(
      loc, resultType, fir::runtime::genCommandArgumentCount(builder, loc));
  ;
}

// CONJG
mlir::Value IntrinsicLibrary::genConjg(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);
  if (resultType != args[0].getType())
    llvm_unreachable("argument type mismatch");

  mlir::Value cplx = args[0];
  auto imag = fir::factory::Complex{builder, loc}.extractComplexPart(
      cplx, /*isImagPart=*/true);
  auto negImag = builder.create<mlir::arith::NegFOp>(loc, imag);
  return fir::factory::Complex{builder, loc}.insertComplexPart(
      cplx, negImag, /*isImagPart=*/true);
}

// COUNT
fir::ExtendedValue
IntrinsicLibrary::genCount(mlir::Type resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);

  // Handle mask argument
  fir::BoxValue mask = builder.createBox(loc, args[0]);
  unsigned maskRank = mask.rank();

  assert(maskRank > 0);

  // Handle optional dim argument
  bool absentDim = isStaticallyAbsent(args[1]);
  mlir::Value dim =
      absentDim ? builder.createIntegerConstant(loc, builder.getIndexType(), 0)
                : fir::getBase(args[1]);

  if (absentDim || maskRank == 1) {
    // Result is scalar if no dim argument or mask is rank 1.
    // So, call specialized Count runtime routine.
    return builder.createConvert(
        loc, resultType,
        fir::runtime::genCount(builder, loc, fir::getBase(mask), dim));
  }

  // Call general CountDim runtime routine.

  // Handle optional kind argument
  bool absentKind = isStaticallyAbsent(args[2]);
  mlir::Value kind = absentKind ? builder.createIntegerConstant(
                                      loc, builder.getIndexType(),
                                      builder.getKindMap().defaultIntegerKind())
                                : fir::getBase(args[2]);

  // Create mutable fir.box to be passed to the runtime for the result.
  mlir::Type type = builder.getVarLenSeqTy(resultType, maskRank - 1);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, type);

  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genCountDim(builder, loc, resultIrBox, fir::getBase(mask), dim,
                            kind);
  // Handle cleanup of allocatable result descriptor and return
  return readAndAddCleanUp(resultMutableBox, resultType, "COUNT");
}

// CPU_TIME
void IntrinsicLibrary::genCpuTime(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  const mlir::Value *arg = args[0].getUnboxed();
  assert(arg && "nonscalar cpu_time argument");
  mlir::Value res1 = fir::runtime::genCpuTime(builder, loc);
  mlir::Value res2 =
      builder.createConvert(loc, fir::dyn_cast_ptrEleTy(arg->getType()), res1);
  builder.create<fir::StoreOp>(loc, res2, *arg);
}

// CSHIFT
fir::ExtendedValue
IntrinsicLibrary::genCshift(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);

  // Handle required ARRAY argument
  fir::BoxValue arrayBox = builder.createBox(loc, args[0]);
  mlir::Value array = fir::getBase(arrayBox);
  unsigned arrayRank = arrayBox.rank();

  // Create mutable fir.box to be passed to the runtime for the result.
  mlir::Type resultArrayType = builder.getVarLenSeqTy(resultType, arrayRank);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  if (arrayRank == 1) {
    // Vector case
    // Handle required SHIFT argument as a scalar
    const mlir::Value *shiftAddr = args[1].getUnboxed();
    assert(shiftAddr && "nonscalar CSHIFT argument");
    auto shift = builder.create<fir::LoadOp>(loc, *shiftAddr);

    fir::runtime::genCshiftVector(builder, loc, resultIrBox, array, shift);
  } else {
    // Non-vector case
    // Handle required SHIFT argument as an array
    mlir::Value shift = builder.createBox(loc, args[1]);

    // Handle optional DIM argument
    mlir::Value dim =
        isStaticallyAbsent(args[2])
            ? builder.createIntegerConstant(loc, builder.getIndexType(), 1)
            : fir::getBase(args[2]);
    fir::runtime::genCshift(builder, loc, resultIrBox, array, shift, dim);
  }
  return readAndAddCleanUp(resultMutableBox, resultType, "CSHIFT");
}

// DATE_AND_TIME
void IntrinsicLibrary::genDateAndTime(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4 && "date_and_time has 4 args");
  llvm::SmallVector<std::optional<fir::CharBoxValue>> charArgs(3);
  for (unsigned i = 0; i < 3; ++i)
    if (const fir::CharBoxValue *charBox = args[i].getCharBox())
      charArgs[i] = *charBox;

  mlir::Value values = fir::getBase(args[3]);
  if (!values)
    values = builder.create<fir::AbsentOp>(
        loc, fir::BoxType::get(builder.getNoneType()));

  fir::runtime::genDateAndTime(builder, loc, charArgs[0], charArgs[1],
                               charArgs[2], values);
}

// DIM
mlir::Value IntrinsicLibrary::genDim(mlir::Type resultType,
                                     llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  if (resultType.isa<mlir::IntegerType>()) {
    mlir::Value zero = builder.createIntegerConstant(loc, resultType, 0);
    auto diff = builder.create<mlir::arith::SubIOp>(loc, args[0], args[1]);
    auto cmp = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::sgt, diff, zero);
    return builder.create<mlir::arith::SelectOp>(loc, cmp, diff, zero);
  }
  assert(fir::isa_real(resultType) && "Only expects real and integer in DIM");
  mlir::Value zero = builder.createRealZeroConstant(loc, resultType);
  auto diff = builder.create<mlir::arith::SubFOp>(loc, args[0], args[1]);
  auto cmp = builder.create<mlir::arith::CmpFOp>(
      loc, mlir::arith::CmpFPredicate::OGT, diff, zero);
  return builder.create<mlir::arith::SelectOp>(loc, cmp, diff, zero);
}

// DOT_PRODUCT
fir::ExtendedValue
IntrinsicLibrary::genDotProduct(mlir::Type resultType,
                                llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);

  // Handle required vector arguments
  mlir::Value vectorA = fir::getBase(args[0]);
  mlir::Value vectorB = fir::getBase(args[1]);
  // Result type is used for picking appropriate runtime function.
  mlir::Type eleTy = resultType;

  if (fir::isa_complex(eleTy)) {
    mlir::Value result = builder.createTemporary(loc, eleTy);
    fir::runtime::genDotProduct(builder, loc, vectorA, vectorB, result);
    return builder.create<fir::LoadOp>(loc, result);
  }

  // This operation is only used to pass the result type
  // information to the DotProduct generator.
  auto resultBox = builder.create<fir::AbsentOp>(loc, fir::BoxType::get(eleTy));
  return fir::runtime::genDotProduct(builder, loc, vectorA, vectorB, resultBox);
}

// DPROD
mlir::Value IntrinsicLibrary::genDprod(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  assert(fir::isa_real(resultType) &&
         "Result must be double precision in DPROD");
  mlir::Value a = builder.createConvert(loc, resultType, args[0]);
  mlir::Value b = builder.createConvert(loc, resultType, args[1]);
  return builder.create<mlir::arith::MulFOp>(loc, a, b);
}

// DSHIFTL
mlir::Value IntrinsicLibrary::genDshiftl(mlir::Type resultType,
                                         llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 3);

  mlir::Value i = args[0];
  mlir::Value j = args[1];
  mlir::Value shift = builder.createConvert(loc, resultType, args[2]);
  mlir::Value bitSize = builder.createIntegerConstant(
      loc, resultType, resultType.getIntOrFloatBitWidth());

  // Per the standard, the value of DSHIFTL(I, J, SHIFT) is equal to
  // IOR (SHIFTL(I, SHIFT), SHIFTR(J, BIT_SIZE(J) - SHIFT))
  mlir::Value diff = builder.create<mlir::arith::SubIOp>(loc, bitSize, shift);

  mlir::Value lArgs[2]{i, shift};
  mlir::Value lft = genShift<mlir::arith::ShLIOp>(resultType, lArgs);

  mlir::Value rArgs[2]{j, diff};
  mlir::Value rgt = genShift<mlir::arith::ShRUIOp>(resultType, rArgs);

  return builder.create<mlir::arith::OrIOp>(loc, lft, rgt);
}

// DSHIFTR
mlir::Value IntrinsicLibrary::genDshiftr(mlir::Type resultType,
                                         llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 3);

  mlir::Value i = args[0];
  mlir::Value j = args[1];
  mlir::Value shift = builder.createConvert(loc, resultType, args[2]);
  mlir::Value bitSize = builder.createIntegerConstant(
      loc, resultType, resultType.getIntOrFloatBitWidth());

  // Per the standard, the value of DSHIFTR(I, J, SHIFT) is equal to
  // IOR (SHIFTL(I, BIT_SIZE(I) - SHIFT), SHIFTR(J, SHIFT))
  mlir::Value diff = builder.create<mlir::arith::SubIOp>(loc, bitSize, shift);

  mlir::Value lArgs[2]{i, diff};
  mlir::Value lft = genShift<mlir::arith::ShLIOp>(resultType, lArgs);

  mlir::Value rArgs[2]{j, shift};
  mlir::Value rgt = genShift<mlir::arith::ShRUIOp>(resultType, rArgs);

  return builder.create<mlir::arith::OrIOp>(loc, lft, rgt);
}

// EOSHIFT
fir::ExtendedValue
IntrinsicLibrary::genEoshift(mlir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);

  // Handle required ARRAY argument
  fir::BoxValue arrayBox = builder.createBox(loc, args[0]);
  mlir::Value array = fir::getBase(arrayBox);
  unsigned arrayRank = arrayBox.rank();

  // Create mutable fir.box to be passed to the runtime for the result.
  mlir::Type resultArrayType = builder.getVarLenSeqTy(resultType, arrayRank);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  // Handle optional BOUNDARY argument
  mlir::Value boundary =
      isStaticallyAbsent(args[2])
          ? builder.create<fir::AbsentOp>(
                loc, fir::BoxType::get(builder.getNoneType()))
          : builder.createBox(loc, args[2]);

  if (arrayRank == 1) {
    // Vector case
    // Handle required SHIFT argument as a scalar
    const mlir::Value *shiftAddr = args[1].getUnboxed();
    assert(shiftAddr && "nonscalar EOSHIFT SHIFT argument");
    auto shift = builder.create<fir::LoadOp>(loc, *shiftAddr);
    fir::runtime::genEoshiftVector(builder, loc, resultIrBox, array, shift,
                                   boundary);
  } else {
    // Non-vector case
    // Handle required SHIFT argument as an array
    mlir::Value shift = builder.createBox(loc, args[1]);

    // Handle optional DIM argument
    mlir::Value dim =
        isStaticallyAbsent(args[3])
            ? builder.createIntegerConstant(loc, builder.getIndexType(), 1)
            : fir::getBase(args[3]);
    fir::runtime::genEoshift(builder, loc, resultIrBox, array, shift, boundary,
                             dim);
  }
  return readAndAddCleanUp(resultMutableBox, resultType, "EOSHIFT");
}

// EXIT
void IntrinsicLibrary::genExit(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);

  mlir::Value status =
      isStaticallyAbsent(args[0])
          ? builder.createIntegerConstant(loc, builder.getDefaultIntegerType(),
                                          EXIT_SUCCESS)
          : fir::getBase(args[0]);

  assert(status.getType() == builder.getDefaultIntegerType() &&
         "STATUS parameter must be an INTEGER of default kind");

  fir::runtime::genExit(builder, loc, status);
}

// EXPONENT
mlir::Value IntrinsicLibrary::genExponent(mlir::Type resultType,
                                          llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genExponent(builder, loc, resultType,
                                fir::getBase(args[0])));
}

// EXTENDS_TYPE_OF
fir::ExtendedValue
IntrinsicLibrary::genExtendsTypeOf(mlir::Type resultType,
                                   llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genExtendsTypeOf(builder, loc, fir::getBase(args[0]),
                                     fir::getBase(args[1])));
}

// FINDLOC
fir::ExtendedValue
IntrinsicLibrary::genFindloc(mlir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 6);

  // Handle required array argument
  mlir::Value array = builder.createBox(loc, args[0]);
  unsigned rank = fir::BoxValue(array).rank();
  assert(rank >= 1);

  // Handle required value argument
  mlir::Value val = builder.createBox(loc, args[1]);

  // Check if dim argument is present
  bool absentDim = isStaticallyAbsent(args[2]);

  // Handle optional mask argument
  auto mask = isStaticallyAbsent(args[3])
                  ? builder.create<fir::AbsentOp>(
                        loc, fir::BoxType::get(builder.getI1Type()))
                  : builder.createBox(loc, args[3]);

  // Handle optional kind argument
  auto kind = isStaticallyAbsent(args[4])
                  ? builder.createIntegerConstant(
                        loc, builder.getIndexType(),
                        builder.getKindMap().defaultIntegerKind())
                  : fir::getBase(args[4]);

  // Handle optional back argument
  auto back = isStaticallyAbsent(args[5]) ? builder.createBool(loc, false)
                                          : fir::getBase(args[5]);

  if (!absentDim && rank == 1) {
    // If dim argument is present and the array is rank 1, then the result is
    // a scalar (since the the result is rank-1 or 0).
    // Therefore, we use a scalar result descriptor with FindlocDim().
    // Create mutable fir.box to be passed to the runtime for the result.
    fir::MutableBoxValue resultMutableBox =
        fir::factory::createTempMutableBox(builder, loc, resultType);
    mlir::Value resultIrBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
    mlir::Value dim = fir::getBase(args[2]);

    fir::runtime::genFindlocDim(builder, loc, resultIrBox, array, val, dim,
                                mask, kind, back);
    // Handle cleanup of allocatable result descriptor and return
    return readAndAddCleanUp(resultMutableBox, resultType, "FINDLOC");
  }

  // The result will be an array. Create mutable fir.box to be passed to the
  // runtime for the result.
  mlir::Type resultArrayType =
      builder.getVarLenSeqTy(resultType, absentDim ? 1 : rank - 1);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  if (absentDim) {
    fir::runtime::genFindloc(builder, loc, resultIrBox, array, val, mask, kind,
                             back);
  } else {
    mlir::Value dim = fir::getBase(args[2]);
    fir::runtime::genFindlocDim(builder, loc, resultIrBox, array, val, dim,
                                mask, kind, back);
  }
  return readAndAddCleanUp(resultMutableBox, resultType, "FINDLOC");
}

// FLOOR
mlir::Value IntrinsicLibrary::genFloor(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  // Optional KIND argument.
  assert(args.size() >= 1);
  mlir::Value arg = args[0];
  // Use LLVM floor that returns real.
  mlir::Value floor = genRuntimeCall("floor", arg.getType(), {arg});
  return builder.createConvert(loc, resultType, floor);
}

// FRACTION
mlir::Value IntrinsicLibrary::genFraction(mlir::Type resultType,
                                          llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genFraction(builder, loc, fir::getBase(args[0])));
}

// GET_COMMAND
void IntrinsicLibrary::genGetCommand(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);
  const fir::ExtendedValue &command = args[0];
  const fir::ExtendedValue &length = args[1];
  const fir::ExtendedValue &status = args[2];
  const fir::ExtendedValue &errmsg = args[3];

  // If none of the optional parameters are present, do nothing.
  if (!isStaticallyPresent(command) && !isStaticallyPresent(length) &&
      !isStaticallyPresent(status) && !isStaticallyPresent(errmsg))
    return;

  mlir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());
  mlir::Value commandBox =
      isStaticallyPresent(command)
          ? fir::getBase(command)
          : builder.create<fir::AbsentOp>(loc, boxNoneTy).getResult();
  mlir::Value lenBox =
      isStaticallyPresent(length)
          ? fir::getBase(length)
          : builder.create<fir::AbsentOp>(loc, boxNoneTy).getResult();
  mlir::Value errBox =
      isStaticallyPresent(errmsg)
          ? fir::getBase(errmsg)
          : builder.create<fir::AbsentOp>(loc, boxNoneTy).getResult();
  mlir::Value stat =
      fir::runtime::genGetCommand(builder, loc, commandBox, lenBox, errBox);
  if (isStaticallyPresent(status)) {
    mlir::Value statAddr = fir::getBase(status);
    mlir::Value statIsPresentAtRuntime =
        builder.genIsNotNullAddr(loc, statAddr);
    builder.genIfThen(loc, statIsPresentAtRuntime)
        .genThen([&]() { builder.createStoreWithConvert(loc, stat, statAddr); })
        .end();
  }
}

// GET_COMMAND_ARGUMENT
void IntrinsicLibrary::genGetCommandArgument(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 5);
  mlir::Value number = fir::getBase(args[0]);
  const fir::ExtendedValue &value = args[1];
  const fir::ExtendedValue &length = args[2];
  const fir::ExtendedValue &status = args[3];
  const fir::ExtendedValue &errmsg = args[4];

  if (!number)
    fir::emitFatalError(loc, "expected NUMBER parameter");

  // If none of the optional parameters are present, do nothing.
  if (!isStaticallyPresent(value) && !isStaticallyPresent(length) &&
      !isStaticallyPresent(status) && !isStaticallyPresent(errmsg))
    return;

  mlir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());
  mlir::Value valBox =
      isStaticallyPresent(value)
          ? fir::getBase(value)
          : builder.create<fir::AbsentOp>(loc, boxNoneTy).getResult();
  mlir::Value lenBox =
      isStaticallyPresent(length)
          ? fir::getBase(length)
          : builder.create<fir::AbsentOp>(loc, boxNoneTy).getResult();
  mlir::Value errBox =
      isStaticallyPresent(errmsg)
          ? fir::getBase(errmsg)
          : builder.create<fir::AbsentOp>(loc, boxNoneTy).getResult();
  mlir::Value stat = fir::runtime::genGetCommandArgument(
      builder, loc, number, valBox, lenBox, errBox);
  if (isStaticallyPresent(status)) {
    mlir::Value statAddr = fir::getBase(status);
    mlir::Value statIsPresentAtRuntime =
        builder.genIsNotNullAddr(loc, statAddr);
    builder.genIfThen(loc, statIsPresentAtRuntime)
        .genThen([&]() { builder.createStoreWithConvert(loc, stat, statAddr); })
        .end();
  }
}

// GET_ENVIRONMENT_VARIABLE
void IntrinsicLibrary::genGetEnvironmentVariable(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 6);
  mlir::Value name = fir::getBase(args[0]);
  const fir::ExtendedValue &value = args[1];
  const fir::ExtendedValue &length = args[2];
  const fir::ExtendedValue &status = args[3];
  const fir::ExtendedValue &trimName = args[4];
  const fir::ExtendedValue &errmsg = args[5];

  if (!name)
    fir::emitFatalError(loc, "expected NAME parameter");

  // If none of the optional parameters are present, do nothing.
  if (!isStaticallyPresent(value) && !isStaticallyPresent(length) &&
      !isStaticallyPresent(status) && !isStaticallyPresent(errmsg))
    return;

  // Handle optional TRIM_NAME argument
  mlir::Value trim;
  if (isStaticallyAbsent(trimName)) {
    trim = builder.createBool(loc, true);
  } else {
    mlir::Type i1Ty = builder.getI1Type();
    mlir::Value trimNameAddr = fir::getBase(trimName);
    mlir::Value trimNameIsPresentAtRuntime =
        builder.genIsNotNullAddr(loc, trimNameAddr);
    trim = builder
               .genIfOp(loc, {i1Ty}, trimNameIsPresentAtRuntime,
                        /*withElseRegion=*/true)
               .genThen([&]() {
                 auto trimLoad = builder.create<fir::LoadOp>(loc, trimNameAddr);
                 mlir::Value cast = builder.createConvert(loc, i1Ty, trimLoad);
                 builder.create<fir::ResultOp>(loc, cast);
               })
               .genElse([&]() {
                 mlir::Value trueVal = builder.createBool(loc, true);
                 builder.create<fir::ResultOp>(loc, trueVal);
               })
               .getResults()[0];
  }

  mlir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());
  mlir::Value valBox =
      isStaticallyPresent(value)
          ? fir::getBase(value)
          : builder.create<fir::AbsentOp>(loc, boxNoneTy).getResult();
  mlir::Value lenBox =
      isStaticallyPresent(length)
          ? fir::getBase(length)
          : builder.create<fir::AbsentOp>(loc, boxNoneTy).getResult();
  mlir::Value errBox =
      isStaticallyPresent(errmsg)
          ? fir::getBase(errmsg)
          : builder.create<fir::AbsentOp>(loc, boxNoneTy).getResult();
  mlir::Value stat = fir::runtime::genGetEnvVariable(builder, loc, name, valBox,
                                                     lenBox, trim, errBox);
  if (isStaticallyPresent(status)) {
    mlir::Value statAddr = fir::getBase(status);
    mlir::Value statIsPresentAtRuntime =
        builder.genIsNotNullAddr(loc, statAddr);
    builder.genIfThen(loc, statIsPresentAtRuntime)
        .genThen([&]() { builder.createStoreWithConvert(loc, stat, statAddr); })
        .end();
  }
}

/// Process calls to Maxval, Minval, Product, Sum intrinsic functions that
/// take a DIM argument.
template <typename FD>
static fir::MutableBoxValue
genFuncDim(FD funcDim, mlir::Type resultType, fir::FirOpBuilder &builder,
           mlir::Location loc, mlir::Value array, fir::ExtendedValue dimArg,
           mlir::Value mask, int rank) {

  // Create mutable fir.box to be passed to the runtime for the result.
  mlir::Type resultArrayType = builder.getVarLenSeqTy(resultType, rank - 1);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  mlir::Value dim =
      isStaticallyAbsent(dimArg)
          ? builder.createIntegerConstant(loc, builder.getIndexType(), 0)
          : fir::getBase(dimArg);
  funcDim(builder, loc, resultIrBox, array, dim, mask);

  return resultMutableBox;
}

/// Process calls to Product, Sum, IAll, IAny, IParity intrinsic functions
template <typename FN, typename FD>
fir::ExtendedValue
IntrinsicLibrary::genReduction(FN func, FD funcDim, llvm::StringRef errMsg,
                               mlir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 3);

  // Handle required array argument
  fir::BoxValue arryTmp = builder.createBox(loc, args[0]);
  mlir::Value array = fir::getBase(arryTmp);
  int rank = arryTmp.rank();
  assert(rank >= 1);

  // Handle optional mask argument
  auto mask = isStaticallyAbsent(args[2])
                  ? builder.create<fir::AbsentOp>(
                        loc, fir::BoxType::get(builder.getI1Type()))
                  : builder.createBox(loc, args[2]);

  bool absentDim = isStaticallyAbsent(args[1]);

  // We call the type specific versions because the result is scalar
  // in the case below.
  if (absentDim || rank == 1) {
    mlir::Type ty = array.getType();
    mlir::Type arrTy = fir::dyn_cast_ptrOrBoxEleTy(ty);
    auto eleTy = arrTy.cast<fir::SequenceType>().getEleTy();
    if (fir::isa_complex(eleTy)) {
      mlir::Value result = builder.createTemporary(loc, eleTy);
      func(builder, loc, array, mask, result);
      return builder.create<fir::LoadOp>(loc, result);
    }
    auto resultBox = builder.create<fir::AbsentOp>(
        loc, fir::BoxType::get(builder.getI1Type()));
    return func(builder, loc, array, mask, resultBox);
  }
  // Handle Product/Sum cases that have an array result.
  auto resultMutableBox =
      genFuncDim(funcDim, resultType, builder, loc, array, args[1], mask, rank);
  return readAndAddCleanUp(resultMutableBox, resultType, errMsg);
}

// IALL
fir::ExtendedValue
IntrinsicLibrary::genIall(mlir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  return genReduction(fir::runtime::genIAll, fir::runtime::genIAllDim, "IALL",
                      resultType, args);
}

// IAND
mlir::Value IntrinsicLibrary::genIand(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  auto arg0 = builder.createConvert(loc, resultType, args[0]);
  auto arg1 = builder.createConvert(loc, resultType, args[1]);
  return builder.create<mlir::arith::AndIOp>(loc, arg0, arg1);
}

// IANY
fir::ExtendedValue
IntrinsicLibrary::genIany(mlir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  return genReduction(fir::runtime::genIAny, fir::runtime::genIAnyDim, "IANY",
                      resultType, args);
}

// IBCLR
mlir::Value IntrinsicLibrary::genIbclr(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  // A conformant IBCLR(I,POS) call satisfies:
  //     POS >= 0
  //     POS < BIT_SIZE(I)
  // Return:  I & (!(1 << POS))
  assert(args.size() == 2);
  mlir::Value pos = builder.createConvert(loc, resultType, args[1]);
  mlir::Value one = builder.createIntegerConstant(loc, resultType, 1);
  mlir::Value ones = builder.createIntegerConstant(loc, resultType, -1);
  auto mask = builder.create<mlir::arith::ShLIOp>(loc, one, pos);
  auto res = builder.create<mlir::arith::XOrIOp>(loc, ones, mask);
  return builder.create<mlir::arith::AndIOp>(loc, args[0], res);
}

// IBITS
mlir::Value IntrinsicLibrary::genIbits(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  // A conformant IBITS(I,POS,LEN) call satisfies:
  //     POS >= 0
  //     LEN >= 0
  //     POS + LEN <= BIT_SIZE(I)
  // Return:  LEN == 0 ? 0 : (I >> POS) & (-1 >> (BIT_SIZE(I) - LEN))
  // For a conformant call, implementing (I >> POS) with a signed or an
  // unsigned shift produces the same result.  For a nonconformant call,
  // the two choices may produce different results.
  assert(args.size() == 3);
  mlir::Value pos = builder.createConvert(loc, resultType, args[1]);
  mlir::Value len = builder.createConvert(loc, resultType, args[2]);
  mlir::Value bitSize = builder.createIntegerConstant(
      loc, resultType, resultType.cast<mlir::IntegerType>().getWidth());
  auto shiftCount = builder.create<mlir::arith::SubIOp>(loc, bitSize, len);
  mlir::Value zero = builder.createIntegerConstant(loc, resultType, 0);
  mlir::Value ones = builder.createIntegerConstant(loc, resultType, -1);
  auto mask = builder.create<mlir::arith::ShRUIOp>(loc, ones, shiftCount);
  auto res1 = builder.create<mlir::arith::ShRSIOp>(loc, args[0], pos);
  auto res2 = builder.create<mlir::arith::AndIOp>(loc, res1, mask);
  auto lenIsZero = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, len, zero);
  return builder.create<mlir::arith::SelectOp>(loc, lenIsZero, zero, res2);
}

// IBSET
mlir::Value IntrinsicLibrary::genIbset(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  // A conformant IBSET(I,POS) call satisfies:
  //     POS >= 0
  //     POS < BIT_SIZE(I)
  // Return:  I | (1 << POS)
  assert(args.size() == 2);
  mlir::Value pos = builder.createConvert(loc, resultType, args[1]);
  mlir::Value one = builder.createIntegerConstant(loc, resultType, 1);
  auto mask = builder.create<mlir::arith::ShLIOp>(loc, one, pos);
  return builder.create<mlir::arith::OrIOp>(loc, args[0], mask);
}

// ICHAR
fir::ExtendedValue
IntrinsicLibrary::genIchar(mlir::Type resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  // There can be an optional kind in second argument.
  assert(args.size() == 2);
  const fir::CharBoxValue *charBox = args[0].getCharBox();
  if (!charBox)
    llvm::report_fatal_error("expected character scalar");

  fir::factory::CharacterExprHelper helper{builder, loc};
  mlir::Value buffer = charBox->getBuffer();
  mlir::Type bufferTy = buffer.getType();
  mlir::Value charVal;
  if (auto charTy = bufferTy.dyn_cast<fir::CharacterType>()) {
    assert(charTy.singleton());
    charVal = buffer;
  } else {
    // Character is in memory, cast to fir.ref<char> and load.
    mlir::Type ty = fir::dyn_cast_ptrEleTy(bufferTy);
    if (!ty)
      llvm::report_fatal_error("expected memory type");
    // The length of in the character type may be unknown. Casting
    // to a singleton ref is required before loading.
    fir::CharacterType eleType = helper.getCharacterType(ty);
    fir::CharacterType charType =
        fir::CharacterType::get(builder.getContext(), eleType.getFKind(), 1);
    mlir::Type toTy = builder.getRefType(charType);
    mlir::Value cast = builder.createConvert(loc, toTy, buffer);
    charVal = builder.create<fir::LoadOp>(loc, cast);
  }
  LLVM_DEBUG(llvm::dbgs() << "ichar(" << charVal << ")\n");
  auto code = helper.extractCodeFromSingleton(charVal);
  if (code.getType() == resultType)
    return code;
  return builder.create<mlir::arith::ExtUIOp>(loc, resultType, code);
}

// IEEE_CLASS_TYPE OPERATOR(==), OPERATOR(/=)
// IEEE_ROUND_TYPE OPERATOR(==), OPERATOR(/=)
template <mlir::arith::CmpIPredicate pred>
fir::ExtendedValue
IntrinsicLibrary::genIeeeTypeCompare(mlir::Type resultType,
                                     llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  mlir::Value arg0 = fir::getBase(args[0]);
  mlir::Value arg1 = fir::getBase(args[1]);
  auto recType =
      fir::unwrapPassByRefType(arg0.getType()).dyn_cast<fir::RecordType>();
  assert(recType.getTypeList().size() == 1 && "expected exactly one component");
  auto [fieldName, fieldType] = recType.getTypeList().front();
  mlir::Type fieldIndexType = fir::FieldType::get(recType.getContext());
  mlir::Value field = builder.create<fir::FieldIndexOp>(
      loc, fieldIndexType, fieldName, recType, fir::getTypeParams(arg0));
  mlir::Value left = builder.create<fir::LoadOp>(
      loc, fieldType,
      builder.create<fir::CoordinateOp>(loc, builder.getRefType(fieldType),
                                        arg0, field));
  mlir::Value right = builder.create<fir::LoadOp>(
      loc, fieldType,
      builder.create<fir::CoordinateOp>(loc, builder.getRefType(fieldType),
                                        arg1, field));
  return builder.create<mlir::arith::CmpIOp>(loc, pred, left, right);
}

// IEEE_IS_FINITE
mlir::Value
IntrinsicLibrary::genIeeeIsFinite(mlir::Type resultType,
                                  llvm::ArrayRef<mlir::Value> args) {
  // IEEE_IS_FINITE(X) is true iff exponent(X) is the max exponent of kind(X).
  assert(args.size() == 1);
  mlir::Value floatVal = fir::getBase(args[0]);
  mlir::FloatType floatType = floatVal.getType().dyn_cast<mlir::FloatType>();
  int floatBits = floatType.getWidth();
  mlir::Type intType = builder.getIntegerType(
      floatType.isa<mlir::Float80Type>() ? 128 : floatBits);
  mlir::Value intVal =
      builder.create<mlir::arith::BitcastOp>(loc, intType, floatVal);
  int significandBits;
  if (floatType.isa<mlir::Float32Type>())
    significandBits = 23;
  else if (floatType.isa<mlir::Float64Type>())
    significandBits = 52;
  else // problems elsewhere for other kinds
    TODO(loc, "intrinsic module procedure: ieee_is_finite");
  mlir::Value significand =
      builder.createIntegerConstant(loc, intType, significandBits);
  int exponentBits = floatBits - 1 - significandBits;
  mlir::Value maxExponent =
      builder.createIntegerConstant(loc, intType, (1 << exponentBits) - 1);
  mlir::Value exponent = genIbits(
      intType, {intVal, significand,
                builder.createIntegerConstant(loc, intType, exponentBits)});
  return builder.createConvert(
      loc, resultType,
      builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne,
                                          exponent, maxExponent));
}

// IEOR
mlir::Value IntrinsicLibrary::genIeor(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  return builder.create<mlir::arith::XOrIOp>(loc, args[0], args[1]);
}

// INDEX
fir::ExtendedValue
IntrinsicLibrary::genIndex(mlir::Type resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() >= 2 && args.size() <= 4);

  mlir::Value stringBase = fir::getBase(args[0]);
  fir::KindTy kind =
      fir::factory::CharacterExprHelper{builder, loc}.getCharacterKind(
          stringBase.getType());
  mlir::Value stringLen = fir::getLen(args[0]);
  mlir::Value substringBase = fir::getBase(args[1]);
  mlir::Value substringLen = fir::getLen(args[1]);
  mlir::Value back =
      isStaticallyAbsent(args, 2)
          ? builder.createIntegerConstant(loc, builder.getI1Type(), 0)
          : fir::getBase(args[2]);
  if (isStaticallyAbsent(args, 3))
    return builder.createConvert(
        loc, resultType,
        fir::runtime::genIndex(builder, loc, kind, stringBase, stringLen,
                               substringBase, substringLen, back));

  // Call the descriptor-based Index implementation
  mlir::Value string = builder.createBox(loc, args[0]);
  mlir::Value substring = builder.createBox(loc, args[1]);
  auto makeRefThenEmbox = [&](mlir::Value b) {
    fir::LogicalType logTy = fir::LogicalType::get(
        builder.getContext(), builder.getKindMap().defaultLogicalKind());
    mlir::Value temp = builder.createTemporary(loc, logTy);
    mlir::Value castb = builder.createConvert(loc, logTy, b);
    builder.create<fir::StoreOp>(loc, castb, temp);
    return builder.createBox(loc, temp);
  };
  mlir::Value backOpt = isStaticallyAbsent(args, 2)
                            ? builder.create<fir::AbsentOp>(
                                  loc, fir::BoxType::get(builder.getI1Type()))
                            : makeRefThenEmbox(fir::getBase(args[2]));
  mlir::Value kindVal = isStaticallyAbsent(args, 3)
                            ? builder.createIntegerConstant(
                                  loc, builder.getIndexType(),
                                  builder.getKindMap().defaultIntegerKind())
                            : fir::getBase(args[3]);
  // Create mutable fir.box to be passed to the runtime for the result.
  fir::MutableBoxValue mutBox =
      fir::factory::createTempMutableBox(builder, loc, resultType);
  mlir::Value resBox = fir::factory::getMutableIRBox(builder, loc, mutBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genIndexDescriptor(builder, loc, resBox, string, substring,
                                   backOpt, kindVal);
  // Read back the result from the mutable box.
  return readAndAddCleanUp(mutBox, resultType, "INDEX");
}

// IOR
mlir::Value IntrinsicLibrary::genIor(mlir::Type resultType,
                                     llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  return builder.create<mlir::arith::OrIOp>(loc, args[0], args[1]);
}

// IPARITY
fir::ExtendedValue
IntrinsicLibrary::genIparity(mlir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  return genReduction(fir::runtime::genIParity, fir::runtime::genIParityDim,
                      "IPARITY", resultType, args);
}

// IS_CONTIGUOUS
fir::ExtendedValue
IntrinsicLibrary::genIsContiguous(mlir::Type resultType,
                                  llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  if (const auto *boxValue = args[0].getBoxOf<fir::BoxValue>())
    if (boxValue->hasAssumedRank())
      TODO(loc, "intrinsic: is_contiguous with assumed rank argument");

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genIsContiguous(builder, loc, fir::getBase(args[0])));
}

// ISHFT
mlir::Value IntrinsicLibrary::genIshft(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  // A conformant ISHFT(I,SHIFT) call satisfies:
  //     abs(SHIFT) <= BIT_SIZE(I)
  // Return:  abs(SHIFT) >= BIT_SIZE(I)
  //              ? 0
  //              : SHIFT < 0
  //                    ? I >> abs(SHIFT)
  //                    : I << abs(SHIFT)
  assert(args.size() == 2);
  mlir::Value bitSize = builder.createIntegerConstant(
      loc, resultType, resultType.cast<mlir::IntegerType>().getWidth());
  mlir::Value zero = builder.createIntegerConstant(loc, resultType, 0);
  mlir::Value shift = builder.createConvert(loc, resultType, args[1]);
  mlir::Value absShift = genAbs(resultType, {shift});
  auto left = builder.create<mlir::arith::ShLIOp>(loc, args[0], absShift);
  auto right = builder.create<mlir::arith::ShRUIOp>(loc, args[0], absShift);
  auto shiftIsLarge = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::sge, absShift, bitSize);
  auto shiftIsNegative = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, shift, zero);
  auto sel =
      builder.create<mlir::arith::SelectOp>(loc, shiftIsNegative, right, left);
  return builder.create<mlir::arith::SelectOp>(loc, shiftIsLarge, zero, sel);
}

// ISHFTC
mlir::Value IntrinsicLibrary::genIshftc(mlir::Type resultType,
                                        llvm::ArrayRef<mlir::Value> args) {
  // A conformant ISHFTC(I,SHIFT,SIZE) call satisfies:
  //     SIZE > 0
  //     SIZE <= BIT_SIZE(I)
  //     abs(SHIFT) <= SIZE
  // if SHIFT > 0
  //     leftSize = abs(SHIFT)
  //     rightSize = SIZE - abs(SHIFT)
  // else [if SHIFT < 0]
  //     leftSize = SIZE - abs(SHIFT)
  //     rightSize = abs(SHIFT)
  // unchanged = SIZE == BIT_SIZE(I) ? 0 : (I >> SIZE) << SIZE
  // leftMaskShift = BIT_SIZE(I) - leftSize
  // rightMaskShift = BIT_SIZE(I) - rightSize
  // left = (I >> rightSize) & (-1 >> leftMaskShift)
  // right = (I & (-1 >> rightMaskShift)) << leftSize
  // Return:  SHIFT == 0 || SIZE == abs(SHIFT) ? I : (unchanged | left | right)
  assert(args.size() == 3);
  mlir::Value bitSize = builder.createIntegerConstant(
      loc, resultType, resultType.cast<mlir::IntegerType>().getWidth());
  mlir::Value I = args[0];
  mlir::Value shift = builder.createConvert(loc, resultType, args[1]);
  mlir::Value size =
      args[2] ? builder.createConvert(loc, resultType, args[2]) : bitSize;
  mlir::Value zero = builder.createIntegerConstant(loc, resultType, 0);
  mlir::Value ones = builder.createIntegerConstant(loc, resultType, -1);
  mlir::Value absShift = genAbs(resultType, {shift});
  auto elseSize = builder.create<mlir::arith::SubIOp>(loc, size, absShift);
  auto shiftIsZero = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, shift, zero);
  auto shiftEqualsSize = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, absShift, size);
  auto shiftIsNop =
      builder.create<mlir::arith::OrIOp>(loc, shiftIsZero, shiftEqualsSize);
  auto shiftIsPositive = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::sgt, shift, zero);
  auto leftSize = builder.create<mlir::arith::SelectOp>(loc, shiftIsPositive,
                                                        absShift, elseSize);
  auto rightSize = builder.create<mlir::arith::SelectOp>(loc, shiftIsPositive,
                                                         elseSize, absShift);
  auto hasUnchanged = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, size, bitSize);
  auto unchangedTmp1 = builder.create<mlir::arith::ShRUIOp>(loc, I, size);
  auto unchangedTmp2 =
      builder.create<mlir::arith::ShLIOp>(loc, unchangedTmp1, size);
  auto unchanged = builder.create<mlir::arith::SelectOp>(loc, hasUnchanged,
                                                         unchangedTmp2, zero);
  auto leftMaskShift =
      builder.create<mlir::arith::SubIOp>(loc, bitSize, leftSize);
  auto leftMask =
      builder.create<mlir::arith::ShRUIOp>(loc, ones, leftMaskShift);
  auto leftTmp = builder.create<mlir::arith::ShRUIOp>(loc, I, rightSize);
  auto left = builder.create<mlir::arith::AndIOp>(loc, leftTmp, leftMask);
  auto rightMaskShift =
      builder.create<mlir::arith::SubIOp>(loc, bitSize, rightSize);
  auto rightMask =
      builder.create<mlir::arith::ShRUIOp>(loc, ones, rightMaskShift);
  auto rightTmp = builder.create<mlir::arith::AndIOp>(loc, I, rightMask);
  auto right = builder.create<mlir::arith::ShLIOp>(loc, rightTmp, leftSize);
  auto resTmp = builder.create<mlir::arith::OrIOp>(loc, unchanged, left);
  auto res = builder.create<mlir::arith::OrIOp>(loc, resTmp, right);
  return builder.create<mlir::arith::SelectOp>(loc, shiftIsNop, I, res);
}

// LEADZ
mlir::Value IntrinsicLibrary::genLeadz(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);

  mlir::Value result =
      builder.create<mlir::math::CountLeadingZerosOp>(loc, args);

  return builder.createConvert(loc, resultType, result);
}

// LEN
// Note that this is only used for an unrestricted intrinsic LEN call.
// Other uses of LEN are rewritten as descriptor inquiries by the front-end.
fir::ExtendedValue
IntrinsicLibrary::genLen(mlir::Type resultType,
                         llvm::ArrayRef<fir::ExtendedValue> args) {
  // Optional KIND argument reflected in result type and otherwise ignored.
  assert(args.size() == 1 || args.size() == 2);
  mlir::Value len = fir::factory::readCharLen(builder, loc, args[0]);
  return builder.createConvert(loc, resultType, len);
}

// LEN_TRIM
fir::ExtendedValue
IntrinsicLibrary::genLenTrim(mlir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  // Optional KIND argument reflected in result type and otherwise ignored.
  assert(args.size() == 1 || args.size() == 2);
  const fir::CharBoxValue *charBox = args[0].getCharBox();
  if (!charBox)
    TODO(loc, "intrinsic: len_trim for character array");
  auto len =
      fir::factory::CharacterExprHelper(builder, loc).createLenTrim(*charBox);
  return builder.createConvert(loc, resultType, len);
}

// LGE, LGT, LLE, LLT
template <mlir::arith::CmpIPredicate pred>
fir::ExtendedValue
IntrinsicLibrary::genCharacterCompare(mlir::Type resultType,
                                      llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  return fir::runtime::genCharCompare(
      builder, loc, pred, fir::getBase(args[0]), fir::getLen(args[0]),
      fir::getBase(args[1]), fir::getLen(args[1]));
}

// LOC
fir::ExtendedValue
IntrinsicLibrary::genLoc(mlir::Type resultType,
                         llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  mlir::Value argValue = fir::getBase(args[0]);
  assert(fir::isa_box_type(argValue.getType()) &&
         "argument must have been lowered to box type");
  bool isFunc = argValue.getType().isa<fir::BoxProcType>();
  mlir::Value argAddr = getAddrFromBox(builder, loc, args[0], isFunc);
  return builder.createConvert(loc, fir::unwrapRefType(resultType), argAddr);
}

// MASKL, MASKR
template <typename Shift>
mlir::Value IntrinsicLibrary::genMask(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);

  mlir::Value zero = builder.createIntegerConstant(loc, resultType, 0);
  mlir::Value ones = builder.createIntegerConstant(loc, resultType, -1);
  mlir::Value bitSize = builder.createIntegerConstant(
      loc, resultType, resultType.getIntOrFloatBitWidth());
  mlir::Value bitsToSet = builder.createConvert(loc, resultType, args[0]);

  // The standard does not specify what to return if the number of bits to be
  // set, I < 0 or I >= BIT_SIZE(KIND). The shift instruction used below will
  // produce a poison value which may return a possibly platform-specific and/or
  // non-deterministic result. Other compilers don't produce a consistent result
  // in this case either, so we choose the most efficient implementation.
  mlir::Value shift =
      builder.create<mlir::arith::SubIOp>(loc, bitSize, bitsToSet);
  mlir::Value shifted = builder.create<Shift>(loc, ones, shift);
  mlir::Value isZero = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, bitsToSet, zero);

  return builder.create<mlir::arith::SelectOp>(loc, isZero, zero, shifted);
}

// MATMUL
fir::ExtendedValue
IntrinsicLibrary::genMatmul(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);

  // Handle required matmul arguments
  fir::BoxValue matrixTmpA = builder.createBox(loc, args[0]);
  mlir::Value matrixA = fir::getBase(matrixTmpA);
  fir::BoxValue matrixTmpB = builder.createBox(loc, args[1]);
  mlir::Value matrixB = fir::getBase(matrixTmpB);
  unsigned resultRank =
      (matrixTmpA.rank() == 1 || matrixTmpB.rank() == 1) ? 1 : 2;

  // Create mutable fir.box to be passed to the runtime for the result.
  mlir::Type resultArrayType = builder.getVarLenSeqTy(resultType, resultRank);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genMatmul(builder, loc, resultIrBox, matrixA, matrixB);
  // Read result from mutable fir.box and add it to the list of temps to be
  // finalized by the StatementContext.
  return readAndAddCleanUp(resultMutableBox, resultType, "MATMUL");
}

// MERGE
fir::ExtendedValue
IntrinsicLibrary::genMerge(mlir::Type,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  mlir::Value tsource = fir::getBase(args[0]);
  mlir::Value fsource = fir::getBase(args[1]);
  mlir::Value rawMask = fir::getBase(args[2]);
  mlir::Type type0 = fir::unwrapRefType(tsource.getType());
  bool isCharRslt = fir::isa_char(type0); // result is same as first argument
  mlir::Value mask = builder.createConvert(loc, builder.getI1Type(), rawMask);
  // FSOURCE has the same type as TSOURCE, but they may not have the same MLIR
  // types (one can have dynamic length while the other has constant lengths,
  // or one may be a fir.logical<> while the other is an i1). Insert a cast to
  // fulfill mlir::SelectOp constraint that the MLIR types must be the same.
  mlir::Value fsourceCast =
      builder.createConvert(loc, tsource.getType(), fsource);
  auto rslt =
      builder.create<mlir::arith::SelectOp>(loc, mask, tsource, fsourceCast);
  if (isCharRslt) {
    // Need a CharBoxValue for character results
    const fir::CharBoxValue *charBox = args[0].getCharBox();
    fir::CharBoxValue charRslt(rslt, charBox->getLen());
    return charRslt;
  }
  return rslt;
}

// MERGE_BITS
mlir::Value IntrinsicLibrary::genMergeBits(mlir::Type resultType,
                                           llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 3);

  mlir::Value i = builder.createConvert(loc, resultType, args[0]);
  mlir::Value j = builder.createConvert(loc, resultType, args[1]);
  mlir::Value mask = builder.createConvert(loc, resultType, args[2]);
  mlir::Value ones = builder.createIntegerConstant(loc, resultType, -1);

  // MERGE_BITS(I, J, MASK) = IOR(IAND(I, MASK), IAND(J, NOT(MASK)))
  mlir::Value notMask = builder.create<mlir::arith::XOrIOp>(loc, mask, ones);
  mlir::Value lft = builder.create<mlir::arith::AndIOp>(loc, i, mask);
  mlir::Value rgt = builder.create<mlir::arith::AndIOp>(loc, j, notMask);

  return builder.create<mlir::arith::OrIOp>(loc, lft, rgt);
}

// MOD
mlir::Value IntrinsicLibrary::genMod(mlir::Type resultType,
                                     llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  if (resultType.isa<mlir::IntegerType>())
    return builder.create<mlir::arith::RemSIOp>(loc, args[0], args[1]);

  // Use runtime.
  return builder.createConvert(
      loc, resultType, fir::runtime::genMod(builder, loc, args[0], args[1]));
}

// MODULO
mlir::Value IntrinsicLibrary::genModulo(mlir::Type resultType,
                                        llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  // No floored modulo op in LLVM/MLIR yet. TODO: add one to MLIR.
  // In the meantime, use a simple inlined implementation based on truncated
  // modulo (MOD(A, P) implemented by RemIOp, RemFOp). This avoids making manual
  // division and multiplication from MODULO formula.
  //  - If A/P > 0 or MOD(A,P)=0, then INT(A/P) = FLOOR(A/P), and MODULO = MOD.
  //  - Otherwise, when A/P < 0 and MOD(A,P) !=0, then MODULO(A, P) =
  //    A-FLOOR(A/P)*P = A-(INT(A/P)-1)*P = A-INT(A/P)*P+P = MOD(A,P)+P
  // Note that A/P < 0 if and only if A and P signs are different.
  if (resultType.isa<mlir::IntegerType>()) {
    auto remainder =
        builder.create<mlir::arith::RemSIOp>(loc, args[0], args[1]);
    auto argXor = builder.create<mlir::arith::XOrIOp>(loc, args[0], args[1]);
    mlir::Value zero = builder.createIntegerConstant(loc, argXor.getType(), 0);
    auto argSignDifferent = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, argXor, zero);
    auto remainderIsNotZero = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, remainder, zero);
    auto mustAddP = builder.create<mlir::arith::AndIOp>(loc, remainderIsNotZero,
                                                        argSignDifferent);
    auto remPlusP =
        builder.create<mlir::arith::AddIOp>(loc, remainder, args[1]);
    return builder.create<mlir::arith::SelectOp>(loc, mustAddP, remPlusP,
                                                 remainder);
  }
  // Real case
  if (resultType == mlir::FloatType::getF128(builder.getContext()))

    TODO(loc, "intrinsic: modulo for floating point of KIND=16");
  auto remainder = builder.create<mlir::arith::RemFOp>(loc, args[0], args[1]);
  mlir::Value zero = builder.createRealZeroConstant(loc, remainder.getType());
  auto remainderIsNotZero = builder.create<mlir::arith::CmpFOp>(
      loc, mlir::arith::CmpFPredicate::UNE, remainder, zero);
  auto aLessThanZero = builder.create<mlir::arith::CmpFOp>(
      loc, mlir::arith::CmpFPredicate::OLT, args[0], zero);
  auto pLessThanZero = builder.create<mlir::arith::CmpFOp>(
      loc, mlir::arith::CmpFPredicate::OLT, args[1], zero);
  auto argSignDifferent =
      builder.create<mlir::arith::XOrIOp>(loc, aLessThanZero, pLessThanZero);
  auto mustAddP = builder.create<mlir::arith::AndIOp>(loc, remainderIsNotZero,
                                                      argSignDifferent);
  auto remPlusP = builder.create<mlir::arith::AddFOp>(loc, remainder, args[1]);
  return builder.create<mlir::arith::SelectOp>(loc, mustAddP, remPlusP,
                                               remainder);
}

void IntrinsicLibrary::genMoveAlloc(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);

  const fir::ExtendedValue &from = args[0];
  const fir::ExtendedValue &to = args[1];
  const fir::ExtendedValue &status = args[2];
  const fir::ExtendedValue &errMsg = args[3];

  mlir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());
  mlir::Value errBox =
      isStaticallyPresent(errMsg)
          ? fir::getBase(errMsg)
          : builder.create<fir::AbsentOp>(loc, boxNoneTy).getResult();

  const fir::MutableBoxValue *fromBox = from.getBoxOf<fir::MutableBoxValue>();
  const fir::MutableBoxValue *toBox = to.getBoxOf<fir::MutableBoxValue>();

  assert(fromBox && toBox && "move_alloc parameters must be mutable arrays");

  mlir::Value fromAddr = fir::factory::getMutableIRBox(builder, loc, *fromBox);
  mlir::Value toAddr = fir::factory::getMutableIRBox(builder, loc, *toBox);

  mlir::Value hasStat = builder.createBool(loc, isStaticallyPresent(status));

  mlir::Value stat = fir::runtime::genMoveAlloc(builder, loc, toAddr, fromAddr,
                                                hasStat, errBox);

  fir::factory::syncMutableBoxFromIRBox(builder, loc, *fromBox);
  fir::factory::syncMutableBoxFromIRBox(builder, loc, *toBox);

  if (isStaticallyPresent(status)) {
    mlir::Value statAddr = fir::getBase(status);
    mlir::Value statIsPresentAtRuntime =
        builder.genIsNotNullAddr(loc, statAddr);
    builder.genIfThen(loc, statIsPresentAtRuntime)
        .genThen([&]() { builder.createStoreWithConvert(loc, stat, statAddr); })
        .end();
  }
}

// MVBITS
void IntrinsicLibrary::genMvbits(llvm::ArrayRef<fir::ExtendedValue> args) {
  // A conformant MVBITS(FROM,FROMPOS,LEN,TO,TOPOS) call satisfies:
  //     FROMPOS >= 0
  //     LEN >= 0
  //     TOPOS >= 0
  //     FROMPOS + LEN <= BIT_SIZE(FROM)
  //     TOPOS + LEN <= BIT_SIZE(TO)
  // MASK = -1 >> (BIT_SIZE(FROM) - LEN)
  // TO = LEN == 0 ? TO : ((!(MASK << TOPOS)) & TO) |
  //                      (((FROM >> FROMPOS) & MASK) << TOPOS)
  assert(args.size() == 5);
  auto unbox = [&](fir::ExtendedValue exv) {
    const mlir::Value *arg = exv.getUnboxed();
    assert(arg && "nonscalar mvbits argument");
    return *arg;
  };
  mlir::Value from = unbox(args[0]);
  mlir::Type resultType = from.getType();
  mlir::Value frompos = builder.createConvert(loc, resultType, unbox(args[1]));
  mlir::Value len = builder.createConvert(loc, resultType, unbox(args[2]));
  mlir::Value toAddr = unbox(args[3]);
  assert(fir::dyn_cast_ptrEleTy(toAddr.getType()) == resultType &&
         "mismatched mvbits types");
  auto to = builder.create<fir::LoadOp>(loc, resultType, toAddr);
  mlir::Value topos = builder.createConvert(loc, resultType, unbox(args[4]));
  mlir::Value zero = builder.createIntegerConstant(loc, resultType, 0);
  mlir::Value ones = builder.createIntegerConstant(loc, resultType, -1);
  mlir::Value bitSize = builder.createIntegerConstant(
      loc, resultType, resultType.cast<mlir::IntegerType>().getWidth());
  auto shiftCount = builder.create<mlir::arith::SubIOp>(loc, bitSize, len);
  auto mask = builder.create<mlir::arith::ShRUIOp>(loc, ones, shiftCount);
  auto unchangedTmp1 = builder.create<mlir::arith::ShLIOp>(loc, mask, topos);
  auto unchangedTmp2 =
      builder.create<mlir::arith::XOrIOp>(loc, unchangedTmp1, ones);
  auto unchanged = builder.create<mlir::arith::AndIOp>(loc, unchangedTmp2, to);
  auto frombitsTmp1 = builder.create<mlir::arith::ShRUIOp>(loc, from, frompos);
  auto frombitsTmp2 =
      builder.create<mlir::arith::AndIOp>(loc, frombitsTmp1, mask);
  auto frombits = builder.create<mlir::arith::ShLIOp>(loc, frombitsTmp2, topos);
  auto resTmp = builder.create<mlir::arith::OrIOp>(loc, unchanged, frombits);
  auto lenIsZero = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, len, zero);
  auto res = builder.create<mlir::arith::SelectOp>(loc, lenIsZero, to, resTmp);
  builder.create<fir::StoreOp>(loc, res, toAddr);
}

// NEAREST
mlir::Value IntrinsicLibrary::genNearest(mlir::Type resultType,
                                         llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);

  mlir::Value realX = fir::getBase(args[0]);
  mlir::Value realS = fir::getBase(args[1]);

  return builder.createConvert(
      loc, resultType, fir::runtime::genNearest(builder, loc, realX, realS));
}

// NINT
mlir::Value IntrinsicLibrary::genNint(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() >= 1);
  // Skip optional kind argument to search the runtime; it is already reflected
  // in result type.
  return genRuntimeCall("nint", resultType, {args[0]});
}

// NORM2
fir::ExtendedValue
IntrinsicLibrary::genNorm2(mlir::Type resultType,
                           llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);

  // Handle required array argument
  mlir::Value array = builder.createBox(loc, args[0]);
  unsigned rank = fir::BoxValue(array).rank();
  assert(rank >= 1);

  // Check if the dim argument is present
  bool absentDim = isStaticallyAbsent(args[1]);

  // If dim argument is absent or the array is rank 1, then the result is
  // a scalar (since the the result is rank-1 or 0). Otherwise, the result is
  // an array.
  if (absentDim || rank == 1) {
    return fir::runtime::genNorm2(builder, loc, array);
  } else {
    // Create mutable fir.box to be passed to the runtime for the result.
    mlir::Type resultArrayType = builder.getVarLenSeqTy(resultType, rank - 1);
    fir::MutableBoxValue resultMutableBox =
        fir::factory::createTempMutableBox(builder, loc, resultArrayType);
    mlir::Value resultIrBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

    mlir::Value dim = fir::getBase(args[1]);
    fir::runtime::genNorm2Dim(builder, loc, resultIrBox, array, dim);
    // Handle cleanup of allocatable result descriptor and return
    return readAndAddCleanUp(resultMutableBox, resultType, "NORM2");
  }
}

// NOT
mlir::Value IntrinsicLibrary::genNot(mlir::Type resultType,
                                     llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);
  mlir::Value allOnes = builder.createIntegerConstant(loc, resultType, -1);
  return builder.create<mlir::arith::XOrIOp>(loc, args[0], allOnes);
}

// NULL
fir::ExtendedValue
IntrinsicLibrary::genNull(mlir::Type, llvm::ArrayRef<fir::ExtendedValue> args) {
  // NULL() without MOLD must be handled in the contexts where it can appear
  // (see table 16.5 of Fortran 2018 standard).
  assert(args.size() == 1 && isStaticallyPresent(args[0]) &&
         "MOLD argument required to lower NULL outside of any context");
  const auto *mold = args[0].getBoxOf<fir::MutableBoxValue>();
  assert(mold && "MOLD must be a pointer or allocatable");
  fir::BaseBoxType boxType = mold->getBoxTy();
  mlir::Value boxStorage = builder.createTemporary(loc, boxType);
  mlir::Value box = fir::factory::createUnallocatedBox(
      builder, loc, boxType, mold->nonDeferredLenParams());
  builder.create<fir::StoreOp>(loc, box, boxStorage);
  return fir::MutableBoxValue(boxStorage, mold->nonDeferredLenParams(), {});
}

// PACK
fir::ExtendedValue
IntrinsicLibrary::genPack(mlir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  [[maybe_unused]] auto numArgs = args.size();
  assert(numArgs == 2 || numArgs == 3);

  // Handle required array argument
  mlir::Value array = builder.createBox(loc, args[0]);

  // Handle required mask argument
  mlir::Value mask = builder.createBox(loc, args[1]);

  // Handle optional vector argument
  mlir::Value vector = isStaticallyAbsent(args, 2)
                           ? builder.create<fir::AbsentOp>(
                                 loc, fir::BoxType::get(builder.getI1Type()))
                           : builder.createBox(loc, args[2]);

  // Create mutable fir.box to be passed to the runtime for the result.
  mlir::Type resultArrayType = builder.getVarLenSeqTy(resultType, 1);
  fir::MutableBoxValue resultMutableBox = fir::factory::createTempMutableBox(
      builder, loc, resultArrayType, {},
      fir::isPolymorphicType(array.getType()) ? array : mlir::Value{});
  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genPack(builder, loc, resultIrBox, array, mask, vector);

  return readAndAddCleanUp(resultMutableBox, resultType, "PACK");
}

// PARITY
fir::ExtendedValue
IntrinsicLibrary::genParity(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 2);
  // Handle required mask argument
  mlir::Value mask = builder.createBox(loc, args[0]);

  fir::BoxValue maskArry = builder.createBox(loc, args[0]);
  int rank = maskArry.rank();
  assert(rank >= 1);

  // Handle optional dim argument
  bool absentDim = isStaticallyAbsent(args[1]);
  mlir::Value dim =
      absentDim ? builder.createIntegerConstant(loc, builder.getIndexType(), 1)
                : fir::getBase(args[1]);

  if (rank == 1 || absentDim)
    return builder.createConvert(
        loc, resultType, fir::runtime::genParity(builder, loc, mask, dim));

  // else use the result descriptor ParityDim() intrinsic

  // Create mutable fir.box to be passed to the runtime for the result.

  mlir::Type resultArrayType = builder.getVarLenSeqTy(resultType, rank - 1);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  // Call runtime. The runtime is allocating the result.
  fir::runtime::genParityDescriptor(builder, loc, resultIrBox, mask, dim);
  return readAndAddCleanUp(resultMutableBox, resultType, "PARITY");
}

// POPCNT
mlir::Value IntrinsicLibrary::genPopcnt(mlir::Type resultType,
                                        llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);

  mlir::Value count = builder.create<mlir::math::CtPopOp>(loc, args);

  return builder.createConvert(loc, resultType, count);
}

// POPPAR
mlir::Value IntrinsicLibrary::genPoppar(mlir::Type resultType,
                                        llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);

  mlir::Value count = genPopcnt(resultType, args);
  mlir::Value one = builder.createIntegerConstant(loc, resultType, 1);

  return builder.create<mlir::arith::AndIOp>(loc, count, one);
}

// PRESENT
fir::ExtendedValue
IntrinsicLibrary::genPresent(mlir::Type,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  return builder.create<fir::IsPresentOp>(loc, builder.getI1Type(),
                                          fir::getBase(args[0]));
}

// PRODUCT
fir::ExtendedValue
IntrinsicLibrary::genProduct(mlir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  return genReduction(fir::runtime::genProduct, fir::runtime::genProductDim,
                      "PRODUCT", resultType, args);
}

// RANDOM_INIT
void IntrinsicLibrary::genRandomInit(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  fir::runtime::genRandomInit(builder, loc, fir::getBase(args[0]),
                              fir::getBase(args[1]));
}

// RANDOM_NUMBER
void IntrinsicLibrary::genRandomNumber(
    llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  fir::runtime::genRandomNumber(builder, loc, fir::getBase(args[0]));
}

// RANDOM_SEED
void IntrinsicLibrary::genRandomSeed(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  mlir::Type boxNoneTy = fir::BoxType::get(builder.getNoneType());
  auto getDesc = [&](int i) {
    return isStaticallyPresent(args[i])
               ? fir::getBase(args[i])
               : builder.create<fir::AbsentOp>(loc, boxNoneTy).getResult();
  };
  mlir::Value size = getDesc(0);
  mlir::Value put = getDesc(1);
  mlir::Value get = getDesc(2);
  fir::runtime::genRandomSeed(builder, loc, size, put, get);
}

// REDUCE
fir::ExtendedValue
IntrinsicLibrary::genReduce(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  TODO(loc, "intrinsic: reduce");
}

// REPEAT
fir::ExtendedValue
IntrinsicLibrary::genRepeat(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  mlir::Value string = builder.createBox(loc, args[0]);
  mlir::Value ncopies = fir::getBase(args[1]);
  // Create mutable fir.box to be passed to the runtime for the result.
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultType);
  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genRepeat(builder, loc, resultIrBox, string, ncopies);
  // Read result from mutable fir.box and add it to the list of temps to be
  // finalized by the StatementContext.
  return readAndAddCleanUp(resultMutableBox, resultType, "REPEAT");
}

// RESHAPE
fir::ExtendedValue
IntrinsicLibrary::genReshape(mlir::Type resultType,
                             llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 4);

  // Handle source argument
  mlir::Value source = builder.createBox(loc, args[0]);

  // Handle shape argument
  mlir::Value shape = builder.createBox(loc, args[1]);
  assert(fir::BoxValue(shape).rank() == 1);
  mlir::Type shapeTy = shape.getType();
  mlir::Type shapeArrTy = fir::dyn_cast_ptrOrBoxEleTy(shapeTy);
  auto resultRank = shapeArrTy.cast<fir::SequenceType>().getShape()[0];

  if (resultRank == fir::SequenceType::getUnknownExtent())
    TODO(loc, "intrinsic: reshape requires computing rank of result");

  // Handle optional pad argument
  mlir::Value pad = isStaticallyAbsent(args[2])
                        ? builder.create<fir::AbsentOp>(
                              loc, fir::BoxType::get(builder.getI1Type()))
                        : builder.createBox(loc, args[2]);

  // Handle optional order argument
  mlir::Value order = isStaticallyAbsent(args[3])
                          ? builder.create<fir::AbsentOp>(
                                loc, fir::BoxType::get(builder.getI1Type()))
                          : builder.createBox(loc, args[3]);

  // Create mutable fir.box to be passed to the runtime for the result.
  mlir::Type type = builder.getVarLenSeqTy(resultType, resultRank);
  fir::MutableBoxValue resultMutableBox = fir::factory::createTempMutableBox(
      builder, loc, type, {},
      fir::isPolymorphicType(source.getType()) ? source : mlir::Value{});

  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genReshape(builder, loc, resultIrBox, source, shape, pad,
                           order);

  return readAndAddCleanUp(resultMutableBox, resultType, "RESHAPE");
}

// RRSPACING
mlir::Value IntrinsicLibrary::genRRSpacing(mlir::Type resultType,
                                           llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genRRSpacing(builder, loc, fir::getBase(args[0])));
}

// SAME_TYPE_AS
fir::ExtendedValue
IntrinsicLibrary::genSameTypeAs(mlir::Type resultType,
                                llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genSameTypeAs(builder, loc, fir::getBase(args[0]),
                                  fir::getBase(args[1])));
}

// SCALE
mlir::Value IntrinsicLibrary::genScale(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);

  mlir::Value realX = fir::getBase(args[0]);
  mlir::Value intI = fir::getBase(args[1]);

  return builder.createConvert(
      loc, resultType, fir::runtime::genScale(builder, loc, realX, intI));
}

// SCAN
fir::ExtendedValue
IntrinsicLibrary::genScan(mlir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 4);

  if (isStaticallyAbsent(args[3])) {
    // Kind not specified, so call scan/verify runtime routine that is
    // specialized on the kind of characters in string.

    // Handle required string base arg
    mlir::Value stringBase = fir::getBase(args[0]);

    // Handle required set string base arg
    mlir::Value setBase = fir::getBase(args[1]);

    // Handle kind argument; it is the kind of character in this case
    fir::KindTy kind =
        fir::factory::CharacterExprHelper{builder, loc}.getCharacterKind(
            stringBase.getType());

    // Get string length argument
    mlir::Value stringLen = fir::getLen(args[0]);

    // Get set string length argument
    mlir::Value setLen = fir::getLen(args[1]);

    // Handle optional back argument
    mlir::Value back =
        isStaticallyAbsent(args[2])
            ? builder.createIntegerConstant(loc, builder.getI1Type(), 0)
            : fir::getBase(args[2]);

    return builder.createConvert(loc, resultType,
                                 fir::runtime::genScan(builder, loc, kind,
                                                       stringBase, stringLen,
                                                       setBase, setLen, back));
  }
  // else use the runtime descriptor version of scan/verify

  // Handle optional argument, back
  auto makeRefThenEmbox = [&](mlir::Value b) {
    fir::LogicalType logTy = fir::LogicalType::get(
        builder.getContext(), builder.getKindMap().defaultLogicalKind());
    mlir::Value temp = builder.createTemporary(loc, logTy);
    mlir::Value castb = builder.createConvert(loc, logTy, b);
    builder.create<fir::StoreOp>(loc, castb, temp);
    return builder.createBox(loc, temp);
  };
  mlir::Value back = fir::isUnboxedValue(args[2])
                         ? makeRefThenEmbox(*args[2].getUnboxed())
                         : builder.create<fir::AbsentOp>(
                               loc, fir::BoxType::get(builder.getI1Type()));

  // Handle required string argument
  mlir::Value string = builder.createBox(loc, args[0]);

  // Handle required set argument
  mlir::Value set = builder.createBox(loc, args[1]);

  // Handle kind argument
  mlir::Value kind = fir::getBase(args[3]);

  // Create result descriptor
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultType);
  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genScanDescriptor(builder, loc, resultIrBox, string, set, back,
                                  kind);

  // Handle cleanup of allocatable result descriptor and return
  return readAndAddCleanUp(resultMutableBox, resultType, "SCAN");
}

// SELECTED_INT_KIND
mlir::Value
IntrinsicLibrary::genSelectedIntKind(mlir::Type resultType,
                                     llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genSelectedIntKind(builder, loc, fir::getBase(args[0])));
}

// SELECTED_REAL_KIND
mlir::Value
IntrinsicLibrary::genSelectedRealKind(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 3);

  // Handle optional precision(P) argument
  mlir::Value precision =
      isStaticallyAbsent(args[0])
          ? builder.create<fir::AbsentOp>(
                loc, fir::ReferenceType::get(builder.getI1Type()))
          : fir::getBase(args[0]);

  // Handle optional range(R) argument
  mlir::Value range =
      isStaticallyAbsent(args[1])
          ? builder.create<fir::AbsentOp>(
                loc, fir::ReferenceType::get(builder.getI1Type()))
          : fir::getBase(args[1]);

  // Handle optional radix(RADIX) argument
  mlir::Value radix =
      isStaticallyAbsent(args[2])
          ? builder.create<fir::AbsentOp>(
                loc, fir::ReferenceType::get(builder.getI1Type()))
          : fir::getBase(args[2]);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genSelectedRealKind(builder, loc, precision, range, radix));
}

// SET_EXPONENT
mlir::Value IntrinsicLibrary::genSetExponent(mlir::Type resultType,
                                             llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genSetExponent(builder, loc, fir::getBase(args[0]),
                                   fir::getBase(args[1])));
}

// SHIFTL, SHIFTR
template <typename Shift>
mlir::Value IntrinsicLibrary::genShift(mlir::Type resultType,
                                       llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);

  // If SHIFT < 0 or SHIFT >= BIT_SIZE(I), return 0. This is not required by
  // the standard. However, several other compilers behave this way, so try and
  // maintain compatibility with them to an extent.

  unsigned bits = resultType.getIntOrFloatBitWidth();
  mlir::Value bitSize = builder.createIntegerConstant(loc, resultType, bits);
  mlir::Value zero = builder.createIntegerConstant(loc, resultType, 0);
  mlir::Value shift = builder.createConvert(loc, resultType, args[1]);

  mlir::Value tooSmall = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, shift, zero);
  mlir::Value tooLarge = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::sge, shift, bitSize);
  mlir::Value outOfBounds =
      builder.create<mlir::arith::OrIOp>(loc, tooSmall, tooLarge);

  mlir::Value shifted = builder.create<Shift>(loc, args[0], shift);
  return builder.create<mlir::arith::SelectOp>(loc, outOfBounds, zero, shifted);
}

// SHIFTA
mlir::Value IntrinsicLibrary::genShiftA(mlir::Type resultType,
                                        llvm::ArrayRef<mlir::Value> args) {
  unsigned bits = resultType.getIntOrFloatBitWidth();
  mlir::Value bitSize = builder.createIntegerConstant(loc, resultType, bits);
  mlir::Value shift = builder.createConvert(loc, resultType, args[1]);
  mlir::Value shiftEqBitSize = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, shift, bitSize);

  // Lowering of mlir::arith::ShRSIOp is using `ashr`. `ashr` is undefined when
  // the shift amount is equal to the element size.
  // So if SHIFT is equal to the bit width then it is handled as a special case.
  mlir::Value zero = builder.createIntegerConstant(loc, resultType, 0);
  mlir::Value minusOne = builder.createIntegerConstant(loc, resultType, -1);
  mlir::Value valueIsNeg = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, args[0], zero);
  mlir::Value specialRes =
      builder.create<mlir::arith::SelectOp>(loc, valueIsNeg, minusOne, zero);

  mlir::Value shifted =
      builder.create<mlir::arith::ShRSIOp>(loc, args[0], shift);
  return builder.create<mlir::arith::SelectOp>(loc, shiftEqBitSize, specialRes,
                                               shifted);
}

// SIGN
mlir::Value IntrinsicLibrary::genSign(mlir::Type resultType,
                                      llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 2);
  if (resultType.isa<mlir::IntegerType>()) {
    mlir::Value abs = genAbs(resultType, {args[0]});
    mlir::Value zero = builder.createIntegerConstant(loc, resultType, 0);
    auto neg = builder.create<mlir::arith::SubIOp>(loc, zero, abs);
    auto cmp = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::slt, args[1], zero);
    return builder.create<mlir::arith::SelectOp>(loc, cmp, neg, abs);
  }
  return genRuntimeCall("sign", resultType, args);
}

// SIZE
fir::ExtendedValue
IntrinsicLibrary::genSize(mlir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  // Note that the value of the KIND argument is already reflected in the
  // resultType
  assert(args.size() == 3);
  if (const auto *boxValue = args[0].getBoxOf<fir::BoxValue>())
    if (boxValue->hasAssumedRank())
      TODO(loc, "intrinsic: size with assumed rank argument");

  // Get the ARRAY argument
  mlir::Value array = builder.createBox(loc, args[0]);

  // The front-end rewrites SIZE without the DIM argument to
  // an array of SIZE with DIM in most cases, but it may not be
  // possible in some cases like when in SIZE(function_call()).
  if (isStaticallyAbsent(args, 1))
    return builder.createConvert(loc, resultType,
                                 fir::runtime::genSize(builder, loc, array));

  // Get the DIM argument.
  mlir::Value dim = fir::getBase(args[1]);
  if (!fir::isa_ref_type(dim.getType()))
    return builder.createConvert(
        loc, resultType, fir::runtime::genSizeDim(builder, loc, array, dim));

  mlir::Value isDynamicallyAbsent = builder.genIsNullAddr(loc, dim);
  return builder
      .genIfOp(loc, {resultType}, isDynamicallyAbsent,
               /*withElseRegion=*/true)
      .genThen([&]() {
        mlir::Value size = builder.createConvert(
            loc, resultType, fir::runtime::genSize(builder, loc, array));
        builder.create<fir::ResultOp>(loc, size);
      })
      .genElse([&]() {
        mlir::Value dimValue = builder.create<fir::LoadOp>(loc, dim);
        mlir::Value size = builder.createConvert(
            loc, resultType,
            fir::runtime::genSizeDim(builder, loc, array, dimValue));
        builder.create<fir::ResultOp>(loc, size);
      })
      .getResults()[0];
}

// TRAILZ
mlir::Value IntrinsicLibrary::genTrailz(mlir::Type resultType,
                                        llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);

  mlir::Value result =
      builder.create<mlir::math::CountTrailingZerosOp>(loc, args);

  return builder.createConvert(loc, resultType, result);
}

static bool hasDefaultLowerBound(const fir::ExtendedValue &exv) {
  return exv.match(
      [](const fir::ArrayBoxValue &arr) { return arr.getLBounds().empty(); },
      [](const fir::CharArrayBoxValue &arr) {
        return arr.getLBounds().empty();
      },
      [](const fir::BoxValue &arr) { return arr.getLBounds().empty(); },
      [](const auto &) { return false; });
}

/// Compute the lower bound in dimension \p dim (zero based) of \p array
/// taking care of returning one when the related extent is zero.
static mlir::Value computeLBOUND(fir::FirOpBuilder &builder, mlir::Location loc,
                                 const fir::ExtendedValue &array, unsigned dim,
                                 mlir::Value zero, mlir::Value one) {
  assert(dim < array.rank() && "invalid dimension");
  if (hasDefaultLowerBound(array))
    return one;
  mlir::Value lb = fir::factory::readLowerBound(builder, loc, array, dim, one);
  if (dim + 1 == array.rank() && array.isAssumedSize())
    return lb;
  mlir::Value extent = fir::factory::readExtent(builder, loc, array, dim);
  zero = builder.createConvert(loc, extent.getType(), zero);
  auto dimIsEmpty = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, extent, zero);
  one = builder.createConvert(loc, lb.getType(), one);
  return builder.create<mlir::arith::SelectOp>(loc, dimIsEmpty, one, lb);
}

/// Create a fir.box to be passed to the LBOUND/UBOUND runtime.
/// This ensure that local lower bounds of assumed shape are propagated and that
/// a fir.box with equivalent LBOUNDs but an explicit shape is created for
/// assumed size arrays to avoid undefined behaviors in codegen or the runtime.
static mlir::Value
createBoxForRuntimeBoundInquiry(mlir::Location loc, fir::FirOpBuilder &builder,
                                const fir::ExtendedValue &array) {
  if (!array.isAssumedSize())
    return array.match(
        [&](const fir::BoxValue &boxValue) -> mlir::Value {
          // This entity is mapped to a fir.box that may not contain the local
          // lower bound information if it is a dummy. Rebox it with the local
          // shape information.
          mlir::Value localShape = builder.createShape(loc, array);
          mlir::Value oldBox = boxValue.getAddr();
          return builder.create<fir::ReboxOp>(loc, oldBox.getType(), oldBox,
                                              localShape,
                                              /*slice=*/mlir::Value{});
        },
        [&](const auto &) -> mlir::Value {
          // This a pointer/allocatable, or an entity not yet tracked with a
          // fir.box. For pointer/allocatable, createBox will forward the
          // descriptor that contains the correct lower bound information. For
          // other entities, a new fir.box will be made with the local lower
          // bounds.
          return builder.createBox(loc, array);
        });
  // Assumed sized are not meant to be emboxed. This could cause the undefined
  // extent cannot safely be understood by the runtime/codegen that will
  // consider that the dimension is empty and that the related LBOUND value must
  // be one. Pretend that the related extent is one to get the correct LBOUND
  // value.
  llvm::SmallVector<mlir::Value> shape =
      fir::factory::getExtents(loc, builder, array);
  assert(!shape.empty() && "assumed size must have at least one dimension");
  shape.back() = builder.createIntegerConstant(loc, builder.getIndexType(), 1);
  auto safeToEmbox = array.match(
      [&](const fir::CharArrayBoxValue &x) -> fir::ExtendedValue {
        return fir::CharArrayBoxValue{x.getAddr(), x.getLen(), shape,
                                      x.getLBounds()};
      },
      [&](const fir::ArrayBoxValue &x) -> fir::ExtendedValue {
        return fir::ArrayBoxValue{x.getAddr(), shape, x.getLBounds()};
      },
      [&](const auto &) -> fir::ExtendedValue {
        fir::emitFatalError(loc, "not an assumed size array");
      });
  return builder.createBox(loc, safeToEmbox);
}

// LBOUND
fir::ExtendedValue
IntrinsicLibrary::genLbound(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2 || args.size() == 3);
  const fir::ExtendedValue &array = args[0];
  if (const auto *boxValue = array.getBoxOf<fir::BoxValue>())
    if (boxValue->hasAssumedRank())
      TODO(loc, "intrinsic: lbound with assumed rank argument");

  //===----------------------------------------------------------------------===//
  mlir::Type indexType = builder.getIndexType();

  // Semantics builds signatures for LBOUND calls as either
  // LBOUND(array, dim, [kind]) or LBOUND(array, [kind]).
  if (args.size() == 2 || isStaticallyAbsent(args, 1)) {
    // DIM is absent.
    mlir::Type lbType = fir::unwrapSequenceType(resultType);
    unsigned rank = array.rank();
    mlir::Type lbArrayType = fir::SequenceType::get(
        {static_cast<fir::SequenceType::Extent>(array.rank())}, lbType);
    mlir::Value lbArray = builder.createTemporary(loc, lbArrayType);
    mlir::Type lbAddrType = builder.getRefType(lbType);
    mlir::Value one = builder.createIntegerConstant(loc, lbType, 1);
    mlir::Value zero = builder.createIntegerConstant(loc, indexType, 0);
    for (unsigned dim = 0; dim < rank; ++dim) {
      mlir::Value lb = computeLBOUND(builder, loc, array, dim, zero, one);
      lb = builder.createConvert(loc, lbType, lb);
      auto index = builder.createIntegerConstant(loc, indexType, dim);
      auto lbAddr =
          builder.create<fir::CoordinateOp>(loc, lbAddrType, lbArray, index);
      builder.create<fir::StoreOp>(loc, lb, lbAddr);
    }
    mlir::Value lbArrayExtent =
        builder.createIntegerConstant(loc, indexType, rank);
    llvm::SmallVector<mlir::Value> extents{lbArrayExtent};
    return fir::ArrayBoxValue{lbArray, extents};
  }
  // DIM is present.
  mlir::Value dim = fir::getBase(args[1]);

  // If it is a compile time constant, skip the runtime call.
  if (std::optional<std::int64_t> cstDim = fir::getIntIfConstant(dim)) {
    mlir::Value one = builder.createIntegerConstant(loc, resultType, 1);
    mlir::Value zero = builder.createIntegerConstant(loc, indexType, 0);
    mlir::Value lb = computeLBOUND(builder, loc, array, *cstDim - 1, zero, one);
    return builder.createConvert(loc, resultType, lb);
  }

  fir::ExtendedValue box = createBoxForRuntimeBoundInquiry(loc, builder, array);
  return builder.createConvert(
      loc, resultType,
      fir::runtime::genLboundDim(builder, loc, fir::getBase(box), dim));
}

// UBOUND
fir::ExtendedValue
IntrinsicLibrary::genUbound(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3 || args.size() == 2);
  if (args.size() == 3) {
    // Handle calls to UBOUND with the DIM argument, which return a scalar
    mlir::Value extent = fir::getBase(genSize(resultType, args));
    mlir::Value lbound = fir::getBase(genLbound(resultType, args));

    mlir::Value one = builder.createIntegerConstant(loc, resultType, 1);
    mlir::Value ubound = builder.create<mlir::arith::SubIOp>(loc, lbound, one);
    return builder.create<mlir::arith::AddIOp>(loc, ubound, extent);
  } else {
    // Handle calls to UBOUND without the DIM argument, which return an array
    mlir::Value kind = isStaticallyAbsent(args[1])
                           ? builder.createIntegerConstant(
                                 loc, builder.getIndexType(),
                                 builder.getKindMap().defaultIntegerKind())
                           : fir::getBase(args[1]);

    // Create mutable fir.box to be passed to the runtime for the result.
    mlir::Type type = builder.getVarLenSeqTy(resultType, /*rank=*/1);
    fir::MutableBoxValue resultMutableBox =
        fir::factory::createTempMutableBox(builder, loc, type);
    mlir::Value resultIrBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

    fir::ExtendedValue box =
        createBoxForRuntimeBoundInquiry(loc, builder, args[0]);
    fir::runtime::genUbound(builder, loc, resultIrBox, fir::getBase(box), kind);

    return readAndAddCleanUp(resultMutableBox, resultType, "UBOUND");
  }
  return mlir::Value();
}

// SPACING
mlir::Value IntrinsicLibrary::genSpacing(mlir::Type resultType,
                                         llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() == 1);

  return builder.createConvert(
      loc, resultType,
      fir::runtime::genSpacing(builder, loc, fir::getBase(args[0])));
}

// SPREAD
fir::ExtendedValue
IntrinsicLibrary::genSpread(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 3);

  // Handle source argument
  mlir::Value source = builder.createBox(loc, args[0]);
  fir::BoxValue sourceTmp = source;
  unsigned sourceRank = sourceTmp.rank();

  // Handle Dim argument
  mlir::Value dim = fir::getBase(args[1]);

  // Handle ncopies argument
  mlir::Value ncopies = fir::getBase(args[2]);

  // Generate result descriptor
  mlir::Type resultArrayType =
      builder.getVarLenSeqTy(resultType, sourceRank + 1);
  fir::MutableBoxValue resultMutableBox = fir::factory::createTempMutableBox(
      builder, loc, resultArrayType, {},
      fir::isPolymorphicType(source.getType()) ? source : mlir::Value{});
  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genSpread(builder, loc, resultIrBox, source, dim, ncopies);

  return readAndAddCleanUp(resultMutableBox, resultType, "SPREAD");
}

// STORAGE_SIZE
fir::ExtendedValue
IntrinsicLibrary::genStorageSize(mlir::Type resultType,
                                 llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2 || args.size() == 1);
  mlir::Value box = fir::getBase(args[0]);
  mlir::Type boxTy = box.getType();
  mlir::Type kindTy = builder.getDefaultIntegerType();
  bool needRuntimeCheck = false;
  std::string errorMsg;

  if (fir::isUnlimitedPolymorphicType(boxTy) &&
      (fir::isAllocatableType(boxTy) || fir::isPointerType(boxTy))) {
    needRuntimeCheck = true;
    errorMsg =
        fir::isPointerType(boxTy)
            ? "unlimited polymorphic disassociated POINTER in STORAGE_SIZE"
            : "unlimited polymorphic unallocated ALLOCATABLE in STORAGE_SIZE";
  } else if (fir::isPolymorphicType(boxTy) && fir::isPointerType(boxTy)) {
    needRuntimeCheck = true;
    errorMsg = "polymorphic disassociated POINTER in STORAGE_SIZE";
  }
  const fir::MutableBoxValue *mutBox = args[0].getBoxOf<fir::MutableBoxValue>();
  if (needRuntimeCheck && mutBox) {
    mlir::Value isNotAllocOrAssoc =
        fir::factory::genIsNotAllocatedOrAssociatedTest(builder, loc, *mutBox);
    builder.genIfThen(loc, isNotAllocOrAssoc)
        .genThen([&]() {
          fir::runtime::genReportFatalUserError(builder, loc, errorMsg);
        })
        .end();
  }

  // Handle optional kind argument
  bool absentKind = isStaticallyAbsent(args, 1);
  if (!absentKind) {
    mlir::Operation *defKind = fir::getBase(args[1]).getDefiningOp();
    assert(mlir::isa<mlir::arith::ConstantOp>(*defKind) &&
           "kind not a constant");
    auto constOp = mlir::dyn_cast<mlir::arith::ConstantOp>(*defKind);
    kindTy = builder.getIntegerType(
        builder.getKindMap().getIntegerBitsize(fir::toInt(constOp)));
  }

  if (box.getType().isa<fir::ReferenceType>())
    box = builder.create<fir::LoadOp>(loc, box);
  mlir::Value eleSize = builder.create<fir::BoxEleSizeOp>(loc, kindTy, box);
  mlir::Value c8 = builder.createIntegerConstant(loc, kindTy, 8);
  return builder.create<mlir::arith::MulIOp>(loc, eleSize, c8);
}

// SUM
fir::ExtendedValue
IntrinsicLibrary::genSum(mlir::Type resultType,
                         llvm::ArrayRef<fir::ExtendedValue> args) {
  return genReduction(fir::runtime::genSum, fir::runtime::genSumDim, "SUM",
                      resultType, args);
}

// SYSTEM_CLOCK
void IntrinsicLibrary::genSystemClock(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  fir::runtime::genSystemClock(builder, loc, fir::getBase(args[0]),
                               fir::getBase(args[1]), fir::getBase(args[2]));
}

// TRANSFER
fir::ExtendedValue
IntrinsicLibrary::genTransfer(mlir::Type resultType,
                              llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() >= 2); // args.size() == 2 when size argument is omitted.

  // Handle source argument
  mlir::Value source = builder.createBox(loc, args[0]);

  // Handle mold argument
  mlir::Value mold = builder.createBox(loc, args[1]);
  fir::BoxValue moldTmp = mold;
  unsigned moldRank = moldTmp.rank();

  bool absentSize = (args.size() == 2);

  // Create mutable fir.box to be passed to the runtime for the result.
  mlir::Type type = (moldRank == 0 && absentSize)
                        ? resultType
                        : builder.getVarLenSeqTy(resultType, 1);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, type);

  if (moldRank == 0 && absentSize) {
    // This result is a scalar in this case.
    mlir::Value resultIrBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

    fir::runtime::genTransfer(builder, loc, resultIrBox, source, mold);
  } else {
    // The result is a rank one array in this case.
    mlir::Value resultIrBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

    if (absentSize) {
      fir::runtime::genTransfer(builder, loc, resultIrBox, source, mold);
    } else {
      mlir::Value sizeArg = fir::getBase(args[2]);
      fir::runtime::genTransferSize(builder, loc, resultIrBox, source, mold,
                                    sizeArg);
    }
  }
  return readAndAddCleanUp(resultMutableBox, resultType, "TRANSFER");
}

// TRANSPOSE
fir::ExtendedValue
IntrinsicLibrary::genTranspose(mlir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 1);

  // Handle source argument
  mlir::Value source = builder.createBox(loc, args[0]);

  // Create mutable fir.box to be passed to the runtime for the result.
  mlir::Type resultArrayType = builder.getVarLenSeqTy(resultType, 2);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genTranspose(builder, loc, resultIrBox, source);
  // Read result from mutable fir.box and add it to the list of temps to be
  // finalized by the StatementContext.
  return readAndAddCleanUp(resultMutableBox, resultType, "TRANSPOSE");
}

// TRIM
fir::ExtendedValue
IntrinsicLibrary::genTrim(mlir::Type resultType,
                          llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  mlir::Value string = builder.createBox(loc, args[0]);
  // Create mutable fir.box to be passed to the runtime for the result.
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultType);
  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);
  // Call runtime. The runtime is allocating the result.
  fir::runtime::genTrim(builder, loc, resultIrBox, string);
  // Read result from mutable fir.box and add it to the list of temps to be
  // finalized by the StatementContext.
  return readAndAddCleanUp(resultMutableBox, resultType, "TRIM");
}

// Compare two FIR values and return boolean result as i1.
template <Extremum extremum, ExtremumBehavior behavior>
static mlir::Value createExtremumCompare(mlir::Location loc,
                                         fir::FirOpBuilder &builder,
                                         mlir::Value left, mlir::Value right) {
  static constexpr mlir::arith::CmpIPredicate integerPredicate =
      extremum == Extremum::Max ? mlir::arith::CmpIPredicate::sgt
                                : mlir::arith::CmpIPredicate::slt;
  static constexpr mlir::arith::CmpFPredicate orderedCmp =
      extremum == Extremum::Max ? mlir::arith::CmpFPredicate::OGT
                                : mlir::arith::CmpFPredicate::OLT;
  mlir::Type type = left.getType();
  mlir::Value result;
  if (fir::isa_real(type)) {
    // Note: the signaling/quit aspect of the result required by IEEE
    // cannot currently be obtained with LLVM without ad-hoc runtime.
    if constexpr (behavior == ExtremumBehavior::IeeeMinMaximumNumber) {
      // Return the number if one of the inputs is NaN and the other is
      // a number.
      auto leftIsResult =
          builder.create<mlir::arith::CmpFOp>(loc, orderedCmp, left, right);
      auto rightIsNan = builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::UNE, right, right);
      result =
          builder.create<mlir::arith::OrIOp>(loc, leftIsResult, rightIsNan);
    } else if constexpr (behavior == ExtremumBehavior::IeeeMinMaximum) {
      // Always return NaNs if one the input is NaNs
      auto leftIsResult =
          builder.create<mlir::arith::CmpFOp>(loc, orderedCmp, left, right);
      auto leftIsNan = builder.create<mlir::arith::CmpFOp>(
          loc, mlir::arith::CmpFPredicate::UNE, left, left);
      result = builder.create<mlir::arith::OrIOp>(loc, leftIsResult, leftIsNan);
    } else if constexpr (behavior == ExtremumBehavior::MinMaxss) {
      // If the left is a NaN, return the right whatever it is.
      result =
          builder.create<mlir::arith::CmpFOp>(loc, orderedCmp, left, right);
    } else if constexpr (behavior == ExtremumBehavior::PgfortranLlvm) {
      // If one of the operand is a NaN, return left whatever it is.
      static constexpr auto unorderedCmp =
          extremum == Extremum::Max ? mlir::arith::CmpFPredicate::UGT
                                    : mlir::arith::CmpFPredicate::ULT;
      result =
          builder.create<mlir::arith::CmpFOp>(loc, unorderedCmp, left, right);
    } else {
      // TODO: ieeeMinNum/ieeeMaxNum
      static_assert(behavior == ExtremumBehavior::IeeeMinMaxNum,
                    "ieeeMinNum/ieeeMaxNum behavior not implemented");
    }
  } else if (fir::isa_integer(type)) {
    result =
        builder.create<mlir::arith::CmpIOp>(loc, integerPredicate, left, right);
  } else if (fir::isa_char(type) || fir::isa_char(fir::unwrapRefType(type))) {
    // TODO: ! character min and max is tricky because the result
    // length is the length of the longest argument!
    // So we may need a temp.
    TODO(loc, "intrinsic: min and max for CHARACTER");
  }
  assert(result && "result must be defined");
  return result;
}

// UNPACK
fir::ExtendedValue
IntrinsicLibrary::genUnpack(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);

  // Handle required vector argument
  mlir::Value vector = builder.createBox(loc, args[0]);

  // Handle required mask argument
  fir::BoxValue maskBox = builder.createBox(loc, args[1]);
  mlir::Value mask = fir::getBase(maskBox);
  unsigned maskRank = maskBox.rank();

  // Handle required field argument
  mlir::Value field = builder.createBox(loc, args[2]);

  // Create mutable fir.box to be passed to the runtime for the result.
  mlir::Type resultArrayType = builder.getVarLenSeqTy(resultType, maskRank);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genUnpack(builder, loc, resultIrBox, vector, mask, field);

  return readAndAddCleanUp(resultMutableBox, resultType, "UNPACK");
}

// VERIFY
fir::ExtendedValue
IntrinsicLibrary::genVerify(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 4);

  if (isStaticallyAbsent(args[3])) {
    // Kind not specified, so call scan/verify runtime routine that is
    // specialized on the kind of characters in string.

    // Handle required string base arg
    mlir::Value stringBase = fir::getBase(args[0]);

    // Handle required set string base arg
    mlir::Value setBase = fir::getBase(args[1]);

    // Handle kind argument; it is the kind of character in this case
    fir::KindTy kind =
        fir::factory::CharacterExprHelper{builder, loc}.getCharacterKind(
            stringBase.getType());

    // Get string length argument
    mlir::Value stringLen = fir::getLen(args[0]);

    // Get set string length argument
    mlir::Value setLen = fir::getLen(args[1]);

    // Handle optional back argument
    mlir::Value back =
        isStaticallyAbsent(args[2])
            ? builder.createIntegerConstant(loc, builder.getI1Type(), 0)
            : fir::getBase(args[2]);

    return builder.createConvert(
        loc, resultType,
        fir::runtime::genVerify(builder, loc, kind, stringBase, stringLen,
                                setBase, setLen, back));
  }
  // else use the runtime descriptor version of scan/verify

  // Handle optional argument, back
  auto makeRefThenEmbox = [&](mlir::Value b) {
    fir::LogicalType logTy = fir::LogicalType::get(
        builder.getContext(), builder.getKindMap().defaultLogicalKind());
    mlir::Value temp = builder.createTemporary(loc, logTy);
    mlir::Value castb = builder.createConvert(loc, logTy, b);
    builder.create<fir::StoreOp>(loc, castb, temp);
    return builder.createBox(loc, temp);
  };
  mlir::Value back = fir::isUnboxedValue(args[2])
                         ? makeRefThenEmbox(*args[2].getUnboxed())
                         : builder.create<fir::AbsentOp>(
                               loc, fir::BoxType::get(builder.getI1Type()));

  // Handle required string argument
  mlir::Value string = builder.createBox(loc, args[0]);

  // Handle required set argument
  mlir::Value set = builder.createBox(loc, args[1]);

  // Handle kind argument
  mlir::Value kind = fir::getBase(args[3]);

  // Create result descriptor
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultType);
  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  fir::runtime::genVerifyDescriptor(builder, loc, resultIrBox, string, set,
                                    back, kind);

  // Handle cleanup of allocatable result descriptor and return
  return readAndAddCleanUp(resultMutableBox, resultType, "VERIFY");
}

/// Process calls to Minloc, Maxloc intrinsic functions
template <typename FN, typename FD>
fir::ExtendedValue
IntrinsicLibrary::genExtremumloc(FN func, FD funcDim, llvm::StringRef errMsg,
                                 mlir::Type resultType,
                                 llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 5);

  // Handle required array argument
  mlir::Value array = builder.createBox(loc, args[0]);
  unsigned rank = fir::BoxValue(array).rank();
  assert(rank >= 1);

  // Handle optional mask argument
  auto mask = isStaticallyAbsent(args[2])
                  ? builder.create<fir::AbsentOp>(
                        loc, fir::BoxType::get(builder.getI1Type()))
                  : builder.createBox(loc, args[2]);

  // Handle optional kind argument
  auto kind = isStaticallyAbsent(args[3])
                  ? builder.createIntegerConstant(
                        loc, builder.getIndexType(),
                        builder.getKindMap().defaultIntegerKind())
                  : fir::getBase(args[3]);

  // Handle optional back argument
  auto back = isStaticallyAbsent(args[4]) ? builder.createBool(loc, false)
                                          : fir::getBase(args[4]);

  bool absentDim = isStaticallyAbsent(args[1]);

  if (!absentDim && rank == 1) {
    // If dim argument is present and the array is rank 1, then the result is
    // a scalar (since the the result is rank-1 or 0).
    // Therefore, we use a scalar result descriptor with Min/MaxlocDim().
    mlir::Value dim = fir::getBase(args[1]);
    // Create mutable fir.box to be passed to the runtime for the result.
    fir::MutableBoxValue resultMutableBox =
        fir::factory::createTempMutableBox(builder, loc, resultType);
    mlir::Value resultIrBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

    funcDim(builder, loc, resultIrBox, array, dim, mask, kind, back);

    // Handle cleanup of allocatable result descriptor and return
    return readAndAddCleanUp(resultMutableBox, resultType, errMsg);
  }

  // Note: The Min/Maxloc/val cases below have an array result.

  // Create mutable fir.box to be passed to the runtime for the result.
  mlir::Type resultArrayType =
      builder.getVarLenSeqTy(resultType, absentDim ? 1 : rank - 1);
  fir::MutableBoxValue resultMutableBox =
      fir::factory::createTempMutableBox(builder, loc, resultArrayType);
  mlir::Value resultIrBox =
      fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

  if (absentDim) {
    // Handle min/maxloc/val case where there is no dim argument
    // (calls Min/Maxloc()/MinMaxval() runtime routine)
    func(builder, loc, resultIrBox, array, mask, kind, back);
  } else {
    // else handle min/maxloc case with dim argument (calls
    // Min/Max/loc/val/Dim() runtime routine).
    mlir::Value dim = fir::getBase(args[1]);
    funcDim(builder, loc, resultIrBox, array, dim, mask, kind, back);
  }
  return readAndAddCleanUp(resultMutableBox, resultType, errMsg);
}

// MAXLOC
fir::ExtendedValue
IntrinsicLibrary::genMaxloc(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  return genExtremumloc(fir::runtime::genMaxloc, fir::runtime::genMaxlocDim,
                        "MAXLOC", resultType, args);
}

/// Process calls to Maxval and Minval
template <typename FN, typename FD, typename FC>
fir::ExtendedValue
IntrinsicLibrary::genExtremumVal(FN func, FD funcDim, FC funcChar,
                                 llvm::StringRef errMsg, mlir::Type resultType,
                                 llvm::ArrayRef<fir::ExtendedValue> args) {

  assert(args.size() == 3);

  // Handle required array argument
  fir::BoxValue arryTmp = builder.createBox(loc, args[0]);
  mlir::Value array = fir::getBase(arryTmp);
  int rank = arryTmp.rank();
  assert(rank >= 1);
  bool hasCharacterResult = arryTmp.isCharacter();

  // Handle optional mask argument
  auto mask = isStaticallyAbsent(args[2])
                  ? builder.create<fir::AbsentOp>(
                        loc, fir::BoxType::get(builder.getI1Type()))
                  : builder.createBox(loc, args[2]);

  bool absentDim = isStaticallyAbsent(args[1]);

  // For Maxval/MinVal, we call the type specific versions of
  // Maxval/Minval because the result is scalar in the case below.
  if (!hasCharacterResult && (absentDim || rank == 1))
    return func(builder, loc, array, mask);

  if (hasCharacterResult && (absentDim || rank == 1)) {
    // Create mutable fir.box to be passed to the runtime for the result.
    fir::MutableBoxValue resultMutableBox =
        fir::factory::createTempMutableBox(builder, loc, resultType);
    mlir::Value resultIrBox =
        fir::factory::getMutableIRBox(builder, loc, resultMutableBox);

    funcChar(builder, loc, resultIrBox, array, mask);

    // Handle cleanup of allocatable result descriptor and return
    return readAndAddCleanUp(resultMutableBox, resultType, errMsg);
  }

  // Handle Min/Maxval cases that have an array result.
  auto resultMutableBox =
      genFuncDim(funcDim, resultType, builder, loc, array, args[1], mask, rank);
  return readAndAddCleanUp(resultMutableBox, resultType, errMsg);
}

// MAXVAL
fir::ExtendedValue
IntrinsicLibrary::genMaxval(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  return genExtremumVal(fir::runtime::genMaxval, fir::runtime::genMaxvalDim,
                        fir::runtime::genMaxvalChar, "MAXVAL", resultType,
                        args);
}

// MINLOC
fir::ExtendedValue
IntrinsicLibrary::genMinloc(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  return genExtremumloc(fir::runtime::genMinloc, fir::runtime::genMinlocDim,
                        "MINLOC", resultType, args);
}

// MINVAL
fir::ExtendedValue
IntrinsicLibrary::genMinval(mlir::Type resultType,
                            llvm::ArrayRef<fir::ExtendedValue> args) {
  return genExtremumVal(fir::runtime::genMinval, fir::runtime::genMinvalDim,
                        fir::runtime::genMinvalChar, "MINVAL", resultType,
                        args);
}

// MIN and MAX
template <Extremum extremum, ExtremumBehavior behavior>
mlir::Value IntrinsicLibrary::genExtremum(mlir::Type,
                                          llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() >= 1);
  mlir::Value result = args[0];
  for (auto arg : args.drop_front()) {
    mlir::Value mask =
        createExtremumCompare<extremum, behavior>(loc, builder, result, arg);
    result = builder.create<mlir::arith::SelectOp>(loc, mask, result, arg);
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Argument lowering rules interface for intrinsic or intrinsic module
// procedure.
//===----------------------------------------------------------------------===//

const fir::IntrinsicArgumentLoweringRules *
fir::getIntrinsicArgumentLowering(llvm::StringRef specificName) {
  llvm::StringRef name = genericName(specificName);
  if (const IntrinsicHandler *handler = findIntrinsicHandler(name))
    if (!handler->argLoweringRules.hasDefaultRules())
      return &handler->argLoweringRules;
  return nullptr;
}

/// Return how argument \p argName should be lowered given the rules for the
/// intrinsic function.
fir::ArgLoweringRule
fir::lowerIntrinsicArgumentAs(const IntrinsicArgumentLoweringRules &rules,
                              unsigned position) {
  assert(position < sizeof(rules.args) / (sizeof(decltype(*rules.args))) &&
         "invalid argument");
  return {rules.args[position].lowerAs,
          rules.args[position].handleDynamicOptional};
}

//===----------------------------------------------------------------------===//
// Public intrinsic call helpers
//===----------------------------------------------------------------------===//

std::pair<fir::ExtendedValue, bool>
fir::genIntrinsicCall(fir::FirOpBuilder &builder, mlir::Location loc,
                      llvm::StringRef name,
                      std::optional<mlir::Type> resultType,
                      llvm::ArrayRef<fir::ExtendedValue> args) {
  return IntrinsicLibrary{builder, loc}.genIntrinsicCall(name, resultType,
                                                         args);
}

mlir::Value fir::genMax(fir::FirOpBuilder &builder, mlir::Location loc,
                        llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() > 0 && "max requires at least one argument");
  return IntrinsicLibrary{builder, loc}
      .genExtremum<Extremum::Max, ExtremumBehavior::MinMaxss>(args[0].getType(),
                                                              args);
}

mlir::Value fir::genMin(fir::FirOpBuilder &builder, mlir::Location loc,
                        llvm::ArrayRef<mlir::Value> args) {
  assert(args.size() > 0 && "min requires at least one argument");
  return IntrinsicLibrary{builder, loc}
      .genExtremum<Extremum::Min, ExtremumBehavior::MinMaxss>(args[0].getType(),
                                                              args);
}

mlir::Value fir::genPow(fir::FirOpBuilder &builder, mlir::Location loc,
                        mlir::Type type, mlir::Value x, mlir::Value y) {
  // TODO: since there is no libm version of pow with integer exponent,
  //       we have to provide an alternative implementation for
  //       "precise/strict" FP mode.
  //       One option is to generate internal function with inlined
  //       implementation and mark it 'strictfp'.
  //       Another option is to implement it in Fortran runtime library
  //       (just like matmul).
  return IntrinsicLibrary{builder, loc}.genRuntimeCall("pow", type, {x, y});
}

mlir::SymbolRefAttr fir::getUnrestrictedIntrinsicSymbolRefAttr(
    fir::FirOpBuilder &builder, mlir::Location loc, llvm::StringRef name,
    mlir::FunctionType signature) {
  return IntrinsicLibrary{builder, loc}.getUnrestrictedIntrinsicSymbolRefAttr(
      name, signature);
}
