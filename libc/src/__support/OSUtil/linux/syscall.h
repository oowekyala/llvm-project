//===----------------------- Linux syscalls ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_OSUTIL_LINUX_SYSCALL_H
#define LLVM_LIBC_SRC_SUPPORT_OSUTIL_LINUX_SYSCALL_H

#include "src/__support/common.h"
#include "src/__support/macros/architectures.h"

#ifdef LIBC_TARGET_ARCH_IS_X86_64
#include "x86_64/syscall.h"
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
#include "aarch64/syscall.h"
#elif defined(LIBC_TARGET_ARCH_IS_ARM)
#include "arm/syscall.h"
#endif

namespace __llvm_libc {

template <typename... Ts>
LIBC_INLINE long syscall_impl(long __number, Ts... ts) {
  static_assert(sizeof...(Ts) <= 6, "Too many arguments for syscall");
  return syscall_impl(__number, (long)ts...);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_OSUTIL_LINUX_SYSCALL_H
