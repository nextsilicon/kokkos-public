//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#define KOKKOS_IMPL_PUBLIC_INCLUDE

#include <nsapi/intrinsics.h>
#include <nsapi/memory.h>

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSiliconSpace.hpp>
#include <NextSilicon/Kokkos_NextSilicon_DeepCopy.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Instance.hpp>

#ifdef KOKKOS_ENABLE_NEXTSILICON

namespace Kokkos {
namespace Impl {
void DeepCopySharedNextSilicon(void* dst, const void* src, size_t n) {
  // Let NextSilicon runtime select the correct implementation.
  nsapi_memory_copy(dst, src, n);
}

void DeepCopyDeviceNextSilicon(void* dst, const void* src, size_t n) {
#ifdef KOKKOS_ENABLE_IMPL_NSAPI_UNAVAIL
  std::memcpy(dst, src, n);
#else
  nsapi_memory_copy(dst, src, n);
#endif
}

}  // namespace Impl
}  // namespace Kokkos

#endif
