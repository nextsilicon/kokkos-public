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
  std::memcpy(dst, src, n);
}

void DeepCopyDeviceNextSilicon(void* dst, const void* src, size_t n) {
#ifdef KOKKOS_IMPL_NSAPI_UNAVAIL
  // FIXME_NEXTSILICON fall back to regular memcpy if nsapi not available
  std::memcpy(dst, src, n);
#else
  if (n > Kokkos::Experimental::Impl::NextSiliconTraits::BmtUseThreshold) {
    llns_memory_device_copy_bmt(dst, src, n);
  } else if (__nsapi_is_on_cg()) {
    llns_memory_device_copy_rma(dst, src, n);
  } else {
    std::memcpy(dst, src, n);
  }
#endif
}

}  // namespace Impl
}  // namespace Kokkos

#endif
