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
#include <NextSilicon/Kokkos_NextSilicon_ZeroMemset.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Instance.hpp>

#ifdef KOKKOS_ENABLE_NEXTSILICON

namespace Kokkos {
namespace Impl {
void ZeroMemsetNextSilicon(void* buffer, size_t buffer_size) {
  auto fill_alignment =
      Kokkos::Experimental::Impl::NextSiliconTraits::FillAlignmend;
  if (buffer_size % fill_alignment != 0 ||
      reinterpret_cast<uintptr_t>(buffer) % fill_alignment != 0) {
    // Can't use hardware acceleration as pointers are not aligned correctly.
    // Valid for all memory space for NextSilicon.
    std::memset(buffer, 0, buffer_size);
  } else if (buffer_size >=
             Kokkos::Experimental::Impl::NextSiliconTraits::BmtUseThreshold) {
    llns_memory_device_fill_bmt(buffer, 0, buffer_size);
  } else {
    uint64_t pattern = 0;
    llns_memory_device_fill_rma(buffer, pattern, sizeof(pattern), buffer_size);
  }
}

}  // namespace Impl
}  // namespace Kokkos

#endif
