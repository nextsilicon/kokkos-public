//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2023) National Technology & Engineering
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

#ifndef KOKKOS_NEXTSILICONSPACE_ZEROMEMSET_HPP
#define KOKKOS_NEXTSILICONSPACE_ZEROMEMSET_HPP

#include <nsapi/memory.h>

#include <Kokkos_Macros.hpp>
#include "Kokkos_NextSiliconSpace.hpp"
#include <impl/Kokkos_ZeroMemset_fwd.hpp>

namespace Kokkos {
namespace Impl {

void ZeroMemsetNextSilicon(void* buffer, size_t buffer_size);

template <>
struct ZeroMemset<Kokkos::Experimental::NextSilicon> {
  ZeroMemset(const Kokkos::Experimental::NextSilicon& exec_space, void* dst, size_t cnt) {
    ZeroMemsetNextSilicon(dst, cnt);
  }
};

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_NEXTSILICONSPACE_ZEROMEMSET_HPP
