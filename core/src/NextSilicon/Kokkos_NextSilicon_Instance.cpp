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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <Kokkos_Core.hpp>

#include "NextSilicon/Kokkos_NextSilicon_Instance.hpp"
#include "NextSilicon/Kokkos_NextSilicon.hpp"
#include "NextSilicon/Kokkos_NextSiliconSpace.hpp"
#include "impl/Kokkos_Profiling.hpp"
#include <ostream>
#include <cstdint>

#if !defined(KOKKOS_ENABLE_IMPL_NSAPI_UNAVAIL) && \
    defined(KOKKOS_ENABLE_IMPL_NSAPI_ADD_TELEM_REGIONS)
#include <nsapi/telem.h>
#endif

namespace Kokkos::Experimental::Impl {

NextSiliconInternal* NextSiliconInternal::singleton() {
  static NextSiliconInternal self;
  return &self;
}

void NextSiliconInternal::initialize() {
#if !defined(KOKKOS_ENABLE_IMPL_NSAPI_UNAVAIL) && \
    defined(KOKKOS_ENABLE_IMPL_NSAPI_ADD_TELEM_REGIONS)
  /* Wrap the entire program with a nsapi telem region.
   * Required for performance estimation to take into account all kernel
   * invocations.
   */
  nsapi_telem_region_enter();
#endif
  m_is_initialized = true;
}

void NextSiliconInternal::finalize() {
#if !defined(KOKKOS_ENABLE_IMPL_NSAPI_UNAVAIL) && \
    defined(KOKKOS_ENABLE_IMPL_NSAPI_ADD_TELEM_REGIONS)
  nsapi_telem_region_exit();
#endif
  nextsilicon_global_unique_token_locks(true);
  m_is_initialized = false;
}

bool NextSiliconInternal::is_initialized() const {
  return m_is_initialized;
}

void NextSiliconInternal::print_configuration(
    std::ostream &os) const {
#if defined(KOKKOS_ENABLE_NEXTSILICON)
  os << "macro  KOKKOS_ENABLE_NEXTSILICON      : defined\n";
#endif
  // FIXME_NEXTSILICON print_configuration doesn't do anything useful, fix
  // once device properties nsapi is available
}

void NextSiliconInternal::fence(
    std::string const &name) const {
  // FIXME_NEXTSILICON fence doesn't do anything in the OpenMP-style interface

  Kokkos::Tools::Experimental::Impl::profile_fence_event<NextSilicon>(
      name,
      Kokkos::Tools::Experimental::Impl::DirectFenceIDHandle{instance_id()},
      [&]() {});
}

uint32_t NextSiliconInternal::instance_id()
    const noexcept {
  return Kokkos::Tools::Experimental::Impl::idForInstance<NextSilicon>(reinterpret_cast<uintptr_t>(this));
}

Kokkos::View<uint32_t *, Kokkos::Experimental::NextSiliconSpace> nextsilicon_global_unique_token_locks(bool deallocate) {
  static Kokkos::View<uint32_t*, NextSiliconSpace> locks = Kokkos::View<uint32_t *, NextSiliconSpace>();
  if (!deallocate && locks.extent(0) == 0) {
    locks = Kokkos::View<uint32_t *, NextSiliconSpace>("Kokkos::UniqueToken<NextSilicon>::m_locks",NextSilicon().concurrency());
  }
  if (deallocate) {
    locks = Kokkos::View<uint32_t *, NextSiliconSpace>();
  }
  return locks;
}

} // namespace Kokkos::Experimental::Impl
