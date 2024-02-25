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

#include <NextSilicon/Kokkos_NextSilicon_Instance.hpp>
#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <impl/Kokkos_Profiling.hpp>

#include <ostream>

Kokkos::Experimental::Impl::NextSiliconInternal*
Kokkos::Experimental::Impl::NextSiliconInternal::singleton() {
  static NextSiliconInternal self;
  return &self;
}

void Kokkos::Experimental::Impl::NextSiliconInternal::initialize() {
  m_is_initialized = true;
}

void Kokkos::Experimental::Impl::NextSiliconInternal::finalize() {
  m_is_initialized = false;
}

bool Kokkos::Experimental::Impl::NextSiliconInternal::is_initialized() const {
  return m_is_initialized;
}

void Kokkos::Experimental::Impl::NextSiliconInternal::print_configuration(
    std::ostream& os) const {
#if defined(KOKKOS_ENABLE_NEXTSILICON)
  os << "macro  KOKKOS_ENABLE_NEXTSILICON      : defined\n";
#endif
  // FIXME_NEXTSILICON print_configuration doesn't do anything useful, fix
  // once device properties nsapi is available
}

void Kokkos::Experimental::Impl::NextSiliconInternal::fence(
    std::string const& name) const {
  // FIXME_NEXTSILICON fence doesn't do anything in the OpenMP-style interface

  Kokkos::Tools::Experimental::Impl::profile_fence_event<
      Kokkos::Experimental::NextSilicon>(
      name,
      Kokkos::Tools::Experimental::Impl::DirectFenceIDHandle{instance_id()},
      [&]() {});
}

uint32_t Kokkos::Experimental::Impl::NextSiliconInternal::instance_id() const
    noexcept {
  return Kokkos::Tools::Experimental::Impl::idForInstance<
      Kokkos::Experimental::NextSilicon>(reinterpret_cast<uintptr_t>(this));
}