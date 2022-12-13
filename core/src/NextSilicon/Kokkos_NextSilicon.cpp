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

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Instance.hpp>
#include <impl/Kokkos_Profiling.hpp>
#include <impl/Kokkos_ExecSpaceManager.hpp>

#include <ostream>

Kokkos::Experimental::NextSilicon::NextSilicon()
    : m_space_instance(Impl::NextSiliconInternal::singleton()) {}

void Kokkos::Experimental::NextSilicon::impl_initialize(
    InitializationSettings const& /*settings*/) {
  Impl::NextSiliconInternal::singleton()->initialize();
}

void Kokkos::Experimental::NextSilicon::impl_finalize() {
  Impl::NextSiliconInternal::singleton()->finalize();
}

bool Kokkos::Experimental::NextSilicon::impl_is_initialized() {
  return Impl::NextSiliconInternal::singleton()->is_initialized();
}

void Kokkos::Experimental::NextSilicon::print_configuration(
    std::ostream& os, bool verbose) const {
  m_space_instance->print_configuration(os, verbose);
}

void Kokkos::Experimental::NextSilicon::fence(std::string const& name) const {
  Impl::NextSiliconInternal::singleton()->fence(name);
}

void Kokkos::Experimental::NextSilicon::impl_static_fence(
    std::string const& name) {
  Kokkos::Tools::Experimental::Impl::profile_fence_event<
      Kokkos::Experimental::NextSilicon>(
      name,
      Kokkos::Tools::Experimental::SpecialSynchronizationCases::
          GlobalDeviceSynchronization,
      [&]() { /*FIXME_NEXTSILICON*/ });
}

uint32_t Kokkos::Experimental::NextSilicon::impl_instance_id() const noexcept {
  return m_space_instance->instance_id();
}

int Kokkos::Experimental::NextSilicon::ns_device_id() const noexcept {
  return Impl::NextSiliconInternal::m_ns_device_id;
}

namespace Kokkos {
namespace Impl {

// 180 is after OpenACC (170)
int g_nextsilicon_space_factory_initialized =
    initialize_space_factory<Experimental::NextSilicon>("180_NextSilicon");
}  // namespace Impl
}  // Namespace Kokkos