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

#ifndef KOKKOS_NEXTSILICON_HPP
#define KOKKOS_NEXTSILICON_HPP

#include <nsapi/intrinsics.h>

#include <NextSilicon/Kokkos_NextSiliconSpace.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <impl/Kokkos_HostSharedPtr.hpp>

namespace Kokkos::Experimental::Impl {
class NextSiliconInternal;
/// An internal-use-only intrinsic used to communicate from Kokkos C++ code to
/// the MLIR compiler stack that a struct at the address `parameter_struct` can
/// be considered immutable and thread invariant for the full duration of the
/// microtask, and that consequently all its members are read only, and that the
/// pointer to the struct and any derived pointer does not alias with anything
/// else (pointers stored in the struct can still alias). This should _only_ be
/// used to annotate the functor objects passed into Kokkos parallel constructs,
/// and should not be used _anywhere_ outside of Kokkos.
extern "C" void __ns_immutable_thread_invariant_parameter_struct(
    const void* parameter_struct);
}  // namespace Kokkos::Experimental::Impl

namespace Kokkos::Experimental {

class NextSilicon {
  Impl::NextSiliconInternal* m_space_instance;

  friend bool operator==(NextSilicon const& lhs, NextSilicon const& rhs) {
    return lhs.impl_internal_space_instance() ==
           rhs.impl_internal_space_instance();
  }
  friend bool operator!=(NextSilicon const& lhs, NextSilicon const& rhs) {
    return !(lhs == rhs);
  }

 public:
  using execution_space = NextSilicon;
#if defined(KOKKOS_ENABLE_IMPL_NEXTSILICON_UNIFIED_MEMORY)
  using memory_space = Kokkos::Experimental::NextSiliconSharedSpace;
#else
  using memory_space = Kokkos::Experimental::NextSiliconSpace;
#endif
  using device_type = Kokkos::Device<execution_space, memory_space>;

  using array_layout = LayoutLeft;
  using size_type    = memory_space::size_type;

  using scratch_memory_space = ScratchMemorySpace<NextSilicon>;

  NextSilicon();

  static void impl_initialize(InitializationSettings const& settings);
  static void impl_finalize();
  static bool impl_is_initialized();

  void print_configuration(std::ostream& os, bool verbose = false) const;

  void fence(std::string const& name =
                 "Kokkos::NextSilicon::fence(): Unnamed Instance Fence") const;
  static void impl_static_fence(std::string const& name);

  static char const* name() { return "NextSilicon"; }
  static int concurrency() { return 64 * 1024; /* FIXME_NEXTSILICON - move to nsapi call */ }
  static bool in_parallel() {
    // true if we're inside a handed-off function, false otherwise
    // FIXME_NEXTSILICON in_parallel unimplemented
    return __nsapi_is_on_cg();
  }
  uint32_t impl_instance_id() const noexcept;
  Impl::NextSiliconInternal* impl_internal_space_instance() const {
    return m_space_instance;
  }

  int ns_device_id() const noexcept;
};

}  // namespace Kokkos::Experimental

template <>
struct Kokkos::Tools::Experimental::DeviceTypeTraits<
    ::Kokkos::Experimental::NextSilicon> {
  static constexpr DeviceType id =
      ::Kokkos::Profiling::Experimental::DeviceType::NextSilicon;

  static int device_id(const Kokkos::Experimental::NextSilicon& exec) {
    return exec.ns_device_id();
  }
};

#endif  // KOKKOS_NEXTSILICON_HPP
