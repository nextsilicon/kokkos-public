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
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif

#ifndef KOKKOS_NEXTSILICON_SPACE_HPP
#define KOKKOS_NEXTSILICON_SPACE_HPP

#include <Kokkos_Concepts.hpp>
#include <impl/Kokkos_Tools.hpp>

#include <iosfwd>

namespace Kokkos::Experimental {

namespace Impl {

template <typename T>
struct is_nextsilicon_type_space : public std::false_type {};

}  // namespace Impl

class NextSilicon;

class NextSiliconSpace {
 public:
  using memory_space    = NextSiliconSpace;
  using execution_space = NextSilicon;
  using device_type     = Kokkos::Device<execution_space, memory_space>;

  using size_type = size_t;

  NextSiliconSpace() = default;

  /**\brief  Allocate untracked memory in the space */
  template <typename ExecutionSpace>
  void* allocate(const ExecutionSpace& exec_space,
                 const size_t arg_alloc_size) const {
    return allocate(exec_space, "[unlabeled]", arg_alloc_size);
  }
  template <typename ExecutionSpace>
  void* allocate(const ExecutionSpace& exec_space, const char* arg_label,
                 const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const {
    return impl_allocate(exec_space, arg_label, arg_alloc_size,
                         arg_logical_size);
  }
  void* allocate(const size_t arg_alloc_size) const;
  void* allocate(const char* arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const;

  /**\brief  Deallocate untracked memory in the space */
  void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const;
  void deallocate(const char* arg_label, void* const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) const;

  static constexpr char const* name() { return "NextSiliconSpace"; }

 private:
  template <typename ExecutionSpace>
  void* impl_allocate(const ExecutionSpace& exec_space, const char* arg_label,
                      const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const Kokkos::Tools::SpaceHandle arg_handle =
                          Kokkos::Tools::make_space_handle(name())) const {
    if (!std::is_same_v<ExecutionSpace, Kokkos::Experimental::NextSilicon>) {
      exec_space.fence();
    }
    return impl_allocate(arg_label, arg_alloc_size, arg_logical_size,
                         arg_handle);
  }
  void* impl_allocate(const char* arg_label, const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const Kokkos::Tools::SpaceHandle =
                          Kokkos::Tools::make_space_handle(name())) const;
  void impl_deallocate(const char* arg_label, void* const arg_alloc_ptr,
                       const size_t arg_alloc_size,
                       const size_t arg_logical_size = 0,
                       const Kokkos::Tools::SpaceHandle =
                           Kokkos::Tools::make_space_handle(name())) const;
};

template <>
struct Impl::is_nextsilicon_type_space<NextSiliconSpace>
    : public std::true_type {};

}  // namespace Kokkos::Experimental

namespace Kokkos::Experimental {

class NextSiliconManagedSpace {
 public:
  using memory_space    = NextSiliconManagedSpace;
  using execution_space = NextSilicon;
  using device_type     = Kokkos::Device<execution_space, memory_space>;

  using size_type = size_t;

  NextSiliconManagedSpace() = default;

  /**\brief  Allocate untracked memory in the space */
  template <typename ExecutionSpace>
  void* allocate(const ExecutionSpace& exec_space,
                 const size_t arg_alloc_size) const {
    return allocate(exec_space, "[unlabeled]", arg_alloc_size);
  }
  template <typename ExecutionSpace>
  void* allocate(const ExecutionSpace& exec_space, const char* arg_label,
                 const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const {
    return impl_allocate(exec_space, arg_label, arg_alloc_size,
                         arg_logical_size);
  }
  void* allocate(const size_t arg_alloc_size) const;
  void* allocate(const char* arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const;

  /**\brief  Deallocate untracked memory in the space */
  void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const;
  void deallocate(const char* arg_label, void* const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) const;

  static constexpr char const* name() { return "NextSiliconManagedSpace"; }

 private:
  template <typename ExecutionSpace>
  void* impl_allocate(const ExecutionSpace& exec_space, const char* arg_label,
                      const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const Kokkos::Tools::SpaceHandle arg_handle =
                          Kokkos::Tools::make_space_handle(name())) const {
    if (!std::is_same_v<ExecutionSpace, Kokkos::Experimental::NextSilicon>) {
      exec_space.fence();
    }
    return impl_allocate(arg_label, arg_alloc_size, arg_logical_size,
                         arg_handle);
  }
  void* impl_allocate(const char* arg_label, const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const Kokkos::Tools::SpaceHandle =
                          Kokkos::Tools::make_space_handle(name())) const;
  void impl_deallocate(const char* arg_label, void* const arg_alloc_ptr,
                       const size_t arg_alloc_size,
                       const size_t arg_logical_size = 0,
                       const Kokkos::Tools::SpaceHandle =
                           Kokkos::Tools::make_space_handle(name())) const;
};

template <>
struct Impl::is_nextsilicon_type_space<NextSiliconManagedSpace>
    : public std::true_type {};

}  // namespace Kokkos::Experimental

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

static_assert(Kokkos::Impl::MemorySpaceAccess<
              Experimental::NextSiliconSpace,
              Experimental::NextSiliconSpace>::assignable);

static_assert(Kokkos::Impl::MemorySpaceAccess<
              Experimental::NextSiliconManagedSpace,
              Experimental::NextSiliconManagedSpace>::assignable);

// Limit access and assignment from HostSpace to NextSiliconSpace
// NextSiliconSpace::execution_space can always access and assign Host memory.
// NextSiliconManagedSpace accessible and assignable from everywhere.

// FIXME_NEXTSILICON Set Device space to be assignable and accessible from
//  host space (which is slow but works without a fault) because of missing
//  constepxr KOKKOS_IF_ON_DEVICE / KOKKOS_IF_ON_HOST
template <>
struct MemorySpaceAccess<Kokkos::HostSpace,
                         Kokkos::Experimental::NextSiliconSpace> {
  enum : bool { assignable = true };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::HostSpace,
                         Kokkos::Experimental::NextSiliconManagedSpace> {
  enum : bool { assignable = true };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::Experimental::NextSiliconSpace,
                         Kokkos::HostSpace> {
  enum : bool { assignable = true };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::Experimental::NextSiliconSpace,
                         Kokkos::Experimental::NextSiliconManagedSpace> {
  enum : bool { assignable = true };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::Experimental::NextSiliconManagedSpace,
                         Kokkos::HostSpace> {
  enum : bool { assignable = true };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::Experimental::NextSiliconManagedSpace,
                         Kokkos::Experimental::NextSiliconSpace> {
  enum : bool { assignable = true };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};

}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/

#endif
