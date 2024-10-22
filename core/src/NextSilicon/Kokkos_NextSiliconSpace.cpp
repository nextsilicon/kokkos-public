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

#include <nsapi/memory.h>

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSiliconSpace.hpp>
#include <NextSilicon/Kokkos_NextSilicon_DeepCopy.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>

namespace Kokkos {
namespace Experimental {

void *NextSiliconSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}

void *NextSiliconSpace::allocate(const char *arg_label,
                                 const size_t arg_alloc_size,
                                 const size_t arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}

void *NextSiliconSpace::impl_allocate(
    const char *arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");

  void *ptr = nullptr;
#ifdef KOKKOS_ENABLE_IMPL_NSAPI_UNAVAIL
  // FIXME_NEXTSILICON default to managed allocation instead of
  // host pinned.
  ptr = malloc(arg_alloc_size);
#else
  ptr = llns_memory_device_allocate(
      arg_alloc_size, /* loopref */ {}, /* cache_capacity */ {},
      /* hit_throughput */ {}, /* miss_throughput */ {});
#endif

  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }

  return ptr;
}

void NextSiliconSpace::deallocate(void *const arg_alloc_ptr,
                                  const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}

void NextSiliconSpace::deallocate(const char *arg_label,
                                  void *const arg_alloc_ptr,
                                  const size_t arg_alloc_size,
                                  const size_t arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}

void NextSiliconSpace::impl_deallocate(
    const char *arg_label, void *const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                      reported_size);
  }

  if (arg_alloc_ptr) {
#ifdef KOKKOS_ENABLE_IMPL_NSAPI_UNAVAIL
    // FIXME_NEXTSILICON Remove once new release is available
    free(arg_alloc_ptr);
#else
    llns_memory_free(arg_alloc_ptr);
#endif
  }
}

void *NextSiliconSharedSpace::allocate(const size_t arg_alloc_size) const {
  return allocate("[unlabeled]", arg_alloc_size);
}

void *NextSiliconSharedSpace::allocate(const char *arg_label,
                                       const size_t arg_alloc_size,
                                       const size_t arg_logical_size) const {
  return impl_allocate(arg_label, arg_alloc_size, arg_logical_size);
}

void *NextSiliconSharedSpace::impl_allocate(
    const char *arg_label, const size_t arg_alloc_size,
    const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");

  void *ptr = nullptr;
  // NextSilicon implements shared UVM over standard memory operations.
  ptr = malloc(arg_alloc_size);

  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::allocateData(arg_handle, arg_label, ptr, reported_size);
  }

  return ptr;
}

void NextSiliconSharedSpace::deallocate(void *const arg_alloc_ptr,
                                        const size_t arg_alloc_size) const {
  deallocate("[unlabeled]", arg_alloc_ptr, arg_alloc_size);
}

void NextSiliconSharedSpace::deallocate(const char *arg_label,
                                        void *const arg_alloc_ptr,
                                        const size_t arg_alloc_size,
                                        const size_t arg_logical_size) const {
  impl_deallocate(arg_label, arg_alloc_ptr, arg_alloc_size, arg_logical_size);
}

void NextSiliconSharedSpace::impl_deallocate(
    const char *arg_label, void *const arg_alloc_ptr,
    const size_t arg_alloc_size, const size_t arg_logical_size,
    const Kokkos::Tools::SpaceHandle arg_handle) const {
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    Kokkos::Profiling::deallocateData(arg_handle, arg_label, arg_alloc_ptr,
                                      reported_size);
  }

  if (arg_alloc_ptr) {
    // NextSilicon implements shared UVM over standard memory operations.
    free(arg_alloc_ptr);
  }
}

}  // namespace Experimental
}  // namespace Kokkos
