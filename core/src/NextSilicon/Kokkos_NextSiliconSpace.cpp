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

#define NS_TILES_PER_DEVICE 4              // assume single-die device
#define NS_SUPPORTED_PAGES 12              // NS supported 12 page sizes
#define NS_SMALLEST_PAGE_SIZE (1UL << 12)  // Smallest page size is 2^12 = 4KiB

namespace Kokkos {
namespace Experimental {

int nsapi_migrate_memory_to_specific_tile(
    void *start, size_t size, int tile,
    enum nsapi_page_location loc = NSAPI_PAGE_LOC_DEVICE) {
  int err = 0;
  assert(start != nullptr && size != 0 && tile >= 0 &&
         tile < NS_TILES_PER_DEVICE);

  struct device_location_request device_location = {
      .location     = {.row = 0, .col = 0, .dcode = 1},
      .rpt_location = RPT_DIRECTIVE,
      .lp           = RLP_CLUSTER,
  };
  device_location.location.quad = tile;
  /*printf(
      "Migrating one contiguous memory chunk: Quad: %i Row: %i Col: %i Size = "
      "%zu, start_addr: %p\n",
      device_location.location.quad, device_location.location.row,
      device_location.location.col, size, start);*/
  err = nsapi_mem_migrate_ex(start, size, loc, false, device_location);
  if (err) {
    printf("NextSilicon: Error migrating memory: %i\n", err);
  }
  return err;
}

static inline size_t round_down_to_multiple(size_t size, size_t multiple) {
  return (size / multiple) * multiple;
}

template <size_t num_page_sizes>
constexpr std::array<size_t, num_page_sizes> generate_page_sizes() {
  std::array<size_t, num_page_sizes> sizes{};
  for (size_t i = 0; i < num_page_sizes; ++i) {
    sizes[i] = 1UL << (num_page_sizes + 2 * i);
  }
  return sizes;
}

uint64_t nsapi_get_migration_chunk_size(size_t size) {
  //  Generate a list of supported page sizes in descending order
  constexpr std::array<size_t, NS_SUPPORTED_PAGES> supported_page_sizes =
      generate_page_sizes<NS_SUPPORTED_PAGES>();

  assert(size >= 0);
  if (size == 0) return 0;

  size_t target_chunk_size = size / NS_TILES_PER_DEVICE;
  size_t best_chunk_size   = supported_page_sizes[0];

  // Find the largest chunk size that is a multiple of one of the supported page
  // sizes
  for (int i = 0; i < NS_SUPPORTED_PAGES; ++i) {
    size_t page_size = supported_page_sizes[i];
    size_t candidate_chunk_size =
        round_down_to_multiple(target_chunk_size, page_size);

    if (candidate_chunk_size > best_chunk_size &&
        candidate_chunk_size <= size) {
      best_chunk_size = candidate_chunk_size;
    }
  }

  return best_chunk_size;
}

void nsapi_migrate_distributed(void *ptr, size_t allocated_size) {
  assert(ptr != nullptr);
  if (allocated_size == 0) return;

  if (allocated_size < NS_SMALLEST_PAGE_SIZE * NS_TILES_PER_DEVICE) {
    nsapi_migrate_memory_to_specific_tile(ptr, allocated_size, 0);
    return;
  }

  // Distribute the memory across all quads.
  size_t chunk_size = nsapi_get_migration_chunk_size(allocated_size);
  int64_t remainder = allocated_size - (chunk_size * NS_TILES_PER_DEVICE);

  assert(remainder >= 0);  // Ensure we didnt mess up the size calculation

  for (unsigned tile = 0; tile < NS_TILES_PER_DEVICE; ++tile) {
    size_t size_to_migrate = chunk_size;
    if (tile == NS_TILES_PER_DEVICE - 1) {
      size_to_migrate += remainder;  // Handle the tail
    }
    nsapi_migrate_memory_to_specific_tile(
        static_cast<char *>(ptr) + tile * chunk_size, size_to_migrate, tile);
  }
}

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
#elif defined(KOKKOS_ENABLE_IMPL_NEXTSILICON_DISTRIBUTE_MEMORY)
  ptr = std::aligned_alloc(arg_alloc_size, arg_alloc_size);
  nsapi_migrate_distributed(ptr, arg_alloc_size);
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
#elif defined(KOKKOS_ENABLE_IMPL_NEXTSILICON_DISTRIBUTE_MEMORY)
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
