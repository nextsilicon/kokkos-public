// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_IN_PARALLEL_REGION_HPP
#define KOKKOS_NEXTSILICON_IN_PARALLEL_REGION_HPP

#include <atomic>
#include <cstddef>
#include <mutex>

// used to implement KOKKOS_IF_ON_HOST / KOKKOS_IF_ON_DEVICE
namespace Kokkos::Impl {

// NextSiliconParallelRegionScopeGuard maintains a flag that
// tracks whether the current thread is executing under the NextSilicon
// device execution space. This is needed to correctly implement
// KOKKOS_IF_ON_DEVICE/KOKKOS_IF_ON_HOST semantics: unlike CUDA or SYCL,
// NextSilicon kernels may run on the host during training/telemetry
// collection while still being under the device execution space, so we
// cannot simply equate "on device" with "offloaded to device".
//
// It also uses a mutex to ensure that the parallel execution is thread-safe,
// i.e. that only one host thread can be in a parallel region at a time.
class NextSiliconParallelRegionScopeGuard {
  // Page-aligned for nextapi_mem_migrate.
  // FIXME_NEXTSILICON: Revisit this if the page size becomes configurable.
  static constexpr int PAGE_SIZE = 4096;

  // Lock guard for RAII mutex locking
  const std::lock_guard<std::mutex> device_lock_;

  // FIXME_NEXTSILICON: It requires only space for a bool and a mutex, but it
  // is aligned to PAGE_SIZE and pinned to the host to prevent migration to
  // device memory, which would cause problems when we try to access it from
  // the host. This should be a facility provided by the toolchain.
  struct alignas(PAGE_SIZE) HostPinnedData {
    // Atomic access is preferable as in() method that accesses this value is
    // public, and callable by anybody. It could race with the  con/destructor.
    std::atomic<bool> is_in_parallel_region_;
    std::mutex mtx_;

    HostPinnedData(const bool b);
  };

  static_assert(sizeof(HostPinnedData) == PAGE_SIZE);
  static HostPinnedData s_data;

 public:
  static bool in() { return s_data.is_in_parallel_region_; }

  NextSiliconParallelRegionScopeGuard();
  ~NextSiliconParallelRegionScopeGuard();

  // Copy and move operations are deleted to emulate semantics of
  // std::lock_guard.
  NextSiliconParallelRegionScopeGuard(
      NextSiliconParallelRegionScopeGuard const&) = delete;
  NextSiliconParallelRegionScopeGuard& operator=(
      NextSiliconParallelRegionScopeGuard const&) = delete;
  NextSiliconParallelRegionScopeGuard(NextSiliconParallelRegionScopeGuard&&) =
      delete;
  NextSiliconParallelRegionScopeGuard& operator=(
      NextSiliconParallelRegionScopeGuard&&) = delete;
};

}  // namespace Kokkos::Impl

#endif
