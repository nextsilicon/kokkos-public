// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <NextSilicon/Kokkos_NextSilicon_InParallelRegion.hpp>
#include <Kokkos_Assert.hpp>
#include <Kokkos_Atomic.hpp>
#include <nextapi/memory.h>

namespace Kokkos::Impl {
/*static*/ NextSiliconParallelRegionScopeGuard::HostPinnedData
    NextSiliconParallelRegionScopeGuard::s_data = false;

NextSiliconParallelRegionScopeGuard::HostPinnedData::HostPinnedData(
    const bool is_in_parallel_region)
    : is_in_parallel_region_(is_in_parallel_region) {
  // Upon construction, pin this variable to host memory to prevent it from
  // being migrated to device.
  nextapi_mem_migrate(this, sizeof(*this), NEXTAPI_PAGE_LOC_HOST, true);
}

NextSiliconParallelRegionScopeGuard::NextSiliconParallelRegionScopeGuard()
    : device_lock_(s_data.mtx_) {
  KOKKOS_ASSERT(s_data.is_in_parallel_region_ == false &&
                "Already in a parallel region");
  s_data.is_in_parallel_region_ = true;
}

NextSiliconParallelRegionScopeGuard::~NextSiliconParallelRegionScopeGuard() {
  KOKKOS_ASSERT(s_data.is_in_parallel_region_ == true &&
                "Not in a parallel region");
  s_data.is_in_parallel_region_ = false;
}

}  // namespace Kokkos::Impl
