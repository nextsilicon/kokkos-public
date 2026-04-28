// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_NEXTSILICON_PAGE_ALIGNED_DATA_HPP
#define KOKKOS_NEXTSILICON_PAGE_ALIGNED_DATA_HPP

#include <utility>
#include <nextapi/memory.h>

namespace Kokkos::Impl {

static constexpr int PAGE_SIZE = 4096;

// This struct is aligned to 4096 bytes (page size) to work around
// issues with NextSilicon page migration. It can be pinned to the host
// to prevent migration to device memory, which would
// cause problems when we try to access it from the host.
template <typename T, bool PinToHost = false>
struct alignas(PAGE_SIZE) PageAlignedData {
  T data;

  operator T&() { return data; }
  operator const T&() const { return data; }

  template <typename... Args>
  PageAlignedData(Args&&... args) : data{std::forward<Args>(args)...} {
    if constexpr (PinToHost) {
      // Pin this variable to host memory to prevent migration to device.
      nextapi_mem_migrate(this, sizeof(*this), NEXTAPI_PAGE_LOC_HOST, true);
    }
  }

  PageAlignedData& operator=(const T& data_) {
    data = data_;
    return *this;
  }

  PageAlignedData(const PageAlignedData&)            = delete;
  PageAlignedData& operator=(const PageAlignedData&) = delete;
  PageAlignedData(PageAlignedData&&)                 = delete;
  PageAlignedData& operator=(PageAlignedData&&)      = delete;
};

static_assert(sizeof(PageAlignedData<int>) == PAGE_SIZE);
static_assert(sizeof(PageAlignedData<bool>) == PAGE_SIZE);

}  // namespace Kokkos::Impl

#endif
