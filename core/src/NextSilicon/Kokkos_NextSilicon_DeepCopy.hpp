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

#ifndef KOKKOS_NEXTSILICON_DEEP_COPY_HPP
#define KOKKOS_NEXTSILICON_DEEP_COPY_HPP

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSiliconSpace.hpp>

#include <Kokkos_Concepts.hpp>

namespace Kokkos {
namespace Impl {

void DeepCopySharedNextSilicon(void* dst, const void* src, size_t n);
void DeepCopyDeviceNextSilicon(void* dst, const void* src, size_t n);

template <>
struct DeepCopy<Kokkos::Experimental::NextSiliconSpace,
                Kokkos::Experimental::NextSiliconSpace,
                Kokkos::Experimental::NextSilicon> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
  DeepCopy(const Kokkos::Experimental::NextSilicon&, void* dst, const void* src,
           size_t n) {
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::Experimental::NextSiliconSpace,
                Kokkos::Experimental::NextSiliconSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "Kokkos::Impl::DeepCopy<NextSiliconSpace, NextSiliconSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
};

template <>
struct DeepCopy<Kokkos::Experimental::NextSiliconSpace, Kokkos::HostSpace,
                Kokkos::Experimental::NextSilicon> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
  DeepCopy(const Kokkos::Experimental::NextSilicon&, void* dst, const void* src,
           size_t n) {
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::Experimental::NextSiliconSpace, Kokkos::HostSpace,
                ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "Kokkos::Impl::DeepCopy<NextSiliconSpace, HostSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
};

template <>
struct DeepCopy<Kokkos::HostSpace, Kokkos::Experimental::NextSiliconSpace,
                Kokkos::Experimental::NextSilicon> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
  DeepCopy(const Kokkos::Experimental::NextSilicon&, void* dst, const void* src,
           size_t n) {
    DeepCopyDeviceNextSilicon(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::HostSpace, Kokkos::Experimental::NextSiliconSpace,
                ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "Kokkos::Impl::DeepCopy<HostSpace, NextSiliconSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    DeepCopySharedNextSilicon(dst, src, n);
  }
};

template <>
struct DeepCopy<Kokkos::Experimental::NextSiliconManagedSpace,
                Kokkos::Experimental::NextSiliconManagedSpace,
                Kokkos::Experimental::NextSilicon> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
  DeepCopy(const Kokkos::Experimental::NextSilicon&, void* dst, const void* src,
           size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::Experimental::NextSiliconManagedSpace,
                Kokkos::Experimental::NextSiliconManagedSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "Kokkos::Impl::DeepCopy<NextSiliconSpace, NextSiliconSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    DeepCopySharedNextSilicon(dst, src, n);
  }
};

template <>
struct DeepCopy<Kokkos::Experimental::NextSiliconManagedSpace,
                Kokkos::HostSpace, Kokkos::Experimental::NextSilicon> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
  DeepCopy(const Kokkos::Experimental::NextSilicon&, void* dst, const void* src,
           size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::Experimental::NextSiliconManagedSpace,
                Kokkos::HostSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "Kokkos::Impl::DeepCopy<NextSiliconSpace, HostSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    DeepCopySharedNextSilicon(dst, src, n);
  }
};

template <>
struct DeepCopy<Kokkos::HostSpace,
                Kokkos::Experimental::NextSiliconManagedSpace,
                Kokkos::Experimental::NextSilicon> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
  DeepCopy(const Kokkos::Experimental::NextSilicon&, void* dst, const void* src,
           size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::HostSpace,
                Kokkos::Experimental::NextSiliconManagedSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    DeepCopySharedNextSilicon(dst, src, n);
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "Kokkos::Impl::DeepCopy<HostSpace, NextSiliconSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    DeepCopySharedNextSilicon(dst, src, n);
  }
};

}  // namespace Impl
}  // namespace Kokkos

#endif
