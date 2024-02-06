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

#ifndef KOKKOS_NEXTSILICON_PARALLELFOR_RANGE_HPP
#define KOKKOS_NEXTSILICON_PARALLELFOR_RANGE_HPP

#include <nsapi/intrinsics.h>
#include <nsapi/parallelism.h>

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSilicon_FunctorAdapter.hpp>
#include <Kokkos_Parallel.hpp>

template <class Functor, class... Traits>
class Kokkos::Impl::ParallelFor<Functor, Kokkos::RangePolicy<Traits...>,
                                Kokkos::Experimental::NextSilicon> {
  using Policy    = Kokkos::RangePolicy<Traits...>;
  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;

  using FunctorWrapper =
      Kokkos::Experimental::Impl::FunctorAdapter<Functor, Policy>;

  FunctorWrapper m_functor;
  Policy m_policy;

 public:
  ParallelFor(Functor const& functor, Policy const& policy)
      : m_functor(functor), m_policy(policy) {}

#pragma ns mark boundary
  __attribute__((noinline)) void execute() const {
    // FIXME_NEXTSILICON: Add a dynamic schedule policy check if `typename
    // Policy::schedule_type::type` is Kokkos::Static or Kokkos::Dynamic
    auto const begin = m_policy.begin();
    auto const end   = m_policy.end();

    if (end <= begin) return;

#ifdef KOKKOS_ENABLE_IMPL_NSAPI_UNAVAIL

#pragma omp parallel for
    for (auto i = begin; i < end; ++i) {
      m_functor(i);
    }

#else

    if (__nsapi_is_on_cg()) {
      const auto num_iterations = end - begin;
      uint32_t team_size        = nsapi_team_get_optimal_size(
          reinterpret_cast<void*>(microtask), num_iterations);

      // FIXME_NEXTSILICON: Change function and policy capture to be more
      // efficient.
      nsapi_team_spawn(
          reinterpret_cast<void*>(microtask), team_size, /* memory_size */ 0,
          nsapi_team_dimensions{static_cast<uint64_t>(num_iterations)},
          &m_functor, &m_policy);
    } else {
      // FIXME_NEXTSILICON Run microtask on host for training
      // for now run in serial.
      microtask(&m_functor, &m_policy);
    }
#endif
  }

#ifndef KOKKOS_ENABLE_IMPL_NSAPI_UNAVAIL

 private:
  __attribute__((noinline)) static void microtask(
      FunctorWrapper const* __restrict functor, const Policy* policy) {
    // TODO: On the device, the optimizer should turn those calls into
    // invariants/feeders
    // On host we're doing a serial for now, so the default values
    // team_size = 1 and team_index = 0 are valid.
    uint32_t thread_index = nsapi_team_get_thread_index();
    uint32_t team_size    = nsapi_team_get_team_size();

    WorkRange range(*policy, thread_index, team_size);
    const auto ibeg = range.begin();
    const auto iend = range.end();

    for (auto i = ibeg; i < iend; ++i) {
      // Invokes the functor itself. Expected to inline (if compiler visible)
      // the functor body while passing the extra this pointer.
      (*functor)(i);
    }
  }
#endif
};

#endif  // KOKKOS_NEXTSILICON_PARALLELFOR_RANGE_HPP
