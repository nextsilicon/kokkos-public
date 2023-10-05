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

    if (__nsapi_is_on_cg()) {
      const auto num_iterations = end - begin;
      uint32_t team_size        = __ns_optimal_team_size(
          reinterpret_cast<void*>(microtask), num_iterations);

      void* team_handle = __ns_team_allocate(team_size);

      uint32_t base_nsptid      = __ns_params_get_team_base(team_handle);
      uint32_t actual_team_size = __ns_params_get_team_size(team_handle);

      NSInvariantsScope scope_id = __ns_scope_invariants_create();

      __ns_scope_invariants_enter(scope_id, &m_functor, base_nsptid,
                                  actual_team_size, &m_policy);

      // FIXME_NEXTSILICON: Use async version of `__ns_team_spawn` once it is
      // available
      __ns_team_spawn(team_handle, reinterpret_cast<void*>(microtask),
                      &m_functor, base_nsptid, actual_team_size, &m_policy);

      __ns_scope_invariants_leave(scope_id);
    } else {
      // FIXME_NEXTSILICON Run microtask on host for training
      // for now run in serial.
      microtask(&m_functor, /* base_nsptid */ 0, /* team_size */ 1, &m_policy);
    }
  }

 private:
// FIXME_NEXTSILICON Because there is no load on microtask / no training
#pragma ns mark import_recursive
  __attribute__((noinline)) static void microtask(
      FunctorWrapper const* __restrict functor, uint32_t base_nsptid,
      uint32_t team_size, const Policy* policy) {
    auto nsptid = []() -> uint32_t {
      if (!__nsapi_is_on_cg()) return 0;
      return __nsapi_get_ns_raw_tid();
    }();
    uint32_t thread_index = nsptid - base_nsptid;

    WorkRange range(*policy, thread_index, team_size);
    const auto ibeg = range.begin();
    const auto iend = range.end();

    for (auto i = ibeg; i < iend; ++i) {
      // Invokes the functor itself. Expected to inline (if compiler visible)
      // the functor body while passing the extra this pointer.
      (*functor)(i);
    }
  }
};

#endif  // KOKKOS_NEXTSILICON_PARALLELFOR_RANGE_HPP
