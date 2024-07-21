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

#ifndef KOKKOS_NEXTSILICON_PARALLELREDUCE_RANGE_HPP
#define KOKKOS_NEXTSILICON_PARALLELREDUCE_RANGE_HPP

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Macros.hpp>
#include <NextSilicon/Kokkos_NextSilicon_FunctorAdapter.hpp>
#include <NextSilicon/Kokkos_NextSilicon_ScheduleType.hpp>
#include <nsapi/intrinsics.h>
#include <nsapi/parallelism.h>
#include <Kokkos_Parallel.hpp>
#include <type_traits>
#include <cstdint>

namespace Kokkos::Experimental::Impl {

// primary template: catch-all non-implemented custom reducers
template <class Functor, class Reducer, class Policy,
          bool = std::is_arithmetic_v<typename Reducer::value_type>>
struct NextSiliconParallelReduceHelper {
  NextSiliconParallelReduceHelper(Functor const&, Reducer const&,
                                  Policy const&) {
    static_assert(
        Kokkos::Impl::always_false<Functor>::value,
        "NextSilicon parallel_reduce not implemented for this value_type");
  }
};

}  // namespace Kokkos::Experimental::Impl

template <class CombinedFunctorReducerType, class... Traits>
class Kokkos::Impl::ParallelReduce<CombinedFunctorReducerType,
                                   Kokkos::RangePolicy<Traits...>,
                                   Kokkos::Experimental::NextSilicon> {
  using ExecSpace   = Kokkos::Experimental::NextSilicon;
  using MemorySpace = typename ExecSpace::memory_space;
  using Policy      = RangePolicy<Traits...>;
  using WorkTag     = typename Policy::work_tag;
  using WorkRange   = typename Policy::WorkRange;
  using FunctorType = typename CombinedFunctorReducerType::functor_type;
  using ReducerType = typename CombinedFunctorReducerType::reducer_type;

  using Pointer       = typename ReducerType::pointer_type;
  using ValueType     = typename ReducerType::value_type;
  using ReferenceType = typename ReducerType::reference_type;

  CombinedFunctorReducerType m_functor_reducer;
  Policy m_policy;
  Pointer m_result_ptr;
  bool m_result_ptr_on_device;

 public:
  template <class ViewType>
  ParallelReduce(CombinedFunctorReducerType const& functor_reducer,
                 Policy const& policy, ViewType const& result)
      : m_functor_reducer(functor_reducer),
        m_policy(policy),
        m_result_ptr(result.data()),
        // FIXME_NEXTSILICON should be true if ViewType::memory_space can
        // access NextSiliconSpace
        m_result_ptr_on_device(false) {}

#pragma ns mark boundary
  __attribute__((noinline)) void execute() const {
    auto const begin = m_policy.begin();
    auto const end   = m_policy.end();

    auto const& functor = m_functor_reducer.get_functor();
    auto const& reducer = m_functor_reducer.get_reducer();

    if (end <= begin) {
      ValueType val;
      reducer.init(&val);
      reducer.copy(m_result_ptr, &val);
      return;
    }

    if (__nsapi_is_on_cg()) {
      const auto num_iterations = end - begin;
      uint32_t team_size        = nsapi_team_get_optimal_size(
          reinterpret_cast<void*>(microtask), num_iterations);

      size_t scratch_memory_byte_size =
          team_size * sizeof(ValueType) * reducer.value_count();

      // TODO: Cache this allocation in the instance
      ValueType* values = reinterpret_cast<ValueType*>(
          MemorySpace().allocate(scratch_memory_byte_size));

      uint32_t* counter =
          reinterpret_cast<uint32_t*>(MemorySpace().allocate(sizeof(uint32_t)));
      // TODO: When moved to fixed allocation in instance, memset once.
      *counter = 0;

      nsapi_team_spawn(
          reinterpret_cast<void*>(microtask), team_size,
          /* memory_size */ 0,
          nsapi_team_dimensions{static_cast<uint64_t>(num_iterations)},
          &functor, &reducer, &m_policy, values, counter, m_result_ptr);

      // TODO: When moving to instance memory, don't deallocate
      MemorySpace().deallocate(values, scratch_memory_byte_size);
      MemorySpace().deallocate(counter, sizeof(*counter));

      // TODO: Fence if !m_result_ptr_on_device - not needed for now as
      // nsapi_team_spawn is always sync.
    } else {
      ValueType* value = reinterpret_cast<ValueType*>(
          alloca(sizeof(ValueType) * reducer.value_count()));
      uint32_t counter = 0;
      microtask(&functor, &reducer, &m_policy, value, &counter, m_result_ptr);
    }
  }

 private:
  static void microtask(const FunctorType* functor, const ReducerType* reducer,
                        const Policy* policy, ValueType* result_arr,
                        uint32_t* counter, ValueType* result_ptr) {
    uint32_t thread_index = nsapi_team_get_thread_index();
    uint32_t team_size    = nsapi_team_get_team_size();

    WorkRange range(*policy, thread_index, team_size);
    const auto ibeg  = range.begin();
    const auto iend  = range.end();
    auto value_count = reducer->value_count();

    ValueType* val =
        reinterpret_cast<ValueType*>(alloca(sizeof(ValueType) * value_count));
    ReferenceType ref = reducer->init(val);

    for (auto i = ibeg; i < iend; ++i) {
      if constexpr (std::is_void_v<WorkTag>) {
        (*functor)(i, ref);
      } else {
        (*functor)(WorkTag{}, i, ref);
      }
    }

    reducer->copy(&result_arr[thread_index * value_count], val);

    uint32_t counter_val = __atomic_add_fetch(counter, 1, __ATOMIC_ACQ_REL);
    if (counter_val < team_size) {
      return;
    }

    // Last thread to contribute reduction value (i.e counter got to team_size)
    *counter = 0;

    // Runs on a single thread, so running on RISC where single thread
    // performance is better
#pragma ns location risc
    for (uint32_t i = 1; i < team_size; ++i) {
      reducer->join(&result_arr[0], &result_arr[i * value_count]);
    }
    reducer->final(&result_arr[0]);
    reducer->copy(result_ptr, &result_arr[0]);
  }
};

#endif  // KOKKOS_NEXTSILICON_PARALLELREDUCE_RANGE_HPP
