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
#include <Kokkos_Parallel.hpp>
#include <type_traits>

namespace Kokkos::Experimental::Impl {

// primary template: catch-all non-implemented custom reducers
template <class Functor, class Reducer, class Policy,
          bool = std::is_arithmetic_v<typename Reducer::value_type>>
struct NextSiliconParallelReduceHelper {
  NextSiliconParallelReduceHelper(Functor const&, Reducer const&,
                                  Policy const&) {
    static_assert(
        std::is_void_v<Functor>::value,
        "NextSilicon parallel_reduce not implemented for this value_type");
  }
};

}  // namespace Kokkos::Experimental::Impl

template <class CombinedFunctorReducerType, class... Traits>
class Kokkos::Impl::ParallelReduce<CombinedFunctorReducerType,
                                   Kokkos::RangePolicy<Traits...>,
                                   Kokkos::Experimental::NextSilicon> {
  using Policy      = RangePolicy<Traits...>;
  using FunctorType = typename CombinedFunctorReducerType::functor_type;
  using ReducerType = typename CombinedFunctorReducerType::reducer_type;

  using Pointer   = typename ReducerType::pointer_type;
  using ValueType = typename ReducerType::value_type;

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

  void execute() const {
    auto const begin = m_policy.begin();
    auto const end   = m_policy.end();

    // FIXME_NEXTSILICON: this should all happen on the device
    ValueType val;
    ReducerType const& reducer = m_functor_reducer.get_reducer();
    reducer.init(&val);

    if (end <= begin) {
      if (m_result_ptr_on_device) {
        // FIXME_NEXTSILICON memcpy val to device
        *m_result_ptr = val;
      } else {
        *m_result_ptr = val;
      }
      return;
    }

    // ReducerType always has value_type.
    // ReducerType is a FunctorAnalysis object that might either wrap the
    // original functor (if the result type is a View or scalar) or the reducer
    // (if the result type is a reducer). If the result type is not a reducer,
    // we effectively do a sum reduction (but not using Kokkos::Sum, at least in
    // the generic implementation). It's fine to use Kokkos::Sum, though, if you
    // can't use the FunctorAnalysis object as the reducer.
    Kokkos::Experimental::Impl::NextSiliconParallelReduceHelper(
        m_functor_reducer.get_functor(),
        std::conditional_t<
            std::is_same_v<FunctorType, typename ReducerType::functor_type>,
            Sum<ValueType>, typename ReducerType::functor_type>(val),
        m_policy);

    // FIXME_NEXTSILICON: this should all happen on the device
    reducer.final(&val);

    // val is on the device,
    if (m_result_ptr_on_device) {
      *m_result_ptr = val;
      // FIXME_NEXTSILICON sync? device has made result available
    } else {
      // FIXME_NEXTSILICON sync: wait for device to finish
      *m_result_ptr = val;  // FIXME_NEXTSILICON memcpy val from device
    }
  }
};

// FIXME_NEXTSILICON: handling of static vs dynamic schedule?
#define KOKKOS_IMPL_NEXTSILICON_PARALLEL_REDUCE_DISPATCH_SCHEDULE(REDUCER,   \
                                                                  OPERATOR)  \
  namespace Kokkos::Experimental::Impl {                                     \
  template <class IndexType, class ValueType, class Functor>                 \
  void NextSiliconParallelReduce##REDUCER(Schedule<Static>, int chunk_size,  \
                                          IndexType begin, IndexType end,    \
                                          ValueType& aval,                   \
                                          Functor const& functor) {          \
    auto val = aval;                                                         \
    if (chunk_size >= 1) {                                                   \
      KOKKOS_IMPL_NS_PRAGMA(omp parallel for reduction(OPERATOR:val))        \
      for (auto i = begin; i < end; i++) {                                   \
        functor(i, val);                                                     \
      }                                                                      \
    } else {                                                                 \
      KOKKOS_IMPL_NS_PRAGMA(omp parallel for reduction(OPERATOR:val))        \
      for (auto i = begin; i < end; i++) {                                   \
        functor(i, val);                                                     \
      }                                                                      \
    }                                                                        \
    aval = val;                                                              \
  }                                                                          \
                                                                             \
  template <class IndexType, class ValueType, class Functor>                 \
  void NextSiliconParallelReduce##REDUCER(Schedule<Dynamic>, int chunk_size, \
                                          IndexType begin, IndexType end,    \
                                          ValueType& aval,                   \
                                          Functor const& functor) {          \
    auto val = aval;                                                         \
    if (chunk_size >= 1) {                                                   \
      KOKKOS_IMPL_NS_PRAGMA(omp parallel for reduction(OPERATOR:val))        \
      for (auto i = begin; i < end; i++) {                                   \
        functor(i, val);                                                     \
      }                                                                      \
    } else {                                                                 \
      KOKKOS_IMPL_NS_PRAGMA(omp parallel for reduction(OPERATOR:val))        \
      for (auto i = begin; i < end; i++) {                                   \
        functor(i, val);                                                     \
      }                                                                      \
    }                                                                        \
    aval = val;                                                              \
  }                                                                          \
  }  // namespace Kokkos::Experimental::Impl

#define KOKKOS_IMPL_NEXTSILICON_PARALLEL_REDUCE_HELPER(REDUCER, OPERATOR)      \
  KOKKOS_IMPL_NEXTSILICON_PARALLEL_REDUCE_DISPATCH_SCHEDULE(REDUCER, OPERATOR) \
  template <class Functor, class Scalar, class Space, class... Traits>         \
  struct Kokkos::Experimental::Impl::NextSiliconParallelReduceHelper<          \
      Functor, Kokkos::REDUCER<Scalar, Space>, Kokkos::RangePolicy<Traits...>, \
      true> {                                                                  \
    using Policy = RangePolicy<Traits...>;                                     \
    using ScheduleType =                                                       \
        Kokkos::Experimental::Impl::NextSiliconScheduleType<Policy>;           \
    using Reducer   = REDUCER<Scalar, Space>;                                  \
    using ValueType = typename Reducer::value_type;                            \
                                                                               \
    NextSiliconParallelReduceHelper(Functor const& functor,                    \
                                    Reducer const& reducer,                    \
                                    Policy const& policy) {                    \
      auto const begin = policy.begin();                                       \
      auto const end   = policy.end();                                         \
                                                                               \
      if (end <= begin) {                                                      \
        return;                                                                \
      }                                                                        \
                                                                               \
      ValueType val;                                                           \
      reducer.init(val);                                                       \
                                                                               \
      int const chunk_size = policy.chunk_size();                              \
                                                                               \
      NextSiliconParallelReduce##REDUCER(ScheduleType(), chunk_size, begin,    \
                                         end, val, functor);                   \
                                                                               \
      reducer.reference() = val;                                               \
    }                                                                          \
  }

// FIXME_NEXTSILICON: workaround for OpenMP not supporting custom reductions
KOKKOS_IMPL_NEXTSILICON_PARALLEL_REDUCE_HELPER(Sum, +);
KOKKOS_IMPL_NEXTSILICON_PARALLEL_REDUCE_HELPER(Prod, *);
KOKKOS_IMPL_NEXTSILICON_PARALLEL_REDUCE_HELPER(Min, min);
KOKKOS_IMPL_NEXTSILICON_PARALLEL_REDUCE_HELPER(Max, max);
KOKKOS_IMPL_NEXTSILICON_PARALLEL_REDUCE_HELPER(LAnd, &&);
KOKKOS_IMPL_NEXTSILICON_PARALLEL_REDUCE_HELPER(LOr, ||);
KOKKOS_IMPL_NEXTSILICON_PARALLEL_REDUCE_HELPER(BAnd, &);
KOKKOS_IMPL_NEXTSILICON_PARALLEL_REDUCE_HELPER(BOr, |);

#undef KOKKOS_IMPL_NEXTSILICON_PARALLEL_REDUCE_HELPER
#undef KOKKOS_IMPL_NEXTSILICON_PARALLEL_REDUCE_DISPATCH_SCHEDULE

#endif  // KOKKOS_NEXTSILICON_PARALLELREDUCE_RANGE_HPP
