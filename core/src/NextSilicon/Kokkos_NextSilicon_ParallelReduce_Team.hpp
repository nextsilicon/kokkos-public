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

#ifndef KOKKOS_NEXTSILICON_PARALLEL_REDUCE_TEAM_HPP
#define KOKKOS_NEXTSILICON_PARALLEL_REDUCE_TEAM_HPP

#include <NextSilicon/Kokkos_NextSilicon_Team.hpp>
#include <NextSilicon/Kokkos_NextSilicon_FunctorAdapter.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Macros.hpp>

#include <nsapi/intrinsics.h>
#if !defined(KOKKOS_ENABLE_IMPL_NSAPI_UNAVAIL)
#include <nsapi/parallelism.h>
#endif

template <class CombinedFunctorReducerType, class... Properties>
class Kokkos::Impl::ParallelReduce<CombinedFunctorReducerType,
                                   Kokkos::TeamPolicy<Properties...>,
                                   Kokkos::Experimental::NextSilicon> {
 private:
  using Policy =
      TeamPolicyInternal<Kokkos::Experimental::NextSilicon, Properties...>;
  using Member      = typename Policy::member_type;
  using FunctorType = typename CombinedFunctorReducerType::functor_type;
  using ReducerType = typename CombinedFunctorReducerType::reducer_type;

  using value_type   = typename ReducerType::value_type;
  using pointer_type = typename ReducerType::pointer_type;

  CombinedFunctorReducerType m_functor_reducer;
  Policy m_policy;
  pointer_type m_result_ptr;
  bool m_result_ptr_on_device;

 public:
  void execute() const {
    // FIXME_NEXTSILICON: team parallel reduce unimplemented
    throw std::runtime_error(
        "team parallel_reduce for Next Silicon unimplemented");
  }

  template <class ViewType>
  ParallelReduce(const CombinedFunctorReducerType& arg_functor_reducer,
                 const Policy& arg_policy, const ViewType& arg_result_view)
      : m_functor_reducer(arg_functor_reducer),
        m_policy(arg_policy),
        m_result_ptr(arg_result_view.data()),
        m_result_ptr_on_device(
            MemorySpaceAccess<Kokkos::Experimental::NextSiliconSpace,
                              typename ViewType::memory_space>::accessible) {}
};

namespace Kokkos {

// Hierarchical Parallelism -> Team thread level implementation
// FIXME_NEXTSILICON: single-thread implementation
template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer<ReducerType>::value>
parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                    iType, Impl::NextSiliconTeamMember>& loop_boundaries,
                const Lambda& lambda, const ReducerType& reducer) {
  using value_type     = typename ReducerType::value_type;
  using WrappedReducer = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::REDUCE, 
      TeamPolicy<typename Impl::NextSiliconTeamMember::execution_space>, ReducerType,
      value_type>::Reducer;

  if (0 == loop_boundaries.team.team_rank()) {
    WrappedReducer wrappedReducer(reducer);
    value_type val;
    wrappedReducer.init(&val);

    for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
      lambda(i, val);
    wrappedReducer.final(&val);
    reducer.reference() = val; // wrappedReducer.reference() = val;
  }
}
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<!Kokkos::is_reducer<ValueType>::value>
parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                    iType, Impl::NextSiliconTeamMember>& loop_boundaries,
                const Lambda& lambda, ValueType& result) {
  using WrappedReducer = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::REDUCE, 
      TeamPolicy<typename Impl::NextSiliconTeamMember::execution_space>, Lambda,
      ValueType>::Reducer;

  static_assert(std::is_same_v<ValueType, typename WrappedReducer::value_type>);
  if (0 == loop_boundaries.team.team_rank()) {
    ValueType val;
    WrappedReducer wrappedReducer(lambda);
    wrappedReducer.init(&val);

    for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
      lambda(i, val);
    wrappedReducer.final(&val);
    result = val;
  }
}

// Hierarchical Parallelism -> Thread vector level implementation
// FIXME_NEXTSILICON: single-vector implementation
template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer<ReducerType>::value>
parallel_reduce(const Impl::ThreadVectorRangeBoundariesStruct<
                    iType, Impl::NextSiliconTeamMember>& loop_boundaries,
                const Lambda& lambda, const ReducerType& reducer) {
  using value_type     = typename ReducerType::value_type;
  using WrappedReducer = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::REDUCE,
      TeamPolicy<typename Impl::NextSiliconTeamMember::execution_space>, ReducerType,
      value_type>::Reducer;

  if (0 ==
      nsapi_team_get_thread_index() % loop_boundaries.team.vector_length()) {
    WrappedReducer wrappedReducer(reducer);
    value_type val;
    wrappedReducer.init(&val);

    for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
      lambda(i, val);
    wrappedReducer.final(&val);
    reducer.reference() = val; // wrappedReducer.reference() = val;
  }
}
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<!Kokkos::is_reducer<ValueType>::value>
parallel_reduce(const Impl::ThreadVectorRangeBoundariesStruct<
                    iType, Impl::NextSiliconTeamMember>& loop_boundaries,
                const Lambda& lambda, ValueType& result) {
  using WrappedReducer = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::REDUCE, 
      TeamPolicy<typename Impl::NextSiliconTeamMember::execution_space>, Lambda,
      ValueType>::Reducer;

  static_assert(std::is_same_v<ValueType, typename WrappedReducer::value_type>);
  if (0 ==
      nsapi_team_get_thread_index() % loop_boundaries.team.vector_length()) {
    ValueType val;
    WrappedReducer wrappedReducer(lambda);
    wrappedReducer.init(&val);

    for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
      lambda(i, val);
    wrappedReducer.final(&val);
    result = val;
  }
}

// Hierarchical Parallelism -> Team vector level implementation
// FIXME_NEXTSILICON: single-vector implementation
template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer<ReducerType>::value>
parallel_reduce(const Impl::TeamVectorRangeBoundariesStruct<
                    iType, Impl::NextSiliconTeamMember>& loop_boundaries,
                const Lambda& lambda, const ReducerType& reducer) {
  using value_type     = typename ReducerType::value_type;
  using WrappedReducer = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::REDUCE, 
      TeamPolicy<typename Impl::NextSiliconTeamMember::execution_space>,
      ReducerType,
      value_type>::Reducer;

  if (0 == nsapi_team_get_thread_index()) {
    WrappedReducer wrappedReducer(reducer);
    value_type val;
    wrappedReducer.init(&val);

    for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
      lambda(i, val);
    wrappedReducer.final(&val);
    wrappedReducer.reference() = val;
  }
}
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<!Kokkos::is_reducer<ValueType>::value>
parallel_reduce(const Impl::TeamVectorRangeBoundariesStruct<
                    iType, Impl::NextSiliconTeamMember>& loop_boundaries,
                const Lambda& lambda, ValueType& result) {
  using WrappedReducer = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::REDUCE, 
      TeamPolicy<typename Impl::NextSiliconTeamMember::execution_space>, Lambda,
      ValueType>::Reducer;

  static_assert(std::is_same_v<ValueType, typename WrappedReducer::value_type>);
  if (0 == nsapi_team_get_thread_index()) {
    ValueType val;
    WrappedReducer wrappedReducer(lambda);
    wrappedReducer.init(&val);

    for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
      lambda(i, val);
    wrappedReducer.final(&val);
    result = val;
  }
}

}  // namespace Kokkos

#endif /* #ifndef KOKKOS_NEXTSILICON_PARALLEL_REDUCE_TEAM_HPP */
