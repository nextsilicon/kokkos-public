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
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<!Kokkos::is_reducer_v<ValueType>>
parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                    iType, Impl::NextSiliconTeamMember>& loop_boundaries,
                const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();
  iType j_start =
      loop_boundaries.team.team_rank() / loop_boundaries.team.vector_length();
  if (j_start == 0) {
    for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
      lambda(i, tmp);
    result = tmp;
  }
}

template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer_v<ReducerType>>
parallel_reduce(const Impl::TeamThreadRangeBoundariesStruct<
                    iType, Impl::NextSiliconTeamMember>& loop_boundaries,
                const Lambda& lambda, const ReducerType& reducer) {
  using ValueType = typename ReducerType::value_type;
  ValueType tmp;
  reducer.init(tmp);
  iType j_start =
      loop_boundaries.team.team_rank() / loop_boundaries.team.vector_length();
  if (j_start == 0) {
    for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
      lambda(i, tmp);
    reducer.reference() = tmp;
  }
}

// Hierarchical Parallelism -> Thread vector level implementation
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<!Kokkos::is_reducer_v<ValueType>>
parallel_reduce(const Impl::ThreadVectorRangeBoundariesStruct<
                    iType, Impl::NextSiliconTeamMember>& loop_boundaries,
                const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();
  iType j_start =
      loop_boundaries.team.team_rank() % loop_boundaries.team.vector_length();
  if (j_start == 0) {
    for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
      lambda(i, tmp);
    }
    result = tmp;
  }
}

template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer_v<ReducerType>>
parallel_reduce(const Impl::ThreadVectorRangeBoundariesStruct<
                    iType, Impl::NextSiliconTeamMember>& loop_boundaries,
                const Lambda& lambda, const ReducerType& reducer) {
  using ValueType = typename ReducerType::value_type;
  ValueType tmp;
  reducer.init(tmp);
  iType j_start =
      loop_boundaries.team.team_rank() % loop_boundaries.team.vector_length();
  if (j_start == 0) {
    for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
      lambda(i, tmp);
    }
    reducer.reference() = tmp;
  }
}

// Hierarchical Parallelism -> Team vector level implementation
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamVectorRangeBoundariesStruct<
        iType, Impl::NextSiliconTeamMember>& loop_boundaries,
    const Lambda& lambda, ValueType& result) {
  ValueType tmp = ValueType();
  iType j_start =
      loop_boundaries.team.team_rank() % loop_boundaries.team.vector_length();
  if (j_start == 0) {
    for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++) {
      lambda(i, tmp);
    }
    result = tmp;
  }
}

}  // namespace Kokkos

#endif /* #ifndef KOKKOS_NEXTSILICON_PARALLEL_REDUCE_TEAM_HPP */
