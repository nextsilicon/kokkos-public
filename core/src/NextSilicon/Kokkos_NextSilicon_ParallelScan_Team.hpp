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

#ifndef KOKKOS_NEXTSILICON_PARALLEL_SCAN_TEAM_HPP
#define KOKKOS_NEXTSILICON_PARALLEL_SCAN_TEAM_HPP

#include <NextSilicon/Kokkos_NextSilicon_Team.hpp>
#include <NextSilicon/Kokkos_NextSilicon_FunctorAdapter.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Macros.hpp>

namespace Kokkos {

template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION std::enable_if_t<Kokkos::is_reducer<ReducerType>::value>
parallel_scan(const Impl::ThreadVectorRangeBoundariesStruct<
                  iType, Impl::NextSiliconTeamMember>& loop_boundaries,
              const Lambda& lambda, const ReducerType& reducer) {
  using value_type = typename ReducerType::value_type;
  using functor_analysis_type =
      typename Impl::FunctorAnalysis<Impl::FunctorPatternInterface::SCAN, void,
                                     ReducerType, value_type>;
  using WrappedReducer = typename functor_analysis_type::Reducer;
  if (0 ==
      nsapi_team_get_thread_index() % loop_boundaries.team.vector_length()) {
    WrappedReducer wrappedReducer(reducer);
    value_type val;
    wrappedReducer.init(&val);

    for (iType i = loop_boundaries.start; i < loop_boundaries.end; i++)
      lambda(i, val, /*final*/ true);
    wrappedReducer.final(&val);
    reducer.reference() = val;
  }
}

template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::NextSiliconTeamMember>& loop_boundaries,
    const Lambda& lambda) {
  using value_type = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::SCAN, void, Lambda,
      void>::value_type;
  value_type dummy;
  parallel_scan(loop_boundaries, lambda, Kokkos::Sum<value_type>(dummy));
}

template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::NextSiliconTeamMember>& loop_boundaries,
    const Lambda& lambda, ValueType& return_val) {
  using lambda_value_type = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::SCAN, void, Lambda,
      ValueType>::value_type;
  static_assert(std::is_same<lambda_value_type, ValueType>::value,
                "Non-matching value types of lambda and return type");

  ValueType accum;
  parallel_scan(loop_boundaries, lambda, Kokkos::Sum<ValueType>(accum));

  return_val = accum;
}

}  // namespace Kokkos

#endif /* #ifndef KOKKOS_NEXTSILICON_PARALLEL_SCAN_TEAM_HPP */
