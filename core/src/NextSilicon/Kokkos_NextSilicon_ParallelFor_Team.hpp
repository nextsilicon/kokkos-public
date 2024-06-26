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

/*! \brief Team Parallel constructs for NextSilicon.
FIXME_NEXTSILICON: Currently implemented by using one range-policy iteration to
execute each team, AND team size = 1 and vector size = 1. The hierarchical
connstructs are written to support variable-sized teams and vectors that will be
mapped onto a 1D iteration space, to serve as a signpost for future
implementation work.

Assuming for a requested team size M and vector size N, at least M * N
underlying "threads" will be launched.

The team rank of a hardware thread is T / N
The vector rank of a hardware thread is T % N
*/

#ifndef KOKKOS_NEXTSILICON_PARALLEL_FOR_TEAM_HPP
#define KOKKOS_NEXTSILICON_PARALLEL_FOR_TEAM_HPP

#include <NextSilicon/Kokkos_NextSilicon_Team.hpp>
#include <NextSilicon/Kokkos_NextSilicon_FunctorAdapter.hpp>

#include <nsapi/intrinsics.h>
#if !defined(KOKKOS_ENABLE_IMPL_NSAPI_UNAVAIL)
#include <nsapi/parallelism.h>
#endif

template <typename Member, typename LeagueIndexType, typename Functor>
class NextSiliconParallelForTeamPolicyFunctor {
 public:
  NextSiliconParallelForTeamPolicyFunctor(Functor const& functor,
                                          const LeagueIndexType leagueSize)
      : functor_(functor), leagueSize_(leagueSize) {}

  KOKKOS_INLINE_FUNCTION void operator()(const LeagueIndexType leagueId) const {
    functor_(Member(leagueId, leagueSize_));
  }

 private:
  const Functor functor_;
  LeagueIndexType leagueSize_;
};

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
// Hierarchical Parallelism -> Team level implementation
template <class FunctorType, class... Properties>
class Kokkos::Impl::ParallelFor<FunctorType, Kokkos::TeamPolicy<Properties...>,
                                Kokkos::Experimental::NextSilicon> {
 private:
  using Policy =
      Kokkos::Impl::TeamPolicyInternal<Kokkos::Experimental::NextSilicon,
                                       Properties...>;
  Kokkos::Experimental::Impl::FunctorAdapter<FunctorType, Policy> m_functor;
  using Member = typename Policy::member_type;

  const Policy m_policy;

 public:
  // Invoke one RangePolicy parallel_for iteration per team
  inline void execute() const {
    auto leagueSize = m_policy.league_size();
    NextSiliconParallelForTeamPolicyFunctor<Member, decltype(leagueSize),
                                            decltype(m_functor)>
        op(m_functor, leagueSize);
    Kokkos::parallel_for(leagueSize, op);
  }

  inline ParallelFor(const FunctorType& arg_functor, Policy arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

namespace Kokkos {

// Hierarchical Parallelism -> Team thread level implementation
template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::TeamThreadRangeBoundariesStruct<
        iType, Impl::NextSiliconTeamMember>& loop_boundaries,
    const Lambda& lambda) {
  iType threadIdx = loop_boundaries.team.team_rank();
  iType j_start   = threadIdx + loop_boundaries.start;
  iType j_end     = loop_boundaries.end;
  iType j_step    = loop_boundaries.team.team_size();
  for (iType j = j_start; j < j_end; j += j_step) {
    lambda(j);
  }
}

// Hierarchical Parallelism -> Thread vector level implementation
template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::ThreadVectorRangeBoundariesStruct<
        iType, Impl::NextSiliconTeamMember>& loop_boundaries,
    const Lambda& lambda) {
  iType vectorIdx =
      nsapi_team_get_thread_index() % loop_boundaries.team.vector_length();
  iType j_start = loop_boundaries.start + vectorIdx;
  iType j_end   = loop_boundaries.end;
  iType j_step  = loop_boundaries.team.vector_length();
  for (iType j = j_start; j < j_end; j += j_step) {
    lambda(j);
  }
}

// Hierarchical Parallelism -> Team vector level implementation
template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::TeamVectorRangeBoundariesStruct<
        iType, Impl::NextSiliconTeamMember>& loop_boundaries,
    const Lambda& lambda) {
  iType j_start = loop_boundaries.start + nsapi_team_get_thread_index();
  iType j_end   = loop_boundaries.end;
  iType j_step =
      loop_boundaries.team.vector_length() * loop_boundaries.team.team_size();
  for (iType j = j_start; j < j_end; j += j_step) {
    lambda(j);
  }
}

}  // namespace Kokkos

#endif /* #ifndef KOKKOS_NEXTSILICON_PARALLEL_FOR_TEAM_HPP */
