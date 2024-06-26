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

#ifndef KOKKOS_NEXTSILICON_TEAM_HPP
#define KOKKOS_NEXTSILICON_TEAM_HPP

#include <impl/Kokkos_Traits.hpp>
#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <Kokkos_ExecPolicy.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

class NextSiliconTeamMember {
 public:
  // FIXME_NEXTSILICON: skeleton team size is always 1
  static constexpr int TEAM_SIZE     = 1;
  static constexpr int TEAM_RANK     = 0;
  static constexpr int VECTOR_LENGTH = 1;

  using execution_space      = Kokkos::Experimental::NextSilicon;
  using scratch_memory_space = execution_space::scratch_memory_space;
  using team_handle          = NextSiliconTeamMember;

  scratch_memory_space m_team_shared;
  int m_team_scratch_size[2];
  int m_league_rank;
  int m_league_size;

 public:
  KOKKOS_FUNCTION
  const execution_space::scratch_memory_space& team_shmem() const {
    return m_team_shared.set_team_thread_mode(0, 1, 0);
  }

  KOKKOS_FUNCTION
  const execution_space::scratch_memory_space& team_scratch(int level) const {
    return m_team_shared.set_team_thread_mode(level, 1,
                                              m_team_scratch_size[level]);
  }

  KOKKOS_FUNCTION
  const execution_space::scratch_memory_space& thread_scratch(int level) const {
    return m_team_shared.set_team_thread_mode(level, team_size(), team_rank());
  }

  KOKKOS_FUNCTION int league_rank() const { return m_league_rank; }
  KOKKOS_FUNCTION int league_size() const { return m_league_size; }
  KOKKOS_FUNCTION int team_rank() const { return TEAM_RANK; }
  KOKKOS_FUNCTION int vector_length() const { return VECTOR_LENGTH; }
  KOKKOS_FUNCTION int team_size() const { return TEAM_SIZE; }

  KOKKOS_FUNCTION void team_barrier() const {
    // FIXME_NEXTSILICON: team_barrier is a no-op, okay while team size is
    // always 1
  }

  template <class ValueType>
  KOKKOS_FUNCTION void team_broadcast(ValueType& value, int thread_id) const {
    (void)value;
    (void)thread_id;
    // FIXME_NEXTSILICON: team_broadcast is a no-op, okay while team size is
    // always 1
  }

  template <class Closure, class ValueType>
  KOKKOS_FUNCTION void team_broadcast(const Closure& f, ValueType& value,
                                      int thread_id) const {
    f(value);
    team_broadcast(value, thread_id);
  }

  // FIXME_NEXTSILICON: team_reduce() is not implemented.
  template <class ValueType, class JoinOp>
  KOKKOS_FUNCTION ValueType team_reduce(const ValueType& /*value*/,
                                        const JoinOp& /*op_in*/) const {
    static_assert(!Kokkos::Impl::always_true<ValueType>::value,
                  "Kokkos Error: team_reduce() is not implemented for the "
                  "NextSilicon backend");
    return ValueType();
  }

  // FIXME_NEXTSILICON: team_scan() assumes a team size of 1
  template <typename ArgType>
  KOKKOS_FUNCTION ArgType team_scan(const ArgType& value,
                                    ArgType* const global_accum) const {
    static_assert(TEAM_SIZE == 1,
                  "FIXME_NEXTSILICON: team_scan() assumes a team size of 1");
    if (global_accum) {
      *global_accum = value;
    }
    return ArgType(0);  // exclusive scan
  }

  template <typename Type>
  KOKKOS_FUNCTION Type team_scan(const Type& value) const {
    return this->template team_scan<Type>(value, 0);
  }

  //----------------------------------------
  // Private for the driver

 private:
  using space = execution_space::scratch_memory_space;

 public:
  NextSiliconTeamMember(
      const int league_rank,
      const int league_size)  // const TeamPolicyInternal<NextSilicon,
                              // Properties ...> & team
      : m_league_rank(league_rank), m_league_size(league_size) {}

  static int team_reduce_size() { return TEAM_SIZE; }
};

template <class... Properties>
class TeamPolicyInternal<Kokkos::Experimental::NextSilicon, Properties...>
    : public PolicyTraits<Properties...> {
 public:
  //! Tag this class as a kokkos execution policy
  using execution_policy = TeamPolicyInternal;

  using traits = PolicyTraits<Properties...>;

  //----------------------------------------

  // FIXME_NEXTSILICON: team_size_max always 1
  template <class FunctorType>
  static int team_size_max(const FunctorType&, const ParallelForTag&) {
    return NextSiliconTeamMember::TEAM_SIZE;
  }

  // FIXME_NEXTSILICON: team_size_max always 1
  template <class FunctorType>
  static int team_size_max(const FunctorType&, const ParallelReduceTag&) {
    return NextSiliconTeamMember::TEAM_SIZE;
  }

  // FIXME_NEXTSILICON: team_size_max always 1
  template <class FunctorType, class ReducerType>
  static int team_size_max(const FunctorType&, const ReducerType&,
                           const ParallelReduceTag&) {
    return NextSiliconTeamMember::TEAM_SIZE;
  }

  // FIXME_NEXTSILICON: team_size_recommended always 1
  template <class FunctorType>
  static int team_size_recommended(const FunctorType&, const ParallelForTag&) {
    return NextSiliconTeamMember::TEAM_SIZE;
  }

  // FIXME_NEXTSILICON: team_size_recommended always 1
  template <class FunctorType>
  static int team_size_recommended(const FunctorType&,
                                   const ParallelReduceTag&) {
    return NextSiliconTeamMember::TEAM_SIZE;
  }

  // FIXME_NEXTSILICON: team_size_recommended always 1
  template <class FunctorType, class ReducerType>
  static int team_size_recommended(const FunctorType&, const ReducerType&,
                                   const ParallelReduceTag&) {
    return NextSiliconTeamMember::TEAM_SIZE;
  }

  //----------------------------------------

 private:
  int m_league_size;
  int m_team_alloc;
  int m_team_iter;
  std::array<size_t, 2> m_team_scratch_size;
  std::array<size_t, 2> m_thread_scratch_size;
  constexpr static int default_team_size_max = NextSiliconTeamMember::TEAM_SIZE;
  constexpr static int default_team_size     = NextSiliconTeamMember::TEAM_SIZE;
  int m_chunk_size;

  void init(const int league_size_request, const int team_size_request,
            const int vector_length_request) {
    m_league_size = league_size_request;
    if (team_size_request != NextSiliconTeamMember::TEAM_SIZE) {
      Kokkos::abort("Kokkos::abort: Requested Team Size for NextSilicon != 1!");
    }
    if (vector_length_request != NextSiliconTeamMember::VECTOR_LENGTH) {
      Kokkos::abort(
          "Kokkos::abort: Requested Vector Length for NextSilicon != 1!");
    }
    set_auto_chunk_size();
  }

  template <typename ExecSpace, typename... OtherProperties>
  friend class TeamPolicyInternal;

 public:
  bool impl_auto_team_size() const {
    return false;  // FIXME_NEXTSILICON: team size is always automatically
                   // chosen to be 1
  }
  bool impl_auto_vector_length() const {
    return false;  // FIXME_NEXTSILICON: team size is always automatically
                   // chosen to be 1;
  }
  void impl_set_team_size(const int size) {
    // FIXME_NEXTSILICON: team size is always 1
    (void)size;
  }
  void impl_set_vector_length(const int length) {
    // FIXME_NEXTSILICON: team size is always 1
    (void)length;
  }
  int impl_vector_length() const {
    // FIXME_NEXTSILICON: vector length is always 1
    return NextSiliconTeamMember::VECTOR_LENGTH;
  }
  int team_size() const {
    // FIXME_NEXTSILICON: team size is always 1
    return NextSiliconTeamMember::TEAM_SIZE;
  }
  int league_size() const { return m_league_size; }
  size_t scratch_size(const int& level, int team_size_ = -1) const {
    if (team_size_ < 0) team_size_ = NextSiliconTeamMember::TEAM_SIZE;
    return m_team_scratch_size[level] +
           team_size_ * m_thread_scratch_size[level];
  }

  Kokkos::Experimental::NextSilicon space() const {
    return Kokkos::Experimental::NextSilicon();
  }

  template <class... OtherProperties>
  TeamPolicyInternal(const TeamPolicyInternal<OtherProperties...>& p)
      : m_league_size(p.m_league_size),
        m_team_alloc(p.m_team_alloc),
        m_team_iter(p.m_team_iter),
        m_team_scratch_size(p.m_team_scratch_size),
        m_thread_scratch_size(p.m_thread_scratch_size),
        m_chunk_size(p.m_chunk_size) {}

  /** \brief  Specify league size, request team size */
  TeamPolicyInternal(const typename traits::execution_space&,
                     int league_size_request, int team_size_request,
                     int vector_length_request = 1)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, team_size_request, vector_length_request);
  }

  TeamPolicyInternal(const typename traits::execution_space&,
                     int league_size_request,
                     const Kokkos::AUTO_t& /* team_size_request */
                     ,
                     int vector_length_request = 1)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, default_team_size, vector_length_request);
  }

  TeamPolicyInternal(const typename traits::execution_space&,
                     int league_size_request,
                     const Kokkos::AUTO_t& /* team_size_request */
                     ,
                     const Kokkos::AUTO_t& /* vector_length_request */)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, default_team_size, 1);
  }
  TeamPolicyInternal(const typename traits::execution_space&,
                     int league_size_request, int team_size_request,
                     const Kokkos::AUTO_t& /* vector_length_request */)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, team_size_request, 1);
  }

  TeamPolicyInternal(int league_size_request, int team_size_request,
                     int vector_length_request = 1)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, team_size_request, vector_length_request);
  }

  TeamPolicyInternal(int league_size_request,
                     const Kokkos::AUTO_t& /* team_size_request */
                     ,
                     int vector_length_request = 1)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, default_team_size, vector_length_request);
  }

  TeamPolicyInternal(int league_size_request,
                     const Kokkos::AUTO_t& /* team_size_request */
                     ,
                     const Kokkos::AUTO_t& /* vector_length_request */)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, default_team_size, 1);
  }
  TeamPolicyInternal(int league_size_request, int team_size_request,
                     const Kokkos::AUTO_t& /* vector_length_request */)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, team_size_request, 1);
  }
  static int vector_length_max() {
    // FIXME_NEXTSILICON: vector length is always 1
    return NextSiliconTeamMember::VECTOR_LENGTH;
  }
  int team_alloc() const { return m_team_alloc; }
  int team_iter() const { return m_team_iter; }
  int chunk_size() const { return m_chunk_size; }

  /** \brief set chunk_size to a discrete value*/
  TeamPolicyInternal& set_chunk_size(typename traits::index_type chunk_size_) {
    m_chunk_size = chunk_size_;
    return *this;
  }

  /** \brief set per team scratch size for a specific level of the scratch
   * hierarchy */
  TeamPolicyInternal& set_scratch_size(const int& level,
                                       const PerTeamValue& per_team) {
    m_team_scratch_size[level] = per_team.value;
    return *this;
  }

  /** \brief set per thread scratch size for a specific level of the scratch
   * hierarchy */
  TeamPolicyInternal& set_scratch_size(const int& level,
                                       const PerThreadValue& per_thread) {
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }

  /** \brief set per thread and per team scratch size for a specific level of
   * the scratch hierarchy */
  TeamPolicyInternal& set_scratch_size(const int& level,
                                       const PerTeamValue& per_team,
                                       const PerThreadValue& per_thread) {
    m_team_scratch_size[level]   = per_team.value;
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }

 private:
  /** \brief finalize chunk_size if it was set to AUTO*/
  void set_auto_chunk_size() {
    // FIXME_NEXTSILICON: set_auto_chunk_size logic copied from OpenACC, needs
    // to be redone

    int concurrency = 2048 * default_team_size;

    if (m_chunk_size > 0) {
      if (!Impl::is_integral_power_of_two(m_chunk_size))
        Kokkos::abort("TeamPolicy blocking granularity must be power of two");
    }

    int new_chunk_size = 1;
    while (new_chunk_size * 100 * concurrency < m_league_size)
      new_chunk_size *= 2;
    if (new_chunk_size < default_team_size) {
      new_chunk_size = 1;
      while ((new_chunk_size * 40 * concurrency < m_league_size) &&
             (new_chunk_size < default_team_size))
        new_chunk_size *= 2;
    }
    m_chunk_size = new_chunk_size;
  }

 public:
  using member_type = Impl::NextSiliconTeamMember;
};
}  // namespace Impl
}  // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <typename iType>
struct TeamThreadRangeBoundariesStruct<iType, NextSiliconTeamMember> {
  using index_type = iType;
  const iType start;
  const iType end;
  const NextSiliconTeamMember& team;

  TeamThreadRangeBoundariesStruct(const NextSiliconTeamMember& thread_,
                                  iType count)
      : start(0), end(count), team(thread_) {}
  TeamThreadRangeBoundariesStruct(const NextSiliconTeamMember& thread_,
                                  iType begin_, iType end_)
      : start(begin_), end(end_), team(thread_) {}
};

template <typename iType>
struct ThreadVectorRangeBoundariesStruct<iType, NextSiliconTeamMember> {
  using index_type = iType;
  const index_type start;
  const index_type end;
  const NextSiliconTeamMember& team;

  ThreadVectorRangeBoundariesStruct(const NextSiliconTeamMember& thread_,
                                    index_type count)
      : start(0), end(count), team(thread_) {}
  ThreadVectorRangeBoundariesStruct(const NextSiliconTeamMember& thread_,
                                    index_type begin_, index_type end_)
      : start(begin_), end(end_), team(thread_) {}
};

template <typename iType>
struct TeamVectorRangeBoundariesStruct<iType, NextSiliconTeamMember> {
  using index_type = iType;
  const index_type start;
  const index_type end;
  const NextSiliconTeamMember& team;

  TeamVectorRangeBoundariesStruct(const NextSiliconTeamMember& thread_,
                                  index_type count)
      : start(0), end(count), team(thread_) {}
  TeamVectorRangeBoundariesStruct(const NextSiliconTeamMember& thread_,
                                  index_type begin_, index_type end_)
      : start(begin_), end(end_), team(thread_) {}
};

}  // namespace Impl
}  // namespace Kokkos

namespace Kokkos {

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::TeamThreadRangeBoundariesStruct<iType, Impl::NextSiliconTeamMember>
    TeamThreadRange(const Impl::NextSiliconTeamMember& thread,
                    const iType& count) {
  return Impl::TeamThreadRangeBoundariesStruct<iType,
                                               Impl::NextSiliconTeamMember>(
      thread, count);
}

template <typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION Impl::TeamThreadRangeBoundariesStruct<
    typename std::common_type<iType1, iType2>::type,
    Impl::NextSiliconTeamMember>
TeamThreadRange(const Impl::NextSiliconTeamMember& thread, const iType1& begin,
                const iType2& end) {
  using iType = typename std::common_type<iType1, iType2>::type;
  return Impl::TeamThreadRangeBoundariesStruct<iType,
                                               Impl::NextSiliconTeamMember>(
      thread, iType(begin), iType(end));
}

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::NextSiliconTeamMember>
    ThreadVectorRange(const Impl::NextSiliconTeamMember& thread,
                      const iType& count) {
  return Impl::ThreadVectorRangeBoundariesStruct<iType,
                                                 Impl::NextSiliconTeamMember>(
      thread, count);
}

template <typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION Impl::ThreadVectorRangeBoundariesStruct<
    typename std::common_type<iType1, iType2>::type,
    Impl::NextSiliconTeamMember>
ThreadVectorRange(const Impl::NextSiliconTeamMember& thread,
                  const iType1& arg_begin, const iType2& arg_end) {
  using iType = typename std::common_type<iType1, iType2>::type;
  return Impl::ThreadVectorRangeBoundariesStruct<iType,
                                                 Impl::NextSiliconTeamMember>(
      thread, iType(arg_begin), iType(arg_end));
}

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::TeamVectorRangeBoundariesStruct<iType, Impl::NextSiliconTeamMember>
    TeamVectorRange(const Impl::NextSiliconTeamMember& thread,
                    const iType& count) {
  return Impl::TeamVectorRangeBoundariesStruct<iType,
                                               Impl::NextSiliconTeamMember>(
      thread, count);
}

template <typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION Impl::TeamVectorRangeBoundariesStruct<
    typename std::common_type<iType1, iType2>::type,
    Impl::NextSiliconTeamMember>
TeamVectorRange(const Impl::NextSiliconTeamMember& thread,
                const iType1& arg_begin, const iType2& arg_end) {
  using iType = typename std::common_type<iType1, iType2>::type;
  return Impl::TeamVectorRangeBoundariesStruct<iType,
                                               Impl::NextSiliconTeamMember>(
      thread, iType(arg_begin), iType(arg_end));
}

KOKKOS_INLINE_FUNCTION
Impl::ThreadSingleStruct<Impl::NextSiliconTeamMember> PerTeam(
    const Impl::NextSiliconTeamMember& thread) {
  return Impl::ThreadSingleStruct<Impl::NextSiliconTeamMember>(thread);
}

KOKKOS_INLINE_FUNCTION
Impl::VectorSingleStruct<Impl::NextSiliconTeamMember> PerThread(
    const Impl::NextSiliconTeamMember& thread) {
  return Impl::VectorSingleStruct<Impl::NextSiliconTeamMember>(thread);
}
}  // namespace Kokkos

namespace Kokkos {

template <class FunctorType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::VectorSingleStruct<Impl::NextSiliconTeamMember>&
    /*single_struct*/,
    const FunctorType& lambda) {
  lambda();
}

template <class FunctorType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::ThreadSingleStruct<Impl::NextSiliconTeamMember>& single_struct,
    const FunctorType& lambda) {
  if (single_struct.team_member.team_rank() == 0) lambda();
}

template <class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::VectorSingleStruct<Impl::NextSiliconTeamMember>&
    /*single_struct*/,
    const FunctorType& lambda, ValueType& val) {
  lambda(val);
}

template <class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION void single(
    const Impl::ThreadSingleStruct<Impl::NextSiliconTeamMember>& single_struct,
    const FunctorType& lambda, ValueType& val) {
  if (single_struct.team_member.team_rank() == 0) {
    lambda(val);
  }
  single_struct.team_member.team_broadcast(val, 0);
}
}  // namespace Kokkos

#endif /* #ifndef KOKKOS_NEXTSILICON_TEAM_HPP */
