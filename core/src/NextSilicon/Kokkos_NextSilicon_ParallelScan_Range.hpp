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

#ifndef KOKKOS_NEXTSILICON_PARALLEL_SCAN_RANGE_HPP
#define KOKKOS_NEXTSILICON_PARALLEL_SCAN_RANGE_HPP

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSilicon_FunctorAdapter.hpp>
#include <NextSilicon/Kokkos_NextSilicon_Macros.hpp>
#include <Kokkos_Parallel.hpp>

namespace Kokkos::Impl {

// FIXME_NEXTSILICON: This is a single-threaded host CPU implementation of
// Kokkos::parallel_scan. This only works with Next Silicon
// transparently-migratable memory, and is intended as a temporary (but correct)
// implementation of parallel_scan so applications that use it can be compiled
// and run against the new backend with bad performance.
template <class Functor, class GivenValueType, class... Traits>
class ParallelScanNextSiliconHostPlaceholder {
 protected:
  using Policy = Kokkos::RangePolicy<Traits...>;
  using Analysis =
      Kokkos::Impl::FunctorAnalysis<Kokkos::Impl::FunctorPatternInterface::SCAN,
                                    Policy, Functor, GivenValueType>;
  using PointerType = typename Analysis::pointer_type;
  using ValueType   = typename Analysis::value_type;
  using MemberType  = typename Policy::member_type;
  using IndexType   = typename Policy::index_type;
  Functor m_functor;
  Policy m_policy;
  ValueType* m_result_ptr;

 public:
  ParallelScanNextSiliconHostPlaceholder(Functor const& arg_functor,
                                         Policy const& arg_policy,
                                         ValueType* arg_result_ptr)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_result_ptr(arg_result_ptr) {}

  void RangePolicy(const IndexType begin, const IndexType end,
                   IndexType /*chunk_size*/) const {
    const Kokkos::Experimental::Impl::FunctorAdapter<Functor, Policy> functor(
        m_functor);
    typename Analysis::Reducer final_reducer(m_functor);

    ValueType update;
    final_reducer.init(&update);
    for (IndexType i = begin; i < end; ++i) {
      functor(i, update, /*final*/ true);
    }
    if (m_result_ptr) {
      *m_result_ptr = update;
    }
  }

  void execute() const {
    const IndexType begin = m_policy.begin();
    const IndexType end   = m_policy.end();
    IndexType chunk_size  = m_policy.chunk_size();
    RangePolicy(begin, end, chunk_size);
  }
};

}  // namespace Kokkos::Impl

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

template <class Functor, class... Traits>
class Kokkos::Impl::ParallelScan<Functor, Kokkos::RangePolicy<Traits...>,
                                 Kokkos::Experimental::NextSilicon>
    : public ParallelScanNextSiliconHostPlaceholder<Functor, void, Traits...> {
  using base_t =
      ParallelScanNextSiliconHostPlaceholder<Functor, void, Traits...>;
  using IndexType = typename base_t::IndexType;

 public:
  void execute() const {
    const IndexType begin = base_t::m_policy.begin();
    const IndexType end   = base_t::m_policy.end();
    IndexType chunk_size  = base_t::m_policy.chunk_size();

    base_t::RangePolicy(begin, end, chunk_size);
  }

  ParallelScan(const Functor& arg_functor,
               const typename base_t::Policy& arg_policy)
      : base_t(arg_functor, arg_policy, nullptr) {}
};

template <class FunctorType, class ReturnType, class... Traits>
class Kokkos::Impl::ParallelScanWithTotal<
    FunctorType, Kokkos::RangePolicy<Traits...>, ReturnType,
    Kokkos::Experimental::NextSilicon>
    : public ParallelScanNextSiliconHostPlaceholder<FunctorType, ReturnType,
                                                    Traits...> {
  using base_t = ParallelScanNextSiliconHostPlaceholder<FunctorType, ReturnType,
                                                        Traits...>;
  using IndexType = typename base_t::IndexType;

 public:
  void execute() const {
    const IndexType begin = base_t::m_policy.begin();
    const IndexType end   = base_t::m_policy.end();
    IndexType chunk_size  = base_t::m_policy.chunk_size();

    if (end <= begin) {
      if (base_t::m_result_ptr != nullptr) {
        *base_t::m_result_ptr = 0;
      }
      return;
    }

    base_t::RangePolicy(begin, end, chunk_size);
  }

  template <class ViewType>
  ParallelScanWithTotal(const FunctorType& arg_functor,
                        const typename base_t::Policy& arg_policy,
                        const ViewType& arg_result_view)
      : base_t(arg_functor, arg_policy, arg_result_view.data()) {}
};

#endif
