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

#ifndef KOKKOS_NEXTSILICON_PARALLELFOR_MDRANGE_HPP
#define KOKKOS_NEXTSILICON_PARALLELFOR_MDRANGE_HPP

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSilicon_FunctorAdapter.hpp>
#include <NextSilicon/Kokkos_NextSilicon_MDRangePolicy.hpp>
#include <Kokkos_Parallel.hpp>
#include <impl/Kokkos_CheckedIntegerOps.hpp>

#include <type_traits>
#include <utility>

namespace Kokkos::Experimental::Impl {

template <int N>
using NextSiliconMDRangeBegin =
    decltype(MDRangePolicy<NextSilicon, Rank<N>>::m_lower);
template <int N>
using NextSiliconMDRangeEnd =
    decltype(MDRangePolicy<NextSilicon, Rank<N>>::m_upper);
template <int N>
using NextSiliconMDRangeTile =
    decltype(MDRangePolicy<NextSilicon, Rank<N>>::m_tile);

template <typename Direction, typename Functor, int Dim>
class NextSiliconParallelForMDRangePolicyFunctor {
 public:
  using IndexType = typename NextSiliconMDRangeEnd<Dim>::value_type;

  NextSiliconParallelForMDRangePolicyFunctor(
      Functor const& functor, NextSiliconMDRangeBegin<Dim> const& begin,
      NextSiliconMDRangeEnd<Dim> const& end)
      : functor_(functor), begin_(begin) {
    for (int i = 0; i < Dim; ++i) {
      ext_[i] = end[i] - begin[i];
    }
  }

  template <typename FlatIndexType>
  KOKKOS_INLINE_FUNCTION void operator()(const FlatIndexType i) const {
    IndexType x[Dim];
    map(i, x);
    call_functor(std::make_index_sequence<Dim>() /* 0, 1, ..., Dim */, x);
  }

 private:
  // call the functor on a Dim-dimensional index
  template <size_t... I>
  KOKKOS_INLINE_FUNCTION void call_functor(std::index_sequence<I...>,
                                           const IndexType (&x)[Dim]) const {
    functor_(x[I]...);
  }

  template <typename FlatIndexType>
  KOKKOS_INLINE_FUNCTION void map(FlatIndexType i, IndexType (&x)[Dim]) const {
    if constexpr (std::is_same_v<Direction, std::integral_constant<
                                                Iterate, Iterate::Left>>) {
      // like
      // for (auto i2 = begin2; i2 < end2; ++i2) {
      //   for (auto i1 = begin1; i1 < end1; ++i1) {
      //     for (auto i0 = begin0; i0 < end0; ++i0) {
      //       ...
      //     }
      //   }
      // }
      for (int j = 0; j < Dim; ++j) {
        auto r = std::div(i, ext_[j]);
        x[j]   = begin_[j] + r.rem;
        i      = r.quot;
      }
    } else if constexpr (std::is_same_v<
                             Direction,
                             std::integral_constant<Iterate, Iterate::Right>>) {
      // like
      // for (auto i0 = begin0; i0 < end0; ++i0) {
      //   for (auto i1 = begin1; i1 < end1; ++i1) {
      //     for (auto i2 = begin2; i2 < end2; ++i2) {
      //       ...
      //     }
      //   }
      // }
      for (int j = Dim - 1; j >= 0; --j) {
        auto r = std::div(i, ext_[j]);
        x[j]   = begin_[j] + r.rem;
        i      = r.quot;
      }
    } else {
      static_assert(std::is_void_v<Functor>,
                    "Expected left or right MDRange iteration");
    }
  }

  Functor functor_;
  NextSiliconMDRangeBegin<Dim> begin_;
  NextSiliconMDRangeEnd<Dim> ext_;  // end - begin
};

template <typename Direction, int Dim, class Functor>
void NextSiliconParallelForMDRangePolicy(
    Functor const& functor, NextSiliconMDRangeBegin<Dim> const& begin,
    NextSiliconMDRangeEnd<Dim> const& end) {
  using FlatIndexType = uint64_t;

  FlatIndexType flat = end[0] - begin[0];
  for (int i = 1; i < Dim; ++i) {
    const FlatIndexType factor = end[i] - begin[i];
    flat = Kokkos::Impl::multiply_overflow_abort(flat, factor);
  }
  Kokkos::parallel_for(
      flat, NextSiliconParallelForMDRangePolicyFunctor<Direction, Functor, Dim>(
                functor, begin, end));
}

}  // namespace Kokkos::Experimental::Impl

template <class Functor, class... Traits>
class Kokkos::Impl::ParallelFor<Functor, Kokkos::MDRangePolicy<Traits...>,
                                Kokkos::Experimental::NextSilicon> {
  using Policy = MDRangePolicy<Traits...>;
  Kokkos::Experimental::Impl::FunctorAdapter<Functor, Policy> m_functor;
  Policy m_policy;

 public:
  ParallelFor(Functor const& functor, Policy const& policy)
      : m_functor(functor), m_policy(policy) {
    static_assert(1 < Policy::rank && Policy::rank < 7);
    static_assert(Policy::inner_direction == Iterate::Left ||
                  Policy::inner_direction == Iterate::Right);
  }

  void execute() const {
    constexpr int rank = Policy::rank;
    for (int i = 0; i < rank; ++i) {
      if (m_policy.m_lower[i] >= m_policy.m_upper[i]) {
        return;
      }
    }

    // FIXME_NEXTSILICON: throw away requested tiling
    using Direction = std::integral_constant<Iterate, Policy::inner_direction>;
    Kokkos::Experimental::Impl::NextSiliconParallelForMDRangePolicy<Direction,
                                                                    rank>(
        m_functor, m_policy.m_lower, m_policy.m_upper);
  }
};

#endif  // KOKKOS_NEXTSILICON_PARALLELFOR_MDRANGE_HPP
