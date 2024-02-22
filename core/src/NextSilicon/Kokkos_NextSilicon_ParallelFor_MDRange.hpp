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

namespace Kokkos::Experimental::Impl {

using NextSiliconIterateLeft  = std::integral_constant<Iterate, Iterate::Left>;
using NextSiliconIterateRight = std::integral_constant<Iterate, Iterate::Right>;
template <int N>
using NextSiliconMDRangeBegin =
    decltype(MDRangePolicy<NextSilicon, Rank<N>>::m_lower);
template <int N>
using NextSiliconMDRangeEnd =
    decltype(MDRangePolicy<NextSilicon, Rank<N>>::m_upper);
template <int N>
using NextSiliconMDRangeTile =
    decltype(MDRangePolicy<NextSilicon, Rank<N>>::m_tile);

// a la std::div_t
template <typename T>
struct Div {
  T quot;
  T rem;
};

template <typename T>
KOKKOS_INLINE_FUNCTION Div<T> divmod(T x, T y) {
  return {x / y, x % y};
}

template <typename Iterate, typename Functor, int Dim>
class NextSiliconParallelForMDRangePolicyFunctor {
  NextSiliconParallelForMDRangePolicyFunctor(
      Functor const& functor, NextSiliconMDRangeBegin<Dim> const& begin,
      NextSiliconMDRangeEnd<Dim> const& end);
};

template <typename Functor>
class NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateLeft,
                                                 Functor, 2> {
 public:
  NextSiliconParallelForMDRangePolicyFunctor(
      Functor const& functor, NextSiliconMDRangeBegin<2> const& begin,
      NextSiliconMDRangeEnd<2> const& end)
      : functor_(functor),
        begin1_(begin[1]),
        begin0_(begin[0]),
        ext1_(end[1] - begin[1]),
        ext0_(end[0] - begin[0]) {}

  // a.k.a
  // for (auto i1 = begin1; i1 < end1; ++i1) {
  //   for (auto i0 = begin0; i0 < end0; ++i0) {
  //     functor(i0, i1);
  //   }
  // }
  void operator()(const int i) const {
    auto [i1, i0] = divmod(i, ext0_);
    functor_(i0 + begin0_, i1 + begin1_);
  }

 private:
  Functor functor_;
  int begin1_, begin0_;
  int ext1_, ext0_;
};

template <typename Functor>
class NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateRight,
                                                 Functor, 2> {
 public:
  NextSiliconParallelForMDRangePolicyFunctor(
      Functor const& functor, NextSiliconMDRangeBegin<2> const& begin,
      NextSiliconMDRangeEnd<2> const& end)
      : functor_(functor),
        begin1_(begin[1]),
        begin0_(begin[0]),
        ext1_(end[1] - begin[1]),
        ext0_(end[0] - begin[0]) {}

  // a.k.a
  // for (auto i0 = begin0; i0 < end0; ++i0) {
  //   for (auto i1 = begin1; i1 < end1; ++i1) {
  //     functor(i0, i1);
  //   }
  // }
  void operator()(const int i) const {
    auto [i0, i1] = divmod(i, ext1_);
    functor_(i0 + begin0_, i1 + begin1_);
  }

 private:
  Functor functor_;
  int begin1_;
  int begin0_;
  int ext1_;
  int ext0_;
};

template <typename Functor>
class NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateLeft,
                                                 Functor, 3> {
 public:
  NextSiliconParallelForMDRangePolicyFunctor(
      Functor const& functor, NextSiliconMDRangeBegin<3> const& begin,
      NextSiliconMDRangeEnd<3> const& end)
      : functor_(functor),
        begin2_(begin[2]),
        begin1_(begin[1]),
        begin0_(begin[0]),
        ext2_(end[2] - begin[2]),
        ext1_(end[1] - begin[1]),
        ext0_(end[0] - begin[0]) {}

  // a.k.a
  // for (auto i2 = begin2; i2 < end2; ++i2) {
  //   for (auto i1 = begin1; i1 < end1; ++i1) {
  //     for (auto i0 = begin0; i0 < end0; ++i0) {
  //       functor(i0, i1, i2);
  //     }
  //   }
  // }
  void operator()(const int i) const {
    auto [_i1, i0] = divmod(i, ext0_);
    auto [i2, i1]  = divmod(_i1, ext1_);
    functor_(i0 + begin0_, i1 + begin1_, i2 + begin2_);
  }

 private:
  Functor functor_;
  int begin2_, begin1_, begin0_;
  int ext2_, ext1_, ext0_;
};

template <typename Functor>
class NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateRight,
                                                 Functor, 3> {
 public:
  NextSiliconParallelForMDRangePolicyFunctor(
      Functor const& functor, NextSiliconMDRangeBegin<3> const& begin,
      NextSiliconMDRangeEnd<3> const& end)
      : functor_(functor),
        begin2_(begin[2]),
        begin1_(begin[1]),
        begin0_(begin[0]),
        ext2_(end[2] - begin[2]),
        ext1_(end[1] - begin[1]),
        ext0_(end[0] - begin[0]) {}

  // a.k.a
  // for (auto i0 = begin0; i0 < end0; ++i0) {
  //   for (auto i1 = begin1; i1 < end1; ++i1) {
  //     for (auto i2 = begin2; i2 < end2; ++i2) {
  //       functor(i0, i1, i2);
  //     }
  //   }
  // }
  void operator()(const int i) const {
    auto [_i1, i2] = divmod(i, ext2_);
    auto [i0, i1]  = divmod(_i1, ext1_);
    functor_(i0 + begin0_, i1 + begin1_, i2 + begin2_);
  }

 private:
  Functor functor_;
  int begin2_, begin1_, begin0_;
  int ext2_, ext1_, ext0_;
};

template <typename Functor>
class NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateLeft,
                                                 Functor, 4> {
 public:
  NextSiliconParallelForMDRangePolicyFunctor(
      Functor const& functor, NextSiliconMDRangeBegin<4> const& begin,
      NextSiliconMDRangeEnd<4> const& end)
      : functor_(functor),
        begin3_(begin[3]),
        begin2_(begin[2]),
        begin1_(begin[1]),
        begin0_(begin[0]),
        ext3_(end[3] - begin[3]),
        ext2_(end[2] - begin[2]),
        ext1_(end[1] - begin[1]),
        ext0_(end[0] - begin[0]) {}

  // a.k.a
  // for (auto i3 = begin3; i3 < end3; ++i3) {
  //   for (auto i2 = begin2; i2 < end2; ++i2) {
  //     for (auto i1 = begin1; i1 < end1; ++i1) {
  //       for (auto i0 = begin0; i0 < end0; ++i0) {
  //         functor(i0, i1, i2, i3);
  //       }
  //     }
  //   }
  // }
  void operator()(const int i) const {
    auto [_i1, i0] = divmod(i, ext0_);
    auto [_i2, i1] = divmod(_i1, ext1_);
    auto [i3, i2]  = divmod(_i2, ext2_);
    functor_(i0 + begin0_, i1 + begin1_, i2 + begin2_, i3 + begin3_);
  }

 private:
  Functor functor_;
  int begin3_, begin2_, begin1_, begin0_;
  int ext3_, ext2_, ext1_, ext0_;
};

template <typename Functor>
class NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateRight,
                                                 Functor, 4> {
 public:
  NextSiliconParallelForMDRangePolicyFunctor(
      Functor const& functor, NextSiliconMDRangeBegin<4> const& begin,
      NextSiliconMDRangeEnd<4> const& end)
      : functor_(functor),
        begin3_(begin[3]),
        begin2_(begin[2]),
        begin1_(begin[1]),
        begin0_(begin[0]),
        ext3_(end[3] - begin[3]),
        ext2_(end[2] - begin[2]),
        ext1_(end[1] - begin[1]),
        ext0_(end[0] - begin[0]) {}

  // a.k.a
  // for (auto i0 = begin0; i0 < end0; ++i0) {
  //   for (auto i1 = begin1; i1 < end1; ++i1) {
  //     for (auto i2 = begin2; i2 < end2; ++i2) {
  //       for (auto i3 = begin3; i3 < end3; ++i3) {
  //         functor(i0, i1, i2, i3);
  //       }
  //     }
  //   }
  // }
  void operator()(const int i) const {
    auto [_i2, i3] = divmod(i, ext3_);
    auto [_i1, i2] = divmod(_i2, ext2_);
    auto [i0, i1]  = divmod(_i1, ext1_);
    functor_(i0 + begin0_, i1 + begin1_, i2 + begin2_, i3 + begin3_);
  }

 private:
  Functor functor_;
  int begin3_, begin2_, begin1_, begin0_;
  int ext3_, ext2_, ext1_, ext0_;
};

template <typename Functor>
class NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateLeft,
                                                 Functor, 5> {
 public:
  NextSiliconParallelForMDRangePolicyFunctor(
      Functor const& functor, NextSiliconMDRangeBegin<5> const& begin,
      NextSiliconMDRangeEnd<5> const& end)
      : functor_(functor),
        begin4_(begin[4]),
        begin3_(begin[3]),
        begin2_(begin[2]),
        begin1_(begin[1]),
        begin0_(begin[0]),
        ext4_(end[4] - begin[4]),
        ext3_(end[3] - begin[3]),
        ext2_(end[2] - begin[2]),
        ext1_(end[1] - begin[1]),
        ext0_(end[0] - begin[0]) {}

  // a.k.a
  // for (auto i4 = begin4; i4 < end4; ++i4) {
  //   for (auto i3 = begin3; i3 < end3; ++i3) {
  //     for (auto i2 = begin2; i2 < end2; ++i2) {
  //       for (auto i1 = begin1; i1 < end1; ++i1) {
  //         for (auto i0 = begin0; i0 < end0; ++i0) {
  //           functor(i0, i1, i2, i3, i4);
  //         }
  //       }
  //     }
  //   }
  // }
  void operator()(const int i) const {
    auto [_i1, i0] = divmod(i, ext0_);
    auto [_i2, i1] = divmod(_i1, ext1_);
    auto [_i3, i2] = divmod(_i2, ext2_);
    auto [i4, i3]  = divmod(_i3, ext3_);
    functor_(i0 + begin0_, i1 + begin1_, i2 + begin2_, i3 + begin3_,
             i4 + begin4_);
  }

 private:
  Functor functor_;
  int begin4_, begin3_, begin2_, begin1_, begin0_;
  int ext4_, ext3_, ext2_, ext1_, ext0_;
};

template <typename Functor>
class NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateRight,
                                                 Functor, 5> {
 public:
  NextSiliconParallelForMDRangePolicyFunctor(
      Functor const& functor, NextSiliconMDRangeBegin<5> const& begin,
      NextSiliconMDRangeEnd<5> const& end)
      : functor_(functor),
        begin4_(begin[4]),
        begin3_(begin[3]),
        begin2_(begin[2]),
        begin1_(begin[1]),
        begin0_(begin[0]),
        ext4_(end[4] - begin[4]),
        ext3_(end[3] - begin[3]),
        ext2_(end[2] - begin[2]),
        ext1_(end[1] - begin[1]),
        ext0_(end[0] - begin[0]) {}

  // a.k.a
  // for (auto i0 = begin0; i0 < end0; ++i0) {
  //   for (auto i1 = begin1; i1 < end1; ++i1) {
  //     for (auto i2 = begin2; i2 < end2; ++i2) {
  //       for (auto i3 = begin3; i3 < end3; ++i3) {
  //         for (auto i4 = begin4; i4 < end4; ++i4) {
  //           functor(i0, i1, i2, i3, i4);
  //         }
  //       }
  //     }
  //   }
  // }
  void operator()(const int i) const {
    auto [_i3, i4] = divmod(i, ext4_);
    auto [_i2, i3] = divmod(_i3, ext3_);
    auto [_i1, i2] = divmod(_i2, ext2_);
    auto [i0, i1]  = divmod(_i1, ext1_);
    functor_(i0 + begin0_, i1 + begin1_, i2 + begin2_, i3 + begin3_,
             i4 + begin4_);
  }

 private:
  Functor functor_;
  int begin4_, begin3_, begin2_, begin1_, begin0_;
  int ext4_, ext3_, ext2_, ext1_, ext0_;
};

template <typename Functor>
class NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateLeft,
                                                 Functor, 6> {
 public:
  NextSiliconParallelForMDRangePolicyFunctor(
      Functor const& functor, NextSiliconMDRangeBegin<5> const& begin,
      NextSiliconMDRangeEnd<5> const& end)
      : functor_(functor),
        begin4_(begin[4]),
        begin3_(begin[3]),
        begin2_(begin[2]),
        begin1_(begin[1]),
        begin0_(begin[0]),
        ext4_(end[4] - begin[4]),
        ext3_(end[3] - begin[3]),
        ext2_(end[2] - begin[2]),
        ext1_(end[1] - begin[1]),
        ext0_(end[0] - begin[0]) {}

  // a.k.a
  // for (auto i5 = begin5; i5 < end5; ++i5) {
  //   for (auto i4 = begin4; i4 < end4; ++i4) {
  //     for (auto i3 = begin3; i3 < end3; ++i3) {
  //       for (auto i2 = begin2; i2 < end2; ++i2) {
  //         for (auto i1 = begin1; i1 < end1; ++i1) {
  //           for (auto i0 = begin0; i0 < end0; ++i0) {
  //             functor(i0, i1, i2, i3, i4, i5);
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  void operator()(const int i) const {
    auto [_i1, i0] = divmod(i, ext0_);
    auto [_i2, i1] = divmod(_i1, ext1_);
    auto [_i3, i2] = divmod(_i2, ext2_);
    auto [_i4, i3] = divmod(_i3, ext3_);
    auto [i5, i4]  = divmod(_i4, ext4_);
    functor_(i0 + begin0_, i1 + begin1_, i2 + begin2_, i3 + begin3_,
             i4 + begin4_, i5 + begin5_);
  }

 private:
  Functor functor_;
  int begin5_, begin4_, begin3_, begin2_, begin1_, begin0_;
  int ext5_, ext4_, ext3_, ext2_, ext1_, ext0_;
};

template <typename Functor>
class NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateRight,
                                                 Functor, 6> {
 public:
  NextSiliconParallelForMDRangePolicyFunctor(
      Functor const& functor, NextSiliconMDRangeBegin<5> const& begin,
      NextSiliconMDRangeEnd<5> const& end)
      : functor_(functor),
        begin4_(begin[4]),
        begin3_(begin[3]),
        begin2_(begin[2]),
        begin1_(begin[1]),
        begin0_(begin[0]),
        ext4_(end[4] - begin[4]),
        ext3_(end[3] - begin[3]),
        ext2_(end[2] - begin[2]),
        ext1_(end[1] - begin[1]),
        ext0_(end[0] - begin[0]) {}

  // a.k.a
  // for (auto i0 = begin0; i0 < end0; ++i0) {
  //   for (auto i1 = begin1; i1 < end1; ++i1) {
  //     for (auto i2 = begin2; i2 < end2; ++i2) {
  //       for (auto i3 = begin3; i3 < end3; ++i3) {
  //         for (auto i4 = begin4; i4 < end4; ++i4) {
  //           for (auto i5 = begin5; i5 < end5; ++i5) {
  //             functor(i0, i1, i2, i3, i4, i5);
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  void operator()(const int i) const {
    auto [_i4, i5] = divmod(i, ext5_);
    auto [_i3, i4] = divmod(_i4, ext4_);
    auto [_i2, i3] = divmod(_i3, ext3_);
    auto [_i1, i2] = divmod(_i2, ext2_);
    auto [i0, i1]  = divmod(_i1, ext1_);
    functor_(i0 + begin0_, i1 + begin1_, i2 + begin2_, i3 + begin3_,
             i4 + begin4_, i5 + begin5_);
  }

 private:
  Functor functor_;
  int begin5_, begin4_, begin3_, begin2_, begin1_, begin0_;
  int ext5_, ext4_, ext3_, ext2_, ext1_, ext0_;
};

template <class Functor>
void NextSiliconParallelForMDRangePolicy(
    NextSiliconIterateLeft, Functor const& functor,
    NextSiliconMDRangeBegin<2> const& begin,
    NextSiliconMDRangeEnd<2> const& end) {
  const auto flat = (end[0] - begin[0]) * (end[1] - begin[1]);
  Kokkos::parallel_for(
      flat, NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateLeft,
                                                       Functor, 2>(functor,
                                                                   begin, end));
}

template <class Functor>
void NextSiliconParallelForMDRangePolicy(
    NextSiliconIterateRight, Functor const& functor,
    NextSiliconMDRangeBegin<2> const& begin,
    NextSiliconMDRangeEnd<2> const& end) {
  const auto flat = (end[0] - begin[0]) * (end[1] - begin[1]);
  Kokkos::parallel_for(
      flat, NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateRight,
                                                       Functor, 2>(functor,
                                                                   begin, end));
}

template <class Functor>
void NextSiliconParallelForMDRangePolicy(
    NextSiliconIterateLeft, Functor const& functor,
    NextSiliconMDRangeBegin<3> const& begin,
    NextSiliconMDRangeEnd<3> const& end) {
  const auto flat =
      (end[0] - begin[0]) * (end[1] - begin[1]) * (end[2] - begin[2]);
  Kokkos::parallel_for(
      flat, NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateLeft,
                                                       Functor, 3>(functor,
                                                                   begin, end));
}

template <class Functor>
void NextSiliconParallelForMDRangePolicy(
    NextSiliconIterateRight, Functor const& functor,
    NextSiliconMDRangeBegin<3> const& begin,
    NextSiliconMDRangeEnd<3> const& end) {
  const auto flat =
      (end[0] - begin[0]) * (end[1] - begin[1]) * (end[2] - begin[2]);
  Kokkos::parallel_for(
      flat, NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateLeft,
                                                       Functor, 3>(functor,
                                                                   begin, end));
}

template <class Functor>
void NextSiliconParallelForMDRangePolicy(
    NextSiliconIterateLeft, Functor const& functor,
    NextSiliconMDRangeBegin<4> const& begin,
    NextSiliconMDRangeEnd<4> const& end) {
  const auto flat = (end[0] - begin[0]) * (end[1] - begin[1]) *
                    (end[2] - begin[2]) * (end[3] - begin[3]);
  Kokkos::parallel_for(
      flat, NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateLeft,
                                                       Functor, 4>(functor,
                                                                   begin, end));
}

template <class Functor>
void NextSiliconParallelForMDRangePolicy(
    NextSiliconIterateRight, Functor const& functor,
    NextSiliconMDRangeBegin<4> const& begin,
    NextSiliconMDRangeEnd<4> const& end) {
  const auto flat = (end[0] - begin[0]) * (end[1] - begin[1]) *
                    (end[2] - begin[2]) * (end[3] - begin[3]);
  Kokkos::parallel_for(
      flat, NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateRight,
                                                       Functor, 4>(functor,
                                                                   begin, end));
}

template <class Functor>
void NextSiliconParallelForMDRangePolicy(
    NextSiliconIterateLeft, Functor const& functor,
    NextSiliconMDRangeBegin<5> const& begin,
    NextSiliconMDRangeEnd<5> const& end) {
  const auto flat = (end[0] - begin[0]) * (end[1] - begin[1]) *
                    (end[2] - begin[2]) * (end[3] - begin[3]) *
                    (end[4] - begin[4]);
  Kokkos::parallel_for(
      flat, NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateLeft,
                                                       Functor, 5>(functor,
                                                                   begin, end));
}

template <class Functor>
void NextSiliconParallelForMDRangePolicy(
    NextSiliconIterateRight, Functor const& functor,
    NextSiliconMDRangeBegin<5> const& begin,
    NextSiliconMDRangeEnd<5> const& end) {
  const auto flat = (end[0] - begin[0]) * (end[1] - begin[1]) *
                    (end[2] - begin[2]) * (end[3] - begin[3]) *
                    (end[4] - begin[4]);
  Kokkos::parallel_for(
      flat, NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateRight,
                                                       Functor, 5>(functor,
                                                                   begin, end));
}

template <class Functor>
void NextSiliconParallelForMDRangePolicy(
    NextSiliconIterateLeft, Functor const& functor,
    NextSiliconMDRangeBegin<6> const& begin,
    NextSiliconMDRangeEnd<6> const& end) {
  const auto flat = (end[0] - begin[0]) * (end[1] - begin[1]) *
                    (end[2] - begin[2]) * (end[3] - begin[3]) *
                    (end[4] - begin[4]) * (end[5] - begin[5]);
  Kokkos::parallel_for(
      flat, NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateLeft,
                                                       Functor, 6>(functor,
                                                                   begin, end));
}

template <class Functor>
void NextSiliconParallelForMDRangePolicy(
    NextSiliconIterateRight, Functor const& functor,
    NextSiliconMDRangeBegin<6> const& begin,
    NextSiliconMDRangeEnd<6> const& end) {
  const auto flat = (end[0] - begin[0]) * (end[1] - begin[1]) *
                    (end[2] - begin[2]) * (end[3] - begin[3]) *
                    (end[4] - begin[4]) * (end[5] - begin[5]);
  Kokkos::parallel_for(
      flat, NextSiliconParallelForMDRangePolicyFunctor<NextSiliconIterateRight,
                                                       Functor, 6>(functor,
                                                                   begin, end));
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
      : m_functor(functor), m_policy(policy) {}

  void execute() const {
    static_assert(1 < Policy::rank && Policy::rank < 7);
    static_assert(Policy::inner_direction == Iterate::Left ||
                  Policy::inner_direction == Iterate::Right);
    constexpr int rank = Policy::rank;
    for (int i = 0; i < rank; ++i) {
      if (m_policy.m_lower[i] >= m_policy.m_upper[i]) {
        return;
      }
    }

    // FIXME_NEXTSILICON: throw away requested tiling
    Kokkos::Experimental::Impl::NextSiliconParallelForMDRangePolicy(
        std::integral_constant<Iterate, Policy::inner_direction>(), m_functor,
        m_policy.m_lower, m_policy.m_upper);
  }
};

#endif  // KOKKOS_NEXTSILICON_PARALLELFOR_MDRANGE_HPP
