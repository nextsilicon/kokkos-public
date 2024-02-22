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

#ifndef KOKKOS_NEXTSILICON_MDRANGE_POLICY_HPP
#define KOKKOS_NEXTSILICON_MDRANGE_POLICY_HPP

#include <KokkosExp_MDRangePolicy.hpp>

template <>
struct Kokkos::default_outer_direction<Kokkos::Experimental::NextSilicon> {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Left;
};

template <>
struct Kokkos::default_inner_direction<Kokkos::Experimental::NextSilicon> {
  using type                     = Iterate;
  static constexpr Iterate value = Iterate::Left;
};

#endif