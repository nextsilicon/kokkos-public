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

#include <TestNextSilicon_Category.hpp>

#include <nsapi/memory.h>

/*! \brief Make sure some symbol from libnsapi can be linked */

namespace Test {
TEST(nextsilicon, nsapi_link) {
  // doesn't really do anything and don't care if it does
  nsapi_mem_advise(nullptr, 0, 0, nullptr);
}
}  // namespace Test
