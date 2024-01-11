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

#ifndef KOKKOS_NEXTSILICON_INSTANCE_HPP
#define KOKKOS_NEXTSILICON_INSTANCE_HPP

#include <cstdint>
#include <iosfwd>
#include <string>

namespace Kokkos::Experimental::Impl {

struct NextSiliconTraits {
  /* Alignment requirement for fill operations. */
  static constexpr size_t FillAlignmend   = 0x40;
  static constexpr size_t BmtUseThreshold = 0x200 /* 512 bytes */;
};

class NextSiliconInternal {
  bool m_is_initialized = false;

  NextSiliconInternal()                                      = default;
  NextSiliconInternal(const NextSiliconInternal&)            = delete;
  NextSiliconInternal& operator=(const NextSiliconInternal&) = delete;

 public:
  const static uint32_t m_ns_device_id = 0;
  static NextSiliconInternal* singleton();

  void initialize();
  void finalize();
  bool is_initialized() const;

  void print_configuration(std::ostream& os, bool verbose = false) const;

  void fence(std::string const& name) const;

  uint32_t instance_id() const noexcept;
};

}  // namespace Kokkos::Experimental::Impl

#endif