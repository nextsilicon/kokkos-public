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

#ifndef KOKKOS_NEXTSILICON_SHARED_ALLOCATION_RECORD_HPP
#define KOKKOS_NEXTSILICON_SHARED_ALLOCATION_RECORD_HPP

#include <NextSilicon/Kokkos_NextSiliconSpace.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

template <>
class Kokkos::Impl::SharedAllocationRecord<Kokkos::Experimental::NextSiliconSpace,
                                           void>
    : public HostInaccessibleSharedAllocationRecordCommon<
          Kokkos::Experimental::NextSiliconSpace> {
 private:
  friend class HostInaccessibleSharedAllocationRecordCommon<
      Kokkos::Experimental::NextSiliconSpace>;
  friend class SharedAllocationRecordCommon<Kokkos::Experimental::NextSiliconSpace>;
  friend Kokkos::Experimental::NextSiliconSpace;

  using base_t = HostInaccessibleSharedAllocationRecordCommon<
      Kokkos::Experimental::NextSiliconSpace>;
  using RecordBase = SharedAllocationRecord<void, void>;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

  /**\brief  Root record for tracked allocations from this NextSiliconSpace
   * instance */
  static RecordBase s_root_record;

  const Kokkos::Experimental::NextSiliconSpace m_space;

 protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() = default;

  template <typename ExecutionSpace>
  SharedAllocationRecord(
      const ExecutionSpace& /*exec_space*/,
      const Kokkos::Experimental::NextSiliconSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate)
      : SharedAllocationRecord(arg_space, arg_label, arg_alloc_size,
                               arg_dealloc) {}

  SharedAllocationRecord(
      const Kokkos::Experimental::NextSilicon& exec_space,
      const Kokkos::Experimental::NextSiliconSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate);

  SharedAllocationRecord(
      const Kokkos::Experimental::NextSiliconSpace& arg_space,
      const std::string& arg_label, const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate);
};

#endif
