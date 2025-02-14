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

#ifndef KOKKOS_NEXTSILICON_PARALLELFOR_RANGE_HPP
#define KOKKOS_NEXTSILICON_PARALLELFOR_RANGE_HPP

#include <nsapi/intrinsics.h>
#if !defined(KOKKOS_ENABLE_IMPL_NSAPI_UNAVAIL)
#include <nsapi/parallelism.h>
#include <nsapi/parallelism_internal.hpp>
#endif

#include <NextSilicon/Kokkos_NextSilicon.hpp>
#include <NextSilicon/Kokkos_NextSilicon_FunctorAdapter.hpp>
#include <Kokkos_Parallel.hpp>

template <class Functor, class... Traits>
class Kokkos::Impl::ParallelFor<Functor, Kokkos::RangePolicy<Traits...>,
                                Kokkos::Experimental::NextSilicon> {
  using Policy    = Kokkos::RangePolicy<Traits...>;
  using WorkTag   = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member    = typename Policy::member_type;
  using IndexType = typename Policy::index_type;

  using FunctorWrapper =
      Kokkos::Experimental::Impl::FunctorAdapter<Functor, Policy>;

  FunctorWrapper m_functor;
  Policy m_policy;

 public:
  ParallelFor(Functor const& functor, Policy const& policy)
      : m_functor(functor), m_policy(policy) {}

  void execute() const {
    // Clone the driver before going into the handoff function.
    // This prevents the stack from getting migrated into device.
    auto cloned_driver = std::make_unique<
        Kokkos::Impl::ParallelFor<Functor, Kokkos::RangePolicy<Traits...>,
                                  Kokkos::Experimental::NextSilicon>>(*this);
    cloned_driver->execute_internal();
  }

 private:
  __attribute__((noinline)) void execute_internal() const {
    // FIXME_NEXTSILICON: Add a dynamic schedule policy check if `typename
    // Policy::schedule_type::type` is Kokkos::Static or Kokkos::Dynamic
    const IndexType begin = m_policy.begin();
    const IndexType end   = m_policy.end();

    if (end <= begin) return;

#ifdef KOKKOS_ENABLE_IMPL_NSAPI_UNAVAIL

#pragma omp parallel for
    for (IndexType i = begin; i < end; ++i) {
      m_functor(i);
    }

#else

    const IndexType chunk_size = m_policy.chunk_size();
    nsapi::experimental::parallel_for(begin, end, chunk_size, parallel_function,
                                      &m_functor);

#endif
  }

#ifndef KOKKOS_ENABLE_IMPL_NSAPI_UNAVAIL

 private:
  /// Executes a single parallel iteration `index` for `functor`.
  static void parallel_function(IndexType index,
                                FunctorWrapper const* __restrict functor) {
    // Communicate to the compiler that the functor is a immutable and thread
    // invariant for the duration of the microtask.
    Kokkos::Experimental::Impl::
        __ns_immutable_thread_invariant_parameter_struct(functor);
    // Invokes the functor itself. Expected to inline (if compiler visible) the
    // functor body while passing the extra this pointer.
    (*functor)(index);
  }

#endif
};

#endif  // KOKKOS_NEXTSILICON_PARALLELFOR_RANGE_HPP
