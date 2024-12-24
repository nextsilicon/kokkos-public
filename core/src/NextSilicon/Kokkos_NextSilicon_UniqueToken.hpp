// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_NEXTSILICON_UNIQUE_TOKEN_HPP
#define KOKKOS_NEXTSILICON_UNIQUE_TOKEN_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_UniqueToken.hpp>

#ifdef KOKKOS_ENABLE_NEXTSILICON

#include <NextSilicon/Kokkos_NextSiliconSpace.hpp>
#include "NextSilicon/Kokkos_NextSilicon_Abort.hpp"

namespace Kokkos {

namespace Experimental::Impl {
Kokkos::View<uint32_t*, Kokkos::Experimental::NextSiliconSpace> nextsilicon_global_unique_token_locks(bool deallocate = false);
}

namespace Experimental {
// both global and instance Unique Tokens are implemented in the same way
// the global version has one shared static lock array underneath
// but it can't be a static member variable since we need to access it on device
// and we share the implementation with the instance version
template <>
class UniqueToken<NextSilicon, UniqueTokenScope::Global> {
 protected:
  Kokkos::View<uint32_t*, Kokkos::Experimental::NextSiliconSpace> m_locks;

 public:
  using execution_space = NextSilicon;
  using size_type       = int32_t;

  explicit UniqueToken(execution_space const& = NextSilicon())
      : m_locks(Impl::nextsilicon_global_unique_token_locks()) {}

 protected:
  // constructors for the Instance version
  UniqueToken(size_type max_size) {
    m_locks = Kokkos::View<uint32_t*, Experimental::NextSiliconSpace>(
        "Kokkos::UniqueToken::m_locks", max_size);
  }
  UniqueToken(size_type max_size, execution_space const& exec) {
    m_locks = Kokkos::View<uint32_t*, Experimental::NextSiliconSpace>(
        Kokkos::view_alloc(exec, "Kokkos::UniqueToken::m_locks"), max_size);
  }

 public:
  KOKKOS_INLINE_FUNCTION
  size_type size() const noexcept { return m_locks.extent(0); }

 private:
  size_type acquireImpl() const {
    ::Kokkos::Impl::nextsilicon_abort("UniqueToken::acquireImpl not implemented for NextSilicon");
    return 0;
  }

 public:
  KOKKOS_INLINE_FUNCTION
  size_type acquire() const {
    KOKKOS_IF_ON_DEVICE(return acquireImpl();)
    KOKKOS_IF_ON_HOST(return 0;)
  }

  KOKKOS_INLINE_FUNCTION
  void release(size_type idx) const noexcept {
    ::Kokkos::Impl::nextsilicon_abort("UniqueToken::release not implemented for NextSilicon");
  }
};

template <>
class UniqueToken<NextSilicon, UniqueTokenScope::Instance>
    : public UniqueToken<NextSilicon, UniqueTokenScope::Global> {
 public:
  UniqueToken()
      : UniqueToken<NextSilicon, UniqueTokenScope::Global>(
            Kokkos::Experimental::NextSilicon::concurrency()) {}
  explicit UniqueToken(execution_space const& arg)
      : UniqueToken<NextSilicon, UniqueTokenScope::Global>(
            Kokkos::Experimental::NextSilicon::concurrency(), arg) {}
  explicit UniqueToken(size_type max_size)
      : UniqueToken<NextSilicon, UniqueTokenScope::Global>(max_size) {}
  UniqueToken(size_type max_size, execution_space const& arg)
      : UniqueToken<NextSilicon, UniqueTokenScope::Global>(max_size, arg) {}
};

}  // namespace Experimental
}  // namespace Kokkos

#endif  // KOKKOS_ENABLE_NEXTSILICON
#endif  // KOKKOS_NEXTSILICON_UNIQUE_TOKEN_HPP
