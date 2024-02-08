#ifndef KOKKOS_NEXTSILICON_ABORT_HPP
#define KOKKOS_NEXTSILICON_ABORT_HPP

namespace Kokkos {
namespace Impl {

[[noreturn]] inline void nextsilicon_abort(char const* msg) {
#ifdef NDEBUG
  // FIXME_NEXTSILICON Add support for printf from inside NS parallel regions.
  Kokkos::printf("Aborting with message %s.\n", msg);
#endif
  // FIXME_NEXTSILICON Add nsapi to abort application from device
  while (true)
    ;
}

}  // namespace Impl
}  // namespace Kokkos

#endif
