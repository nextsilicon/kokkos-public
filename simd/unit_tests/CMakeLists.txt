kokkos_include_directories(${KOKKOS_SOURCE_DIR}/simd/unit_tests/include)

if(NOT (Kokkos_ENABLE_CUDA AND WIN32) and NOT Kokkos_ENABLE_NEXTSILICON) # FIXME_NEXTSILICON
  kokkos_add_executable_and_test(UnitTest_SIMD SOURCES UnitTestMain.cpp TestSIMD.cpp)
endif()
