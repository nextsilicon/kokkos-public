# as of nextcxx version 0.4.7 (check inside nextcxx)
#   $ cat $(which nextcxx)
#
# nextcxx searches the following include paths automatically:
#   $ nextcxx -v main.cpp
#   #include <...> search starts here:
#   /opt/nextsilicon/bin/../sysroot/usr/include/c++/v1
#   /opt/nextsilicon/bin/../sysroot/usr/include
#   /usr/local/include
#   /opt/nextsilicon/llvm/lib/clang/12.0.1/include
#
#   /opt/nextsilicon/bin/../sysroot/usr/include has a copy of the
#   nsapi headers, so we do not need to search for them here

KOKKOS_FIND_IMPORTED(NSAPI 
LIBRARY nsapi
LIBRARY_PATHS /opt/nextsilicon
)
