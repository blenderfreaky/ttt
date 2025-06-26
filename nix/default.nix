{pkgs}: rec {
  # mkl = pkgs.callPackage ./mkl.nix {};
  kokkos = pkgs.callPackage ./kokkos.nix {};
  kokkos-kernels = pkgs.callPackage ./kokkos-kernels.nix {inherit kokkos;};
  kokkos-tools = pkgs.callPackage ./kokkos-tools.nix {inherit kokkos;};
  base = pkgs.callPackage ./base.nix {inherit libffi_3_2_1 hwloc_1 opencl-clang_14 libgdbm-legacy;};
  hpc = pkgs.callPackage ./hpc.nix {inherit libffi_3_2_1 hwloc_1 opencl-clang_14 libgdbm-legacy;};
  libffi_3_2_1 = pkgs.callPackage ./libffi_3_2_1.nix {};
  hwloc_1 = pkgs.callPackage ./hwloc_1.nix {};
  opencl-clang_14 = pkgs.callPackage ./opencl-clang_14.nix {};
  libgdbm-legacy = pkgs.callPackage ./libgdbm-legacy.nix {};
  # kokkos-with-toools = kokkos.override {
  #   env.KOKKOS_TOOLS_LIBS = kokkos-tools;
  # };
}
