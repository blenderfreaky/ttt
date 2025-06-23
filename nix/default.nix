{pkgs}: rec {
  # mkl = pkgs.callPackage ./mkl.nix {};
  kokkos = pkgs.callPackage ./kokkos.nix {};
  kokkos-kernels = pkgs.callPackage ./kokkos-kernels.nix {inherit kokkos;};
  kokkos-tools = pkgs.callPackage ./kokkos-tools.nix {inherit kokkos;};
  base = pkgs.callPackage ./base.nix { inherit libffi_3_2_1; };
  hpc = pkgs.callPackage ./hpc.nix {};
  libffi_3_2_1 = pkgs.callPackage ./libffi_3_2_1.nix {};
  # kokkos-with-toools = kokkos.override {
  #   env.KOKKOS_TOOLS_LIBS = kokkos-tools;
  # };
}
