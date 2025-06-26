{pkgs}: rec {
  # mkl = pkgs.callPackage ./mkl.nix {};
  kokkos = pkgs.callPackage ./kokkos.nix {};
  kokkos-kernels = pkgs.callPackage ./kokkos-kernels.nix {inherit kokkos;};
  kokkos-tools = pkgs.callPackage ./kokkos-tools.nix {inherit kokkos;};
  # kokkos-with-toools = kokkos.override {
  #   env.KOKKOS_TOOLS_LIBS = kokkos-tools;
  # };

  intelPackages = pkgs.callPackage ./intel {};
}
