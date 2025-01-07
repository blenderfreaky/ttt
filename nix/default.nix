{pkgs}: {
  kokkos = pkgs.callPackage ./kokkos.nix {};
  adaptivecpp = pkgs.callPackage ./adaptivecpp.nix {};
}
