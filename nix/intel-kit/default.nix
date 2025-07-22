{ pkgs }:
rec {
  intel-oneapi-basekit = pkgs.callPackage ./base.nix {
    inherit
      libffi_3_2_1
      opencl-clang_14
      gdbm_1_13
      ;
  };
  intel-oneapi-hpckit = pkgs.callPackage ./hpc.nix {
    inherit
      libffi_3_2_1
      opencl-clang_14
      gdbm_1_13
      ;
  };
  libffi_3_2_1 = pkgs.callPackage ./libffi_3_2_1.nix { };
  opencl-clang_14 = pkgs.callPackage ./opencl-clang_14.nix { };
  gdbm_1_13 = pkgs.callPackage ./gdbm_1_13.nix { };
}
