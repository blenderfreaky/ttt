{pkgs}: rec {
  intel-oneapi-basekit = pkgs.callPackage ./base.nix {inherit libffi_3_2_1 hwloc_1 opencl-clang_14 libgdbm-legacy;};
  intel-oneapi-hpckit = pkgs.callPackage ./hpc.nix {inherit libffi_3_2_1 hwloc_1 opencl-clang_14 libgdbm-legacy;};
  libffi_3_2_1 = pkgs.callPackage ./libffi_3_2_1.nix {};
  hwloc_1 = pkgs.callPackage ./hwloc_1.nix {};
  opencl-clang_14 = pkgs.callPackage ./opencl-clang_14.nix {};
  libgdbm-legacy = pkgs.callPackage ./libgdbm-legacy.nix {};
}
