{
  callPackage,

  libffi_3_2_1,
  opencl-clang_14,
  gdbm_1_13,
}:
(callPackage ./intel-installer-common.nix {
  inherit
    libffi_3_2_1
    opencl-clang_14
    gdbm_1_13
    ;
})
  {
    kit = "base";
    version = "2025.2.0.592";
    uuid = "bd1d0273-a931-4f7e-ab76-6a2a67d646c7";
    sha256 = "sha256-XPYLFTjtt02/ihxvDziGmCY52rcJu3V2PxelrO+6xTA=";
  }
