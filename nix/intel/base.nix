{
  stdenv,
  lib,
  fetchurl,
  callPackage,
  alsa-lib,
  atk,
  at-spi2-atk,
  at-spi2-core,
  bzip2,
  cairo,
  libxcrypt,
  cudaPackages,
  cups,
  dbus,
  libdrm,
  libepoxy,
  elfutils,
  expat,
  libffi_3_2_1,
  fontconfig,
  freetype,
  mesa,
  libgdbm-legacy,
  gdk-pixbuf,
  gtk2,
  glib,
  gtk3,
  hwloc_1,
  rdma-core,
  ncurses5,
  nspr,
  nss,
  numactl,
  opencl-clang_14,
  pango,
  libpsm2,
  sqlite,
  intel-compute-runtime,
  ucx,
  systemd,
  libuuid,
  xorg,
  libxkbcommon,
  level-zero,
  zlib,
  libxcrypt-legacy,
  # Optional Dependencies
  full ? true,
  xdg-utils,
  libnotify,
  libgbm,
  gcc,
}:
let
  version = "2025.2.0.592";
  uuid = "bd1d0273-a931-4f7e-ab76-6a2a67d646c7";
in
(callPackage ./intel-installer-common.nix { }).overrideAttrs {
  pname = "intel-oneapi-basekit";
  inherit version;

  src = fetchurl {
    url = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/${uuid}/intel-oneapi-base-toolkit-${version}_offline.sh";
    hash = "sha256-XPYLFTjtt02/ihxvDziGmCY52rcJu3V2PxelrO+6xTA=";
  };

  buildInputs =
    [
      alsa-lib
      atk
      at-spi2-atk
      at-spi2-core
      bzip2
      cairo
      libxcrypt
      cudaPackages.cudatoolkit
      cups
      dbus
      libdrm
      libepoxy
      elfutils
      expat
      libffi_3_2_1
      fontconfig
      freetype
      mesa
      libgdbm-legacy
      gdk-pixbuf
      gtk2
      glib
      gtk3
      hwloc_1.lib
      rdma-core
      ncurses5
      nspr
      nss
      numactl
      opencl-clang_14
      pango
      libpsm2
      sqlite
      intel-compute-runtime
      ucx
      systemd
      libuuid
      xorg.libX11
      xorg.libxcb
      xorg.libXcomposite
      xorg.libXdamage
      xorg.libXext
      xorg.libXfixes
      libxkbcommon
      libxcrypt-legacy
      xorg.libXrandr
      xorg.libXxf86vm
      level-zero
      zlib
    ]
    ++ lib.optionals full [
      xdg-utils
      libnotify
      libgbm
      gcc
    ];

  autoPatchelfIgnoreMissingDeps = [
    "libsycl.so.6"
    "libcuda.so.1"
  ];

  meta = {
    description = "Intel oneAPI Base Toolkit";
    homepage = "https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html";
  };
}
