{
  stdenv,
  lib,
  fetchurl,
  callPackage,
  # The components to install
  # Note that because the offline installer is used, all components will still
  # be downloaded, however only the selected components will be installed.
  # Options:
  # "all", "default", or ["foo", "bar", ...], ["default", "foo", ...] where foo, bar
  # are components of the Intel OneAPI HPC Toolkit like:
  #  intel.oneapi.lin.dpcpp-cpp-compiler
  #  intel.oneapi.lin.dpl
  #  intel.oneapi.lin.vtune
  #  ...
  components ? "all",
  # Core dependencies
  level-zero,
  zlib,
  gcc,
  # MPI dependencies
  rdma-core,
  numactl,
  libpsm2,
  ucx,
  libuuid,
  # UI dependencies
  alsa-lib,
  at-spi2-atk,
  bzip2,
  cairo,
  libxcrypt-legacy,
  cups, # .lib
  dbus, # .lib
  libdrm,
  expat,
  libffi_3_2_1,
  libgbm,
  gdbm_1_13,
  glib,
  ncurses5,
  nspr,
  nss,
  pango,
  sqlite,
  systemd, # For libudev
  #libuuid,
  xorg, # .libX11, .libxcb, .libXcomposite, .libXdamage, .libXext, .libXfixes, .libXrandr
  libxkbcommon,
  # VTune-only dependencies
  elfutils,
  gtk3,
  opencl-clang_14,
  xdg-utils,
  # Advisor-only dependencies
  fontconfig,
  freetype,
  gdk-pixbuf,
  gtk2,
  # xorg.libXxf86vm
}: let
  version = "2025.2.0.575";
  uuid = "e974de81-57b7-4ac1-b039-0512f8df974e";
  components_all = components == "all";
  components_default =
    components == "default" || (lib.isList components && lib.elem "default" components);
  components_string =
    if lib.isString components
    then
      if components_all || components_default
      then components
      else throw "Invalid string for Intel oneAPI components specification. Expected 'all' or 'default', but got \"${components}\"."
    else if lib.isList components
    then
      if lib.all lib.isString components
      then
        if lib.elem "default" components
        then let
          # "default" is a special-case, and the final string should start with it
          otherComponents = lib.filter (c: c != "default") components;
          orderedComponents = ["default"] ++ otherComponents;
        in
          lib.concatStringsSep ":" orderedComponents
        else lib.concatStringsSep ":" components
      else throw "Invalid list for oneAPI components specification. The list must only contain strings representing component names."
    else throw "Invalid type for oneAPI components specification. Expected a string ('all' or 'default') or a list of component names, but got a ${lib.typeOf components}.";
  components_used = {
    mpi = components_all || components_default || lib.elem "intel.oneapi.lin.mpi.devel" components;
    vtune = components_all || lib.elem "intel.oneapi.lin.vtune" components;
    advisor = components_all || lib.elem "intel.oneapi.lin.advisor" components;
    ui = components_used.vtune || components_used.advisor;
  };
in
  (callPackage ./intel-installer-common.nix {}).overrideAttrs {
    pname = "intel-oneapi-hpckit";
    inherit version;

    src = fetchurl {
      url = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/${uuid}/intel-oneapi-hpc-toolkit-${version}_offline.sh";
      hash = "sha256-xu00FBbRgyVE7kWOI29Kt2l58ZTs7E90vYMTV9d7LX4=";
    };

    # This is read in the build phase in the FHSEnv
    COMPONENTS = "${components_string}";

    buildInputs =
      [
        level-zero
        zlib
        gcc
      ]
      ++ lib.optionals components_used.mpi [
        rdma-core
        numactl
        libpsm2
        ucx
        libuuid
      ]
      ++ lib.optionals components_used.ui [
        alsa-lib
        at-spi2-atk
        bzip2
        cairo
        libxcrypt-legacy
        cups.lib
        dbus.lib
        libdrm
        expat
        libffi_3_2_1
        libgbm
        gdbm_1_13
        glib
        ncurses5
        nspr
        nss
        pango
        sqlite
        systemd # For libudev
        libuuid
        xorg.libX11
        xorg.libxcb
        xorg.libXcomposite
        xorg.libXdamage
        xorg.libXext
        xorg.libXfixes
        xorg.libXrandr
        libxkbcommon
      ]
      ++ lib.optionals components_used.vtune [
        elfutils
        gtk3
        opencl-clang_14
        xdg-utils
      ]
      ++ lib.optionals components_used.advisor [
        fontconfig
        freetype
        gdk-pixbuf
        gtk2
        xorg.libXxf86vm
      ];

    autoPatchelfIgnoreMissingDeps = [
      "libcuda.so.1"
    ];
    meta = {
      description = "Intel oneAPI HPC Toolkit";
      homepage = "https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html";
    };
  }
