{
  buildFHSEnv,
  stdenv,
  autoPatchelfHook,
  lib,
  fetchurl,
  writeShellScript,
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
  gdbm,
  gdk-pixbuf,
  gtk2,
  glib,
  gtk3,
  hwloc,
  rdma-core,
  ncurses5,
  nspr,
  nss,
  numactl,
  opencl-clang,
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
  autoAddDriverRunpath,
}: let
  version = "2025.1.3.7";
in
  stdenv.mkDerivation {
    pname = "intel-oneapi-basekit";
    inherit version;

    src = fetchurl {
      url = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/4a5320d1-0b48-458d-9668-fd0e4501208c/intel-oneapi-base-toolkit-2025.1.3.7_offline.sh";
      hash = "sha256-jJkCjQwV3JH16WZzoSJlaoZ9w01PHJTx66DB347CLaw=";
    };

    unpackPhase = ''
      sh $src \
        --extract-only --extract-folder . \
        --remove-extracted-files no --log ./extract.log
    '';
    #dontUnpack = true;

    nativeBuildInputs = [autoPatchelfHook autoAddDriverRunpath];

    buildInputs = [
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
      gdbm
      gdk-pixbuf
      gtk2
      glib
      gtk3
      hwloc.lib
      rdma-core
      ncurses5
      nspr
      nss
      numactl
      opencl-clang
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
    ];

    installPhase = let
      fhs = buildFHSEnv rec {
        pname = "oneapi-installer-fhs-env";
        inherit version;

        profile = "oneapi-installer-fhs-env";

        targetPkgs = pkgs: (with pkgs; [
          bash
          patchelf
          stdenv.cc.cc.lib
          fakeroot
          # These aren't actually used, but the bootstrapper links to them so we need them present.
          ncurses
          qt6Packages.qtbase
          qt6Packages.qt5compat
        ]);

        nativeBuildInputs = [
          autoPatchelfHook
        ];

        buildInputs = [
          stdenv.cc.cc.lib
        ];

        extraBwrapArgs = ["--bind" "$out" "$out"];
      };
      install = writeShellScript "intel-oneapi-basekit-installer" ''
        #sh $src \
        #  --extract-only --extract-folder . \
        #  --remove-extracted-files no --log ./extract.log

        ls
        pwd
        cd intel*

        patchelf --set-interpreter "$(cat $NIX_CC/nix-support/dynamic-linker)" \
          ./bootstrapper

        mkdir log

        exec sh ./install.sh \
          --silent \
          --eula accept \
          --components all \
          --log-dir ./log \
          --ignore-errors \
          --install-dir $OUT \
      '';
    in ''
      runHook preInstall

      export OUT=$out/opt/intel/oneapi
      mkdir -p $OUT
      ${fhs}/bin/oneapi-installer-fhs-env -- ${install}

      runHook postInstall
    '';

    meta = {
      description = "Intel oneAPI Base Toolkit";
      homepage = "https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html";
      license = lib.licenses.unfree;
      platforms = lib.platforms.linux;
      maintainers = [lib.maintainers.blenderfreaky];
    };
  }
