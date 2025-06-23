{
  buildFHSEnv,
  stdenv,
  autoPatchelfHook,
  lib,
  fetchurl,
  writeShellScript,
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

    nativeBuildInputs = [autoPatchelfHook];

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

        touch $OUT/bye
        echo i am going to touches you

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

      #mkdir -p ./out
      #oneapi-installer-fhs-env --install-dir ./out

      export OUT=$out/opt/intel/oneapi
      mkdir -p $OUT
      touch $OUT/hi
      ${fhs}/bin/oneapi-installer-fhs-env -- ${install}

      #export DIR=$(mktemp -d)
      #oneapi-installer-fhs-env --install-dir $DIR

      #mkdir -p $out/opt/intel/oneapi
      #cp -r $OUT_DIR/* $out/opt/intel/oneapi

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
