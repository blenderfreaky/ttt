{
  buildFHSEnv,
  stdenv,
  autoPatchelfHook,
  lib,
  writeShellScript,
  autoAddDriverRunpath,
}:
stdenv.mkDerivation {
  unpackPhase = ''
    runHook preUnpack

    sh $src \
      --extract-only --extract-folder . \
      --remove-extracted-files no --log ./extract.log

    runHook postUnpack
  '';

  #nativeBuildInputs = [autoPatchelfHook autoAddDriverRunpath];
  nativeBuildInputs = [autoPatchelfHook];

  installPhase = let
    fhs = buildFHSEnv rec {
      name = "oneapi-installer-fhs-env";

      #profile = "oneapi-installer-fhs-env";

      targetPkgs = pkgs: (with pkgs; [
        patchelf
        stdenv.cc.cc.lib
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
    license = lib.licenses.unfree;
    platforms = lib.platforms.linux;
    maintainers = [lib.maintainers.blenderfreaky];
  };
}
