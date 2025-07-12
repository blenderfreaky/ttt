{
  buildFHSEnv,
  stdenv,
  autoPatchelfHook,
  lib,
  writeShellScript,
  autoAddDriverRunpath,

  config,
  cudaSupport ? config.cudaSupport,
  cudaPackages,
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
  nativeBuildInputs =
    [ autoPatchelfHook ]
    ++ lib.optionals cudaSupport [
      autoAddDriverRunpath
      cudaPackages.cuda_nvcc
      (lib.getDev cudaPackages.cuda_cudart)
    ];

  installPhase =
    let
      # The installer will try to act as root if we don't wrap it like this.
      # fakeroot did not work for me.
      fhs = buildFHSEnv {
        name = "oneapi-installer-fhs-env";

        targetPkgs =
          pkgs:
          (with pkgs; [
            patchelf
            stdenv.cc.cc.lib
          ]);

        nativeBuildInputs = [
          autoPatchelfHook
        ];

        # The installer also links to qt6 and other libraries,
        # but these are only used for the GUI.
        # Notably, these are vendored in $src/lib.
        # The installer runs fine with just the C++ library however.
        buildInputs = [
          stdenv.cc.cc.lib
        ];

        # This allows the installer to actually write to the $out directory.
        # Otherwise you'd get permission denied errors.
        extraBwrapArgs = [
          "--bind"
          "$out"
          "$out"
        ];
      };
      install = writeShellScript "intel-oneapi-basekit-installer" ''
        cd intel*

        patchelf --set-interpreter "$(cat $NIX_CC/nix-support/dynamic-linker)" \
          ./bootstrapper

        mkdir log

        echo "Installing the following components: $COMPONENTS"

        exec sh ./install.sh \
          --silent \
          --eula accept \
          --components "$COMPONENTS" \
          --log-dir ./log \
          --ignore-errors \
          --install-dir $OUT \
      '';
    in
    ''
      runHook preInstall

      export OUT=$out/opt/intel/oneapi
      mkdir -p $OUT
      ${fhs}/bin/oneapi-installer-fhs-env -- ${install}

      runHook postInstall
    '';

  passthru.updateScript = ./packaging-scripts/update.sh;

  meta = {
    license = lib.licenses.unfree;
    platforms = lib.platforms.linux;
    maintainers = [ lib.maintainers.blenderfreaky ];
  };
}
