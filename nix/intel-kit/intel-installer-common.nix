{
  buildFHSEnv,
  stdenv,
  lib,
  writeShellScript,
  fetchurl,
  autoPatchelfHook,

  config,
  cudaSupport ? config.cudaSupport,
  cudaPackages,
  autoAddDriverRunpath,

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
  libuuid,
  xorg, # .libX11, .libxcb, .libXcomposite, .libXdamage, .libXext, .libXfixes, .libXrandr
  libxkbcommon,

  # Advisor-only dependencies
  fontconfig,
  freetype,
  gdk-pixbuf,
  gtk2,
  # xorg.libXxf86vm

  # VTune-only dependencies
  elfutils,
  gtk3,
  opencl-clang_14,
  xdg-utils,
}:
{
  # "base" or "hpc"
  kit,
  version,
  uuid,
  sha256,
}:
let
  components_all = components == "all";
  components_default =
    components == "default" || (lib.isList components && lib.elem "default" components);
  components_string =
    if lib.isString components then
      if components_all || components_default then
        components
      else
        throw "Invalid string for Intel oneAPI components specification. Expected 'all' or 'default', but got \"${components}\"."
    else if lib.isList components then
      if lib.all lib.isString components then
        if lib.elem "default" components then
          let
            # "default" is a special-case, and the final string should start with it
            otherComponents = lib.filter (c: c != "default") components;
            orderedComponents = [ "default" ] ++ otherComponents;
          in
          lib.concatStringsSep ":" orderedComponents
        else
          lib.concatStringsSep ":" components
      else
        throw "Invalid list for oneAPI components specification. The list must only contain strings representing component names."
    else
      throw "Invalid type for oneAPI components specification. Expected a string ('all' or 'default') or a list of component names, but got a ${lib.typeOf components}.";
  components_used = {
    vtune = components_all || lib.elem "intel.oneapi.lin.vtune" components;
    advisor = components_all || lib.elem "intel.oneapi.lin.advisor" components;
    ui = components_used.vtune || components_used.advisor;
  };
in
stdenv.mkDerivation (finalAttrs: {
  pname = "intel-oneapi-${kit}kit";
  inherit version;

  src = fetchurl {
    url = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/${uuid}/intel-oneapi-base-toolkit-${version}_offline.sh";
    inherit sha256;
  };

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

  buildInputs =
    [
      level-zero
      zlib
      gcc
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
      export COMPONENTS=${components_string}
      mkdir -p $OUT
      ${fhs}/bin/oneapi-installer-fhs-env -- ${install}

      runHook postInstall
    '';

  passthru = {
    updateScript = ./packaging-scripts/update.sh;
    withComponents =
      components:
      finalAttrs.finalPackage.override {
        inherit components;
      };
  };

  autoPatchelfIgnoreMissingDeps = [
    "libcuda.so.1"
  ];

  meta =
    {
      license = lib.licenses.unfree;
      platforms = lib.platforms.linux;
      maintainers = [ lib.maintainers.blenderfreaky ];
    }
    // (
      if kit == "base" then
        {
          description = "Intel OneAPI BaseKit";
          homepage = "https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html";
        }
      else if kit == "hpc" then
        {
          description = "Intel oneAPI HPC Toolkit";
          homepage = "https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html";
        }
      else
        throw "kit must be either 'base' or 'hpc'"
    );
})
