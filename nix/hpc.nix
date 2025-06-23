{pkgs ? import <nixpkgs> {}}: let
  _major_ver = "2025";
  _minor_ver = "0";
  _patch_ver = "1";
  pkgver = "${_major_ver}.${_minor_ver}.${_patch_ver}";
  pkgver_base = "${pkgver}.46";
  pkgver_hpc = "${pkgver}.47";
  _urlver_base = "dfc4a434-838c-4450-a6fe-2fa903b75aa7";
  _urlver_hpc = "b7f71cf2-8157-4393-abae-8cea815509f7";
  base_kit = pkgs.fetchurl {
    url = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/${_urlver_base}/intel-oneapi-base-toolkit-${pkgver_base}_offline.sh";
    hash = "sha256-CUp4crvbHB4dDmXaVTSX3oGRMrKYsN32JLRQA82oKEQ=";
  };
  hpc_kit = pkgs.fetchurl {
    url = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/${_urlver_hpc}/intel-oneapi-hpc-toolkit-${pkgver_hpc}_offline.sh";
    hash = "sha256-lBpNTMwFz7YPXtvnwsXjQZfM5WHBLVZc2Mb+/mt1L7Y=";
  };
in
  pkgs.stdenv.mkDerivation {
    pname = "intel-oneapi-hpckit";
    version = pkgver;

    #srcs = [
    #  base_kit
    #  hpc_kit
    #];

    src = base_kit;

    unpackPhase = ''
      ls -lRaht
      ls $base_kit
      ls $src
      pwd
    '';

    nativeBuildInputs = [pkgs.bash];

    #    sourceRoot = ".";

    installPhase = ''
      mkdir -p $out/opt/intel/oneapi
      echo $base_kit
      ls $base_kit
      ls $hpc_kit
      ls
      #cat env-vars
      sh $base_kit/intel-oneapi-base-toolkit-${pkgver_base}_offline.sh -a \
        --silent --eula accept \
        --components all \
        --install-dir "$out/opt/intel/oneapi"
      sh $hpc_kit/intel-oneapi-hpc-toolkit-${pkgver_hpc}_offline.sh -a \
        --silent --eula accept \
        --components all \
        --install-dir "$out/opt/intel/oneapi"
    '';

    #setupHook = ./setup-hook.sh;

    meta = with pkgs.lib; {
      description = "Intel oneAPI Base and HPC Toolkit for Linux";
      homepage = "https://software.intel.com/content/www/us/en/develop/tools/oneapi.html";
      license = licenses.unfree;
      platforms = ["x86_64-linux"];
    };
  }
