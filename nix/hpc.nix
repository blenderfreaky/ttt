{ pkgs ? import <nixpkgs> { } }:

let
  _major_ver = "2025";
  _minor_ver = "0";
  _patch_ver = "1";
  pkgver = "${_major_ver}.${_minor_ver}.${_patch_ver}";
  pkgver_base = "${pkgver}.46";
  pkgver_hpc = "${pkgver}.47";
  _urlver_base = "dfc4a434-838c-4450-a6fe-2fa903b75aa7";
  _urlver_hpc = "b7f71cf2-8157-4393-abae-8cea815509f7";
in
pkgs.stdenv.mkDerivation {
  pname = "intel-oneapi-hpckit";
  version = pkgver;

  srcs = [
    (pkgs.fetchurl {
      url = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/${_urlver_base}/intel-oneapi-base-toolkit-${pkgver_base}_offline.sh";
      sha384 = "8f315562c26104eea7790e1fba868d63562f0fd1888623f0f4a286a234ec799beefefe78ffa904bd71dc6b8fb479df79";
    })
    (pkgs.fetchurl {
      url = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/${_urlver_hpc}/intel-oneapi-hpc-toolkit-${pkgver_hpc}_offline.sh";
      sha384 = "be217e7242c19d23698bf3055ecd992e9e1a469a8e23dc9de62767d985171c311db130c46f6cf979299e428c8c7c6f37";
    })
  ];

  nativeBuildInputs = [ pkgs.bash ];

  installPhase = ''
    mkdir -p $out/opt/intel/oneapi
    sh intel-oneapi-base-toolkit-${pkgver_base}_offline.sh -a \
      --silent --eula accept \
      --components all \
      --install-dir "$out/opt/intel/oneapi"
    sh intel-oneapi-hpc-toolkit-${pkgver_hpc}_offline.sh -a \
      --silent --eula accept \
      --components all \
      --install-dir "$out/opt/intel/oneapi"
  '';

  setupHook = ./setup-hook.sh;

  meta = with pkgs.lib; {
    description = "Intel oneAPI Base and HPC Toolkit for Linux";
    homepage = "https://software.intel.com/content/www/us/en/develop/tools/oneapi.html";
    license = licenses.unfree;
    platforms = [ "x86_64-linux" ];
  };
}
