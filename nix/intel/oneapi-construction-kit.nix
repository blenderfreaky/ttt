{
  stdenv,
  cmake,
  ninja,
  lib,
  fetchFromGitHub,
  llvmPackages_20,
  symlinkJoin,
}:
let
  # version = "v4.0.0";
  version = "d0a32d701e34b3285de7ce776ea36abfec673df7";
  llvm = symlinkJoin {
    name = "llvm";
    paths = [
      llvmPackages_20.libllvm.dev
      llvmPackages_20.clang-unwrapped.dev
    ];
  };
in
stdenv.mkDerivation {
  pname = "oneapi-construction-kit";
  inherit version;

  src = fetchFromGitHub {
    owner = "uxlfoundation";
    repo = "oneapi-construction-kit";
    # tag = "v${version}";
    rev = "${version}";
    sha256 = "sha256-d0uwd5bF+qhTjX/chrjew91QHuGANekpEdHSjQUOYUA=";
  };
  postPatch = ''
    type installPhase
    type ninjaInstallPhase
    type configurePhase
  '';

  nativeBuildInputs = [
    cmake
    ninja
  ];

  cmakeFlags = [
    (lib.cmakeFeature "CA_LLVM_INSTALL_DIR" "${llvm}")
  ];
}
