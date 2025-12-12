{
  lib,
  stdenv,
  fetchFromGitHub,
  gitUpdater,
  cmake,
  kokkos,
}:
stdenv.mkDerivation (finalAttrs: {
  pname = "kokkos-kernels";
  version = "5.0.0";

  src = fetchFromGitHub {
    owner = "kokkos";
    repo = "kokkos-kernels";
    rev = finalAttrs.version;
    hash = "sha256-zwLPi8g3wS3TbsZVkAYnjV6ZvPMqN7Dv/cL5SOwkc+8=";
  };

  nativeBuildInputs = [
    kokkos
    cmake
  ];

  cmakeFlags = [
    (lib.cmakeBool "KokkosKernels_ENABLE_TESTS" true)
  ];

  doCheck = true;
  passthru.updateScript = gitUpdater {};

  meta = with lib; {
    description = "Kokkos C++ Performance Portability Programming Ecosystem: Math Kernels - Provides BLAS, Sparse BLAS and Graph Kernels";
    homepage = "https://github.com/kokkos/kokkos-kernels";
    changelog = "https://github.com/kokkos/kokkos-kernels/blob/${finalAttrs.src.rev}/CHANGELOG.md";
    license = with licenses; [asl20];
    maintainers = with maintainers; [];
    platforms = platforms.unix;
    broken = stdenv.hostPlatform.isDarwin;
  };
})
