{
  lib,
  stdenv,
  fetchFromGitHub,
  gitUpdater,
  cmake,
  kokkos,
}:
stdenv.mkDerivation (finalAttrs: {
  pname = "kokkos-tools";
  version = "6493964";

  src = fetchFromGitHub {
    owner = "kokkos";
    repo = "kokkos-tools";
    rev = finalAttrs.version;
    hash = "sha256-EpygUxOawa0O0YbpsyIXr/w1mYZP2A1l8BdhG8S0EBc=";
  };

  nativeBuildInputs = [
    kokkos
    cmake
  ];

  cmakeFlags = [
  ];

  doCheck = true;
  passthru.updateScript = gitUpdater {};

  meta = with lib; {
    description = "Kokkos C++ Performance Portability Programming Ecosystem: Profiling and Debugging Tools";
    homepage = "https://github.com/kokkos/kokkos-tools";
    changelog = "https://github.com/kokkos/kokkos-tools/blob/${finalAttrs.src.rev}/CHANGELOG.md";
    license = with licenses; [asl20];
    maintainers = with maintainers; [];
    platforms = platforms.unix;
    broken = stdenv.hostPlatform.isDarwin;
  };
})
