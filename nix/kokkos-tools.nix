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
  version = "dd08b98";

  src = fetchFromGitHub {
    owner = "kokkos";
    repo = "kokkos-tools";
    rev = finalAttrs.version;
    hash = "sha256-ST8VHhLy06BnJTUsrU7dCYNKxr/h4bUQXGvGKYn8qdg=";
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
