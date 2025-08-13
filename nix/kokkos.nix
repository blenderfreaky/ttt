{
  lib,
  stdenv,
  fetchFromGitHub,
  gitUpdater,
  cmake,
  python3,
  rocmPackages,
  rocmSupport ? false,
}:
stdenv.mkDerivation (finalAttrs: {
  pname = "kokkos";
  version = "4.6.02";

  src = fetchFromGitHub {
    owner = "kokkos";
    repo = "kokkos";
    rev = finalAttrs.version;
    hash = "sha256-gpnaxQ3X+bqKiP9203I1DELDGXocRwMPN9nHFk5r6pM=";
  };

  nativeBuildInputs =
    [
      cmake
      python3
    ]
    ++ lib.optionals rocmSupport [
      rocmPackages.clr
    ];

  cmakeFlags = [
    (lib.cmakeBool "Kokkos_ENABLE_LIBDL" true)
    (lib.cmakeBool "Kokkos_ENABLE_TESTS" true)
    (lib.cmakeBool "Kokkos_ENABLE_HIP" rocmSupport)
  ];

  postPatch = ''
    patchShebangs .
  '';

  doCheck = true;
  passthru.updateScript = gitUpdater {};

  meta = with lib; {
    description = "C++ Performance Portability Programming EcoSystem";
    homepage = "https://github.com/kokkos/kokkos";
    changelog = "https://github.com/kokkos/kokkos/blob/${finalAttrs.src.rev}/CHANGELOG.md";
    license = with licenses; [asl20];
    maintainers = with maintainers; [Madouura];
    platforms = platforms.unix;
    broken = stdenv.hostPlatform.isDarwin;
  };
})
