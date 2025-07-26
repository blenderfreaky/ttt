{
  stdenv,
  fetchFromGitHub,
  lib,
  cmake,
  ninja,
  level-zero,
  hwloc,
  # jemalloc,
  tbb,
  numactl,
  pkg-config,
  cudaPackages,
  # cudaSupport ? config.cudaSupport
  cudaSupport ? false,
  levelZeroSupport ? true,

  ctestCheckHook,
  lit,
  filecheck,
  buildTests ? false,

  python3,
  doxygen,
  sphinx,
  buildDocs ? false,
}:
let
  version = "1.0.0";
  tag = "v${version}";
  gtest = fetchFromGitHub {
    owner = "google";
    repo = "googletest";
    tag = "v1.15.2";
    sha256 = "sha256-1OJ2SeSscRBNr7zZ/a8bJGIqAnhkg45re0j3DtPfcXM=";
  };
  gbench = fetchFromGitHub {
    owner = "google";
    repo = "benchmark";
    tag = "v1.9.0";
    sha256 = "sha256-5cl1PIjhXaL58kSyWZXRWLq6BITS2BwEovPhwvk2e18=";
  };
in
stdenv.mkDerivation (finalAttrs: {
  name = "unified-memory-framework";
  inherit version;

  nativeBuildInputs = [
    cmake
    ninja
    level-zero
    hwloc
    # jemalloc
    tbb
    pkg-config
  ]
  ++ lib.optionals cudaSupport [
    cudaPackages.cuda_cudart
  ]
  ++ lib.optionals buildTests [
    numactl
  ]
  ++ lib.optionals buildDocs [
    python3
    doxygen
    sphinx
  ];

  nativeCheckInputs = lib.optionals buildTests [
    ctestCheckHook
  ];

  src = fetchFromGitHub {
    owner = "oneapi-src";
    repo = "unified-memory-framework";
    inherit tag;
    sha256 = "sha256-nolnyxnupHDzz92/uFpIJsmEkcvD9MgI0oMX0V8aM1s=";
  };

  postPatch = ''
    # The CMake tries to find out the version via git.
    # Since we're not in a clone, git describe won't work.
    substituteInPlace cmake/helpers.cmake \
      --replace-fail "git describe --always" "echo ${tag}"

    #substituteInPlace \
    #  --replace-fail ";$<LINK_ONLY:>" ""
  '';

  cmakeFlags = [
    (lib.cmakeBool "FETCHCONTENT_FULLY_DISCONNECTED" true)
    (lib.cmakeBool "FETCHCONTENT_QUIET" false)

    (lib.cmakeBool "UMF_BUILD_CUDA_PROVIDER" cudaSupport)
    (lib.cmakeBool "UMF_BUILD_LEVEL_ZERO_PROVIDER" levelZeroSupport)

    # (lib.cmakeBool "UMF_BUILD_LIBUMF_POOL_JEMALLOC" true)

    (lib.cmakeBool "UMF_BUILD_TESTS" buildTests)
    (lib.cmakeBool "UMF_BUILD_GPU_TESTS" buildTests)
    (lib.cmakeBool "UMF_BUILD_BENCHMARKS" buildTests)
    (lib.cmakeBool "UMF_BUILD_EXAMPLES" buildTests)
    (lib.cmakeBool "UMF_BUILD_GPU_EXAMPLES" buildTests)
  ]
  ++ lib.optionals buildTests [
    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_GOOGLETEST" "${gtest}")
    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_GOOGLEBENCHMARK" "${gbench}")
  ];

  passthru.tests = finalAttrs.finalPackage.override { buildTests = true; };
})
