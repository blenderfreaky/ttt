{
  stdenv,
  fetchFromGitHub,
  lib,
  cmake,
  ninja,
  unified-memory-framework,
  zlib,
  libbacktrace,
  python3,

  rocmPackages ? { },
  cudaPackages ? { },
  autoAddDriverRunpath,
  level-zero,
  # intel-compute-runtime,
  opencl-headers,
  ocl-icd,
  # khronos-ocl-icd-loader,
  levelZeroSupport ? true,
  openclSupport ? true,
  # Broken
  cudaSupport ? false,
  rocmSupport ? true,

  lit,
  filecheck,
  buildTests ? false,
}:
let
  version = "0.11.10";
  gtest = fetchFromGitHub {
    owner = "google";
    repo = "googletest";
    tag = "v1.15.2";
    sha256 = "sha256-1OJ2SeSscRBNr7zZ/a8bJGIqAnhkg45re0j3DtPfcXM=";
  };
  # rocmLibs = [
  #   rocmPackages.clr
  #   rocmPackages.rocm-device-libs
  #   rocmPackages.rocm-smi
  # ];
  # rocmPath = buildEnv {
  #   name = "rocm-path";
  #   paths = rocmLibs;
  # };
  compute-runtime = fetchFromGitHub {
    owner = "intel";
    repo = "compute-runtime";
    tag = "25.05.32567.17";
    sha256 = "sha256-/9UQJ5Ng2ip+3cNcVZOtKAmnx4LpmPja+aTghIqF1bc=";
  };
  hdr-histogram = fetchFromGitHub {
    owner = "HdrHistogram";
    repo = "HdrHistogram_c";
    tag = "0.11.8";
    sha256 = "sha256-TFlrC4bgK8o5KRZcLMlYU5EO9Oqaqe08PjJgmsUl51M=";
  };
in
stdenv.mkDerivation {
  name = "unified-runtime";
  inherit version;

  nativeBuildInputs = [
    cmake
    ninja
    python3
    unified-memory-framework
    zlib
    libbacktrace
  ]
  ++ lib.optionals openclSupport [
    opencl-headers
    ocl-icd
    # khronos-ocl-icd-loader
  ]
  ++ lib.optionals cudaSupport [
    cudaPackages.cuda_cudart
    # cudaPackages.markForCudatoolkitRootHook
    autoAddDriverRunpath
  ]
  ++ lib.optionals levelZeroSupport [
    level-zero
    # intel-compute-runtime
  ]
  ++ lib.optionals buildTests [
    lit
    filecheck
  ];

  src = fetchFromGitHub {
    owner = "oneapi-src";
    repo = "unified-runtime";
    # tag = "v${version}";
    # TODO
    rev = "a6437589c67c3acdeffa80a0d544a6612e63a29b";
    sha256 = "sha256-zMZc0sy+waqcGyGVu+T0/+ZSpH8oHcQzNp1B3MNUopI=";
  };

  # postPatch = ''
  #   # This is removed in newer versions, but we're on the latest tag and it's still here.
  #   # umf::disjoint_pool seems to have been removed in newer versions of Unified Memory Framework.
  #   substituteInPlace source/common/CMakeLists.txt \
  #     --replace-fail "umf::disjoint_pool" ""
  # '';

  preConfigure = ''
    # mkdir -p /build/source/build/content-exp-headers/level_zero/
    # ln -s ${compute-runtime}/level_zero/include /build/source/build/content-exp-headers/level_zero/include
    # ls -l /build/source/build/content-exp-headers
    # ls -l /build/source/build/content-exp-headers/level_zero/include
    # ls -l /build/source/build/content-exp-headers/level_zero/include/
    # ls -l /build/source/build/content-exp-headers/level_zero/include/*
  '';

  cmakeFlags = [
    (lib.cmakeBool "FETCHCONTENT_FULLY_DISCONNECTED" true)
    (lib.cmakeBool "FETCHCONTENT_QUIET" false)

    # (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_CONTENT_EXP_HEADERS" "${compute-runtime}")
    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_HDR_HISTOGRAM" "${hdr-histogram}")

    # Their CMake code will print that it's fetching from/will download from github anyways.
    # This can be safely ignored - it's not actually doing that.
    (lib.cmakeBool "UR_USE_EXTERNAL_UMF" true)

    (lib.cmakeBool "UR_ENABLE_LATENCY_HISTOGRAM" true)

    (lib.cmakeFeature "UR_OPENCL_INCLUDE_DIR" "${lib.getInclude opencl-headers}/include")
    (lib.cmakeBool "UR_COMPUTE_RUNTIME_FETCH_REPO" false)
    (lib.cmakeFeature "UR_COMPUTE_RUNTIME_REPO" "${compute-runtime}")

    (lib.cmakeBool "UR_BUILD_EXAMPLES" buildTests)
    (lib.cmakeBool "UR_BUILD_TESTS" buildTests)

    (lib.cmakeBool "UR_BUILD_ADAPTER_L0" levelZeroSupport)
    (lib.cmakeBool "UR_BUILD_ADAPTER_L0_V2" false)
    # (lib.cmakeBool "UR_BUILD_ADAPTER_L0_V2" levelZeroSupport)
    (lib.cmakeBool "UR_BUILD_ADAPTER_OPENCL" openclSupport)
    (lib.cmakeBool "UR_BUILD_ADAPTER_CUDA" cudaSupport)
    (lib.cmakeBool "UR_BUILD_ADAPTER_HIP" rocmSupport)
    (lib.cmakeBool "UR_BUILD_ADAPTER_NATIVE_CPU" true)
    # (lib.cmakeBool "UR_BUILD_ADAPTER_ALL" false)
  ]
  ++ lib.optionals cudaSupport [
    (lib.cmakeFeature "CUDA_TOOLKIT_ROOT_DIR" "${cudaPackages.cudatoolkit}")
    (lib.cmakeFeature "CUDAToolkit_ROOT" "${cudaPackages.cudatoolkit}")
    (lib.cmakeFeature "CUDA_INCLUDE_DIRS" "${cudaPackages.cudatoolkit}/include/")
    (lib.cmakeFeature "CUDA_CUDA_LIBRARY" "${cudaPackages.cudatoolkit}/lib/")
  ]
  ++ lib.optionals rocmSupport [
    (lib.cmakeFeature "UR_HIP_ROCM_DIR" "${rocmPackages.clr}")
  ]
  ++ lib.optionals levelZeroSupport [
    (lib.cmakeFeature "UR_LEVEL_ZERO_INCLUDE_DIR" "${lib.getInclude level-zero}/include")
    (lib.cmakeFeature "UR_LEVEL_ZERO_LOADER_LIBRARY" "${lib.getLib level-zero}/lib")
  ]
  ++ lib.optionals buildTests [
    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_GOOGLETEST" "${gtest}")
  ];

}
