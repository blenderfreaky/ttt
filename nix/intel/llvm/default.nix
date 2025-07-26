{
  lib,
  stdenv,
  fetchFromGitHub,
  # fetchpatch,
  cmake,
  ninja,
  python3,
  # zlib,
  # libffi,
  # libedit,
  # libxml2,
  # ncurses,
  # elfutils,
  # libunwind,
  pkg-config,
  hwloc,
  zstd,
  valgrind,
  # git,
  # spirv-llvm-translator,
  # opencl-headers,
  # ocl-icd,
  # level-zero,
  # tbb,
  # boost,
  # enableSharedLibraries ? true,
  # enableRTTI ? true,
  # enableAssertions ? false,
  # enableDebugInfo ? false,
  # enableSYCL ? true,
  # targets ? [
  #   "X86"
  #   "AMDGPU"
  #   "NVPTX"
  #   "SPIRV"
  # ],
  # We use the in-tree unified-runtime, but we need all the same flags as the out-of-tree version.
  # Rather than duplicating the flags, we can simply use the existing flags.
  # We can also use this to debug unified-runtime without building the entire LLVM project.
  unified-runtime,
  sphinx,
  level-zero,
  libcxx,
  libxml2,
  # opencl-headers,
  # ocl-icd,
  callPackage,
  spirv-tools,

  levelZeroSupport ? true,
  openclSupport ? true,
  # Broken
  cudaSupport ? false,
  rocmSupport ? true,

  buildTests ? false,
}:

let
  version = "nightly-2025-07-18";
  deps = callPackage ./deps.nix { };
  unified-runtime' = unified-runtime.override {
    inherit
      levelZeroSupport
      openclSupport
      cudaSupport
      rocmSupport
      buildTests
      ;
  };
in
stdenv.mkDerivation rec {
  pname = "intel-llvm";
  inherit version;

  src = fetchFromGitHub {
    owner = "intel";
    repo = "llvm";
    tag = "nightly-2025-07-18";
    sha256 = "sha256-xpL3M24T+e3hDrdSLRGBTRxC+IzBec5rP1V5wbRmJxs=";
  };

  nativeBuildInputs = [
    cmake
    ninja
    python3
    pkg-config
    hwloc
    zstd
    sphinx
    spirv-tools
    libxml2
    valgrind.dev
  ]
  ++ unified-runtime'.nativeBuildInputs;

  postPatch = ''
    # The latter is used everywhere except this one file. For some reason,
    # the former is not set, at least when building with Nix, so we replace it.
    substituteInPlace unified-runtime/cmake/helpers.cmake \
      --replace-fail "PYTHON_EXECUTABLE" "Python3_EXECUTABLE"
  '';

  preConfigure = ''
    # For some reason, it doesn't create this on its own,
    # causing a cryptic Permission denied error.
    mkdir -p /build/source/build/unified-runtime/source/common/level_zero_loader/level_zero/
  '';

  cmakeFlags = [
    (lib.cmakeBool "FETCHCONTENT_FULLY_DISCONNECTED" true)
    (lib.cmakeBool "FETCHCONTENT_QUIET" false)

    # (lib.cmakeBool "SYCL_UR_USE_FETCH_CONTENT" false)
    # (lib.cmakeFeature "SYCL_UR_SOURCE_DIR" "${deps.unified-runtime}")
    (lib.cmakeFeature "LLVMGenXIntrinsics_SOURCE_DIR" "${deps.vc-intrinsics}")

    # (lib.cmakeBool "UR_COMPUTE_RUNTIME_FETCH_REPO" false)
    # (lib.cmakeFeature "UR_COMPUTE_RUNTIME_REPO" "${deps.compute-runtime}")
    # (lib.cmakeFeature "UR_OPENCL_INCLUDE_DIR" "${opencl-headers}/include/CL")
    # (lib.cmakeFeature "UR_OPENCL_ICD_LOADER_LIBRARY" "${ocl-icd}/lib/libOpenCL.so")
    #
    # (lib.cmakeFeature "OpenCL_HEADERS" "${opencl-headers}")
    # (lib.cmakeFeature "OPENCL_INCLUDE_DIR" "${lib.getInclude opencl-headers}/include")
    # (lib.cmakeFeature "OPENCL_ICD_LOADER_HEADERS_DIR" "${lib.getLib opencl-headers}/lib/libOpenCL.so")

    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_EMHASH" "${deps.emhash}")
    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_SPIRV-HEADERS" "${deps.spirv-headers}")
    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_PARALLEL-HASHMAP" "${deps.parallel-hashmap}")
    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_OCL-HEADERS" "${deps.opencl-headers}")
    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_OCL-ICD" "${deps.opencl-icd-loader}")

    # (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_UNIFIED_MEMORY_FRAMEWORK" "${deps.unified-memory-framework
    # }")
    # (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_OPENCL_HEADERS" "${deps.opencl-headers}")
    # (lib.cmakeFeature )
    # -DFETCHCONTENT_UNIFIED_MEMORY_FRAMEWORK_SOURCE_DIR=${deps.unified-memory-framework} \
    # -DOpenCL_HEADERS={deps.ocl-headers} \
    # -DOpenCL_LIBRARY_SRC={deps.ocl-loader} \
    # -DOpenCL-ICD=${ocl-icd}/lib/libOpenCL.so \
    # -DUR_LEVEL_ZERO_INCLUDE=${lib.getInclude level-zero}/include \
    # -DUR_LEVEL_ZERO_LOADER_LIBRARY=${lib.getLib level-zero}/lib \

    # "-DUR_USE_EXTERNAL_UMF=ON"
    # "-DUR_OPENCL_INCLUDE_DIR=${opencl-headers}/include/CL"
    # "-DUR_LEVEL_ZERO_LOADER_LIBRARY=${level-zero}/lib/libze_loader.so"
    # "-DUR_LEVEL_ZERO_INCLUDE_DIR=${level-zero}/include"
    # "-DUR_COMPUTE_RUNTIME_LEVEL_ZERO_INCLUDE_DIR=${deps.compute-runtime}"

    # "-DLEVEL_ZERO_INCLUDE_DIR=${level-zero}/include/level_zero"
    # "-DLEVEL_ZERO_LIBRARY=${level-zero}/lib/libze_loader.so"

    # "-DBOOST_MP11_SOURCE_DIR={pins.mp11}"
    # "-DBOOST_MODULE_SRC_DIR={boost.dev}"

    # "-DXPTIFW_EMHASH_HEADERS={pins.emhash}"
    # "-DXPTIFW_PARALLEL_HASHMAP_HEADERS={pins.parallel-hashmap}"

  ]
  ++ unified-runtime'.cmakeFlags;

  configurePhase = ''
    runHook preConfigure

    python buildbot/configure.py \
    -t Release \
    --enable-all-llvm-targets \
    --docs \
    --shared-libs \
    --cmake-gen Ninja \
    --l0-headers ${lib.getInclude level-zero}/include/level_zero \
    --l0-loader ${lib.getLib level-zero}/lib/libze_loader.so \
    $cmakeFlags
    # --native_cpu \
    # --enable-backends \
    # --libcxx-include ${lib.getInclude libcxx}/include \
    # --libcxx-library ${lib.getLib libcxx}/lib \

    runHook postConfigure
  '';

  buildPhase = ''
    runHook preBuild
    python buildbot/compile.py --verbose
    runHook postBuild
  '';

  requiredSystemFeatures = [ "big-parallel" ];
  enableParallelBuilding = true;

  doCheck = true;

  meta = with lib; {
    description = "Intel LLVM-based compiler with SYCL support";
    longDescription = ''
      Intel's LLVM-based compiler toolchain with Data Parallel C++ (DPC++)
      and SYCL support for heterogeneous computing across CPUs, GPUs, and FPGAs.
    '';
    homepage = "https://github.com/intel/llvm";
    # TODO: Apache with LLVM exceptions
    # license = with licenses; [ ncsa ];
    maintainers = with maintainers; [ blenderfreaky ];
    platforms = platforms.linux;
    # This is a large build that requires significant resources
    hydraPlatforms = [ ];
  };

  passthru = {
    isClang = true;
    inherit version;
  };
}
