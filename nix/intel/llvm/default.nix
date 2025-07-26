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
  lld,
  callPackage,
  spirv-tools,

  levelZeroSupport ? true,
  openclSupport ? true,
  # Broken
  cudaSupport ? false,
  rocmSupport ? true,
  nativeCpuSupport ? true,
  vulkanSupport ? true,

  useLibcxx ? true,
  useLdd ? true,

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
      nativeCpuSupport
      vulkanSupport
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
  ++ lib.optionals useLdd [
    lld
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

    (lib.cmakeFeature "LLVMGenXIntrinsics_SOURCE_DIR" "${deps.vc-intrinsics}")

    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_EMHASH" "${deps.emhash}")
    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_SPIRV-HEADERS" "${deps.spirv-headers}")
    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_PARALLEL-HASHMAP" "${deps.parallel-hashmap}")
    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_OCL-HEADERS" "${deps.opencl-headers}")
    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_OCL-ICD" "${deps.opencl-icd-loader}")
    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_ONEAPI-CK" "${deps.oneapi-ck}")
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
    ${lib.optionalString cudaSupport "--cuda"} \
    ${lib.optionalString rocmSupport "--hip"} \
    ${lib.optionalString nativeCpuSupport "--native_cpu"} \
    ${lib.optionalString useLibcxx "--use-libcxx"} \
    ${lib.optionalString useLibcxx "--libcxx-include ${lib.getInclude libcxx}/include"} \
    ${lib.optionalString useLibcxx "--libcxx-library ${lib.getLib libcxx}/lib"} \
    ${lib.optionalString useLdd "--use-lld"} \
    $cmakeFlags

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
