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
  sphinx,
  level-zero,
  libcxx,
  opencl-headers,
  ocl-icd,
  callPackage,
}:

let
  version = "nightly-2025-07-18";
  deps = callPackage ./deps.nix { };
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
    # git
  ];

  # postPatch = ''
  #   for file in sycl/cmake/modules/FetchUnifiedRuntime.cmake \
  #     llvm/lib/SYCLNativeCPUUtils/CMakeLists.txt \
  #     xptifw/src/CMakeLists.txt
  #     # ${unified-runtime}/cmake/FetchLevelZero.cmake
  #   do
  #     substituteInPlace $file \
  #     --replace-fail "FetchContent_Populate" "FetchContent_MakeAvailable"
  #   done
  # '';

  cmakeFlags = [
    (lib.cmakeBool "FETCHCONTENT_FULLY_DISCONNECTED" true)
    (lib.cmakeBool "FETCHCONTENT_QUIET" false)

    (lib.cmakeBool "SYCL_UR_USE_FETCH_CONTENT" false)
    (lib.cmakeFeature "SYCL_UR_SOURCE_DIR" "${deps.unified-runtime}")
    (lib.cmakeFeature "LLVMGenXIntrinsics_SOURCE_DIR" "${deps.vc-intrinsics}")

    (lib.cmakeBool "UR_COMPUTE_RUNTIME_FETCH_REPO" false)
    (lib.cmakeFeature "UR_COMPUTE_RUNTIME_REPO" "${deps.compute-runtime}")
    (lib.cmakeFeature "UR_OPENCL_INCLUDE_DIR" "${opencl-headers}/include/CL")
    (lib.cmakeFeature "UR_OPENCL_ICD_LOADER_LIBRARY" "${ocl-icd}/lib/libOpenCL.so")

    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_EMHASH" "${deps.emhash}")
    (lib.cmakeFeature "FETCHCONTENT_SOURCE_DIR_UNIFIED_MEMORY_FRAMEWORK" "${deps.unified-memory-framework
    }")
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

  ];

  configurePhase = ''
    runHook preConfigure

    # pwd
    # ls
    # ls /build/source/
    # ls /build/source/llvm
    # ls /build/source/llvm/unified-runtime

    mkdir -p /build/source/build/
    # cp -r ${deps.compute-runtime}/* /build/source/build/content-exp-headers/
    ln -s ${deps.compute-runtime} /build/source/build/content-exp-headers
    ls /build/source/build/content-exp-headers/

    substituteInPlace buildbot/configure.py --replace-fail "abs_obj_dir = " "print('abs_src_dir = ', abs_src_dir); abs_obj_dir = "

    python buildbot/configure.py \
    -t Release \
    --enable-all-llvm-targets \
    --docs \
    --shared-libs \
    --cmake-gen Ninja \
    --l0-headers ${lib.getInclude level-zero}/include \
    --l0-loader ${lib.getLib level-zero}/lib \
    $cmakeFlags
    # It seems like on newest branch this is vendored
    # -DSYCL_UR_SOURCE_DIR=${deps.unified-runtime} \
    # -DSYCL_UR_SOURCE_DIR=/build/source/llvm/unified-runtime \
    # --native_cpu \
    # -DFETCHCONTENT_SOURCE_DIR_VC_INTRINSICS="${deps.vc-intrinsics}" \
    # --enable-backends \
    # --libcxx-include ${lib.getInclude libcxx}/include \
    # --libcxx-library ${lib.getLib libcxx}/lib \

    runHook postConfigure
  '';

  # buildInputs =
  #   [
  #     zlib
  #     libffi
  #     libedit
  #     libxml2
  #     ncurses
  #     elfutils
  #     libunwind
  #   ]
  #   ++ lib.optionals enableSYCL [
  #     spirv-llvm-translator
  #     opencl-headers
  #     ocl-icd
  #     level-zero
  #     tbb
  #     boost
  #   ];

  patches = [
    # Add any necessary patches here
  ];

  # cmakeFlags =
  #   [
  #     "-DCMAKE_BUILD_TYPE=${if enableDebugInfo then "Debug" else "Release"}"
  #     "-DLLVM_ENABLE_PROJECTS=clang;clang-tools-extra;compiler-rt;lld;lldb"
  #     "-DLLVM_ENABLE_RUNTIMES=libcxx;libcxxabi;libunwind"
  #     "-DLLVM_TARGETS_TO_BUILD=${lib.concatStringsSep ";" targets}"
  #     "-DLLVM_INCLUDE_EXAMPLES=OFF"
  #     "-DLLVM_INCLUDE_TESTS=OFF"
  #     "-DLLVM_INCLUDE_BENCHMARKS=OFF"
  #     "-DLLVM_INCLUDE_DOCS=OFF"
  #     "-DLLVM_ENABLE_BINDINGS=OFF"
  #     "-DLLVM_ENABLE_TERMINFO=${if ncurses != null then "ON" else "OFF"}"
  #     "-DLLVM_ENABLE_ZLIB=${if zlib != null then "ON" else "OFF"}"
  #     "-DLLVM_ENABLE_LIBXML2=${if libxml2 != null then "ON" else "OFF"}"
  #     "-DLLVM_ENABLE_FFI=${if libffi != null then "ON" else "OFF"}"
  #     "-DLLVM_BUILD_SHARED_LIBS=${if enableSharedLibraries then "ON" else "OFF"}"
  #     "-DLLVM_ENABLE_RTTI=${if enableRTTI then "ON" else "OFF"}"
  #     "-DLLVM_ENABLE_ASSERTIONS=${if enableAssertions then "ON" else "OFF"}"
  #     "-DLLVM_INSTALL_UTILS=ON"
  #     "-DLLVM_OPTIMIZED_TABLEGEN=ON"
  #     "-DLLVM_PARALLEL_LINK_JOBS=1"
  #     "-DCMAKE_INSTALL_PREFIX=${placeholder "out"}"
  #   ]
  #   ++ lib.optionals enableSYCL [
  #     # SYCL-specific flags
  #     "-DSYCL_BUILD_PI_CUDA=OFF" # Disable CUDA by default
  #     "-DSYCL_BUILD_PI_HIP=OFF" # Disable HIP by default
  #     "-DSYCL_BUILD_PI_OPENCL=ON" # Enable OpenCL
  #     "-DSYCL_BUILD_PI_LEVEL_ZERO=ON" # Enable Level Zero
  #     "-DLLVM_EXTERNAL_PROJECTS=sycl"
  #     "-DLLVM_EXTERNAL_SYCL_SOURCE_DIR=${src}/sycl"
  #     "-DSYCL_ENABLE_WERROR=OFF"
  #     "-DSYCL_INCLUDE_TESTS=OFF"
  #   ];

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
