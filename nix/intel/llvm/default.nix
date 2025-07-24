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
  callPackage,
}:

let
  version = "6.1.0";
  deps = callPackage ./deps.nix { };
in
stdenv.mkDerivation rec {
  pname = "intel-llvm";
  inherit version;

  src = fetchFromGitHub {
    owner = "intel";
    repo = "llvm";
    tag = "v${version}";
    sha256 = "sha256-yyvbG8GwBPA+Nv6xd4ifeInAtfCggLZIcNe8FHp9k9M=";
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

    python buildbot/configure.py \
    --l0-headers ${lib.getInclude level-zero}/include \
    -DUR_LEVEL_ZERO_INCLUDE=${lib.getInclude level-zero}/include \
    --l0-loader ${lib.getLib level-zero}/lib \
    -DUR_LEVEL_ZERO_LOADER_LIBRARY=${lib.getLib level-zero}/lib \
    -t Release \
    --enable-all-llvm-targets \
    --docs \
    --shared-libs \
    --cmake-gen Ninja \
    -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
    -DFETCHCONTENT_UPDATES_DISCONNECTED=ON \
    -DSYCL_UR_USE_FETCH_CONTENT=OFF \
    -DSYCL_UR_SOURCE_DIR=${deps.unified-runtime} \
    -DLLVMGenXIntrinsics_SOURCE_DIR=${deps.vc-intrinsics} \
    -DFETCHCONTENT_UNIFIED_MEMORY_FRAMEWORK_SOURCE_DIR=${deps.unified-memory-framework} \
    # -DFETCHCONTENT_LEVEL-ZERO-LOADER_SOURCE_DIR=${deps.level_zero_loader_src} \
    # -DFETCHCONTENT_EXP-HEADERS_SOURCE_DIR=${deps.compute_runtime_src} \
    # -DFETCHCONTENT_UNIFIED-MEMORY-FRAMEWORK_SOURCE_DIR=${deps.unified_memory_framework_src} \
    # -DFETCHCONTENT_HDR_HISTOGRAM_SOURCE_DIR=${deps.hdr_histogram_c_src} \
    # -DFETCHCONTENT_OPENCL-HEADERS_SOURCE_DIR=${deps.opencl_headers_src} \
    # -DFETCHCONTENT_OPENCL-ICD-LOADER_SOURCE_DIR=${deps.opencl_icd_loader_main_src} \
    # -DFETCHCONTENT_OCL-HEADERS_SOURCE_DIR=${deps.opencl_headers_src} \
    # -DFETCHCONTENT_OCL-ICD_SOURCE_DIR=${deps.opencl_icd_loader_hash_src} \
    # -DFETCHCONTENT_UNIFIED-RUNTIME_SOURCE_DIR=${deps.unified-runtime} \
    # -DFETCHCONTENT_BOOST_MP11_SOURCE_DIR=${deps.boost_mp11_src} \
    # -DFETCHCONTENT_BOOST_UNORDERED_SOURCE_DIR=${deps.boost_unordered_src} \
    # -DFETCHCONTENT_BOOST_ASSERT_SOURCE_DIR=${deps.boost_assert_src} \
    # -DFETCHCONTENT_BOOST_CONFIG_SOURCE_DIR=${deps.boost_config_src} \
    # -DFETCHCONTENT_BOOST_CONTAINER_HASH_SOURCE_DIR=${deps.boost_container_hash_src} \
    # -DFETCHCONTENT_BOOST_CORE_SOURCE_DIR=${deps.boost_core_src} \
    # -DFETCHCONTENT_BOOST_DESCRIBE_SOURCE_DIR=${deps.boost_describe_src} \
    # -DFETCHCONTENT_BOOST_PREDEF_SOURCE_DIR=${deps.boost_predef_src} \
    # -DFETCHCONTENT_BOOST_STATIC_ASSERT_SOURCE_DIR=${deps.boost_static_assert_src} \
    # -DFETCHCONTENT_BOOST_THROW_EXCEPTION_SOURCE_DIR=${deps.boost_throw_exception_src} \
    # -DFETCHCONTENT_SPIRV-HEADERS_SOURCE_DIR=${deps.spirv_headers_src} \
    # -DFETCHCONTENT_EMHASH-HEADERS_SOURCE_DIR=${deps.emhash_src} \
    # -DFETCHCONTENT_PARALLEL-HASHMAP_SOURCE_DIR=${deps.parallel_hashmap_src} \
    # -DFETCHCONTENT_EXP_HEADERS_SOURCE_DIR=${deps.compute-runtime}/level_zero/include \
    # -DFETCHCONTENT_EXP_HEADERS_POPULATE_SOURCE_DIR=${deps.compute-runtime} \
    # -DFETCHCONTENT_SOURCE_DIR_EXP_HEADERS=${deps.compute-runtime} \
    # -DFETCHCONTENT_SOURCE_DIR_EXP_HEADERS_POPULATE=${deps.compute-runtime} \

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
