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
}:

let
  version = "6.1.0";

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

  unified-runtime = fetchFromGitHub {
    owner = "oneapi-src";
    repo = "unified-runtime";
    tag = "v0.11.10";
    sha256 = "sha256-tVnTAPWkIJafj/kEZczx/XmCShxLNSec6NxvURRPVSg=";
  };

  unified-memory-framework = fetchFromGitHub {
    owner = "oneapi-src";
    repo = "unified-memory-framework";
    tag = "v0.10.0";
    sha256 = "sha256-8X08hlLulq132drznb4QQcv2qXWwCc6LRMFDDRcU3bk=";
  };

  vc-intrinsics = fetchFromGitHub {
    owner = "intel";
    repo = "vc-intrinsics";
    tag = "v0.23.1";
    sha256 = "sha256-7coQegLcgIKiqnonZmgrKlw6FCB3ltSh6oMMvdopeQc=";
  };

  # TODO: Sparse fetch just "level_zero/include"
  compute-runtime = fetchFromGitHub {
    owner = "intel";
    repo = "compute-runtime";
    tag = "24.39.31294.12";
    sha256 = "sha256-7GNtAo20DgxAxYSPt6Nh92nuuaS9tzsQGH+sLnsvBKU=";
  };

  # level_zero_loader_src = fetchFromGitHub {
  #   owner = "oneapi-src";
  #   repo = "level-zero";
  #   tag = "v1.19.2";
  #   sha256 = "sha256-MnTPu7jsjHR+PDHzj/zJiBKi9Ou/cjJvrf87yMdSnz0=";
  # };

  # compute_runtime_src = fetchFromGitHub {
  #   owner = "intel";
  #   repo = "compute-runtime";
  #   tag = "24.39.31294.12";
  #   sha256 = "";
  # };

  # unified_memory_framework_src = fetchFromGitHub {
  #   owner = "oneapi-src";
  #   repo = "unified-memory-framework";
  #   tag = "v0.10.0";
  #   sha256 = "";
  # };

  # hdr_histogram_c_src = fetchFromGitHub {
  #   owner = "HdrHistogram";
  #   repo = "HdrHistogram_c";
  #   tag = "0.11.8";
  #   sha256 = "";
  # };

  # opencl_headers_src = fetchFromGitHub {
  #   owner = "KhronosGroup";
  #   repo = "OpenCL-Headers";
  #   tag = "542d7a8f65ecfd88b38de35d8b10aa67b36b33b2";
  #   sha256 = "";
  # };

  # opencl_icd_loader_main_src = fetchFromGitHub {
  #   owner = "KhronosGroup";
  #   repo = "OpenCL-ICD-Loader";
  #   tag = "main";
  #   sha256 = "";
  # };

  # opencl_icd_loader_hash_src = fetchFromGitHub {
  #   owner = "KhronosGroup";
  #   repo = "OpenCL-ICD-Loader";
  #   tag = "804b6f040503c47148bee535230070da6b857ae4";
  #   sha256 = "";
  # };

  # boost_mp11_src = fetchFromGitHub {
  #   owner = "boostorg";
  #   repo = "mp11";
  #   tag = "863d8b8d2b20f2acd0b5870f23e553df9ce90e6c";
  #   sha256 = "";
  # };

  # boost_unordered_src = fetchFromGitHub {
  #   owner = "boostorg";
  #   repo = "unordered";
  #   tag = "5e6b9291deb55567d41416af1e77c2516dc1250f";
  #   sha256 = "";
  # };

  # boost_assert_src = fetchFromGitHub {
  #   owner = "boostorg";
  #   repo = "assert";
  #   tag = "447e0b3a331930f8708ade0e42683d12de9dfbc3";
  #   sha256 = "";
  # };

  # boost_config_src = fetchFromGitHub {
  #   owner = "boostorg";
  #   repo = "config";
  #   tag = "11385ec21012926e15a612e3bf9f9a71403c1e5b";
  #   sha256 = "";
  # };

  # boost_container_hash_src = fetchFromGitHub {
  #   owner = "boostorg";
  #   repo = "container_hash";
  #   tag = "6d214eb776456bf17fbee20780a034a23438084f";
  #   sha256 = "";
  # };

  # boost_core_src = fetchFromGitHub {
  #   owner = "boostorg";
  #   repo = "core";
  #   tag = "083b41c17e34f1fc9b43ab796b40d0d8bece685c";
  #   sha256 = "";
  # };

  # boost_describe_src = fetchFromGitHub {
  #   owner = "boostorg";
  #   repo = "describe";
  #   tag = "50719b212349f3d1268285c586331584d3dbfeb5";
  #   sha256 = "";
  # };

  # boost_predef_src = fetchFromGitHub {
  #   owner = "boostorg";
  #   repo = "predef";
  #   tag = "0fdfb49c3a6789e50169a44e88a07cc889001106";
  #   sha256 = "";
  # };

  # boost_static_assert_src = fetchFromGitHub {
  #   owner = "boostorg";
  #   repo = "static_assert";
  #   tag = "ba72d3340f3dc6e773868107f35902292f84b07e";
  #   sha256 = "";
  # };

  # boost_throw_exception_src = fetchFromGitHub {
  #   owner = "boostorg";
  #   repo = "throw_exception";
  #   tag = "7c8ec2114bc1f9ab2a8afbd629b96fbdd5901294";
  #   sha256 = "";
  # };

  # spirv_headers_src = fetchFromGitHub {
  #   owner = "KhronosGroup";
  #   repo = "SPIRV-Headers";
  #   tag = "SPIRV_HEADERS_TAG_FROM_FILE"; # Tag read from spirv-headers-tag.conf
  #   sha256 = "";
  # };

  # emhash_src = fetchFromGitHub {
  #   owner = "ktprime";
  #   repo = "emhash";
  #   tag = "96dcae6fac2f5f90ce97c9efee61a1d702ddd634";
  #   sha256 = "";
  # };

  # parallel_hashmap_src = fetchFromGitHub {
  #   owner = "greg7mdp";
  #   repo = "parallel-hashmap";
  #   tag = "8a889d3699b3c09ade435641fb034427f3fd12b6";
  #   sha256 = "";
  # };

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
    # cp -r ${compute-runtime}/* /build/source/build/content-exp-headers/
    ln -s ${compute-runtime} /build/source/build/content-exp-headers
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
    -DSYCL_UR_SOURCE_DIR=${unified-runtime} \
    -DLLVMGenXIntrinsics_SOURCE_DIR=${vc-intrinsics} \
    -DFETCHCONTENT_UNIFIED_MEMORY_FRAMEWORK_SOURCE_DIR=${unified-memory-framework} \
    # -DFETCHCONTENT_LEVEL-ZERO-LOADER_SOURCE_DIR=${level_zero_loader_src} \
    # -DFETCHCONTENT_EXP-HEADERS_SOURCE_DIR=${compute_runtime_src} \
    # -DFETCHCONTENT_UNIFIED-MEMORY-FRAMEWORK_SOURCE_DIR=${unified_memory_framework_src} \
    # -DFETCHCONTENT_HDR_HISTOGRAM_SOURCE_DIR=${hdr_histogram_c_src} \
    # -DFETCHCONTENT_OPENCL-HEADERS_SOURCE_DIR=${opencl_headers_src} \
    # -DFETCHCONTENT_OPENCL-ICD-LOADER_SOURCE_DIR=${opencl_icd_loader_main_src} \
    # -DFETCHCONTENT_OCL-HEADERS_SOURCE_DIR=${opencl_headers_src} \
    # -DFETCHCONTENT_OCL-ICD_SOURCE_DIR=${opencl_icd_loader_hash_src} \
    # -DFETCHCONTENT_UNIFIED-RUNTIME_SOURCE_DIR=${unified-runtime} \
    # -DFETCHCONTENT_BOOST_MP11_SOURCE_DIR=${boost_mp11_src} \
    # -DFETCHCONTENT_BOOST_UNORDERED_SOURCE_DIR=${boost_unordered_src} \
    # -DFETCHCONTENT_BOOST_ASSERT_SOURCE_DIR=${boost_assert_src} \
    # -DFETCHCONTENT_BOOST_CONFIG_SOURCE_DIR=${boost_config_src} \
    # -DFETCHCONTENT_BOOST_CONTAINER_HASH_SOURCE_DIR=${boost_container_hash_src} \
    # -DFETCHCONTENT_BOOST_CORE_SOURCE_DIR=${boost_core_src} \
    # -DFETCHCONTENT_BOOST_DESCRIBE_SOURCE_DIR=${boost_describe_src} \
    # -DFETCHCONTENT_BOOST_PREDEF_SOURCE_DIR=${boost_predef_src} \
    # -DFETCHCONTENT_BOOST_STATIC_ASSERT_SOURCE_DIR=${boost_static_assert_src} \
    # -DFETCHCONTENT_BOOST_THROW_EXCEPTION_SOURCE_DIR=${boost_throw_exception_src} \
    # -DFETCHCONTENT_SPIRV-HEADERS_SOURCE_DIR=${spirv_headers_src} \
    # -DFETCHCONTENT_EMHASH-HEADERS_SOURCE_DIR=${emhash_src} \
    # -DFETCHCONTENT_PARALLEL-HASHMAP_SOURCE_DIR=${parallel_hashmap_src} \
    # -DFETCHCONTENT_EXP_HEADERS_SOURCE_DIR=${compute-runtime}/level_zero/include \
    # -DFETCHCONTENT_EXP_HEADERS_POPULATE_SOURCE_DIR=${compute-runtime} \
    # -DFETCHCONTENT_SOURCE_DIR_EXP_HEADERS=${compute-runtime} \
    # -DFETCHCONTENT_SOURCE_DIR_EXP_HEADERS_POPULATE=${compute-runtime} \
    || ls /build/source/build/

    # It seems like on newest branch this is vendored
    # -DSYCL_UR_SOURCE_DIR=''${unified-runtime} \
    # -DSYCL_UR_SOURCE_DIR=/build/source/llvm/unified-runtime \
    # --native_cpu \
    # -DFETCHCONTENT_SOURCE_DIR_VC_INTRINSICS="${vc-intrinsics}" \
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
