{
  lib,
  stdenv,
  fetchFromGitHub,
  fetchpatch,
  cmake,
  ninja,
  python3,
  zlib,
  libffi,
  libedit,
  libxml2,
  ncurses,
  elfutils,
  libunwind,
  pkg-config,
  git,
  spirv-llvm-translator,
  opencl-headers,
  ocl-icd,
  level-zero,
  tbb,
  boost,
  enableSharedLibraries ? true,
  enableRTTI ? true,
  enableAssertions ? false,
  enableDebugInfo ? false,
  enableSYCL ? true,
  targets ? [
    "X86"
    "AMDGPU"
    "NVPTX"
    "SPIRV"
  ],
}:

let
  version = "2024.2.0";

  # Intel LLVM for SYCL requires specific branch
  rev = "sycl-nightly/20241201";

in
stdenv.mkDerivation rec {
  pname = "intel-llvm";
  inherit version;

  src = fetchFromGitHub {
    owner = "intel";
    repo = "llvm";
    inherit rev;
    sha256 = lib.fakeSha256; # You'll need to update this with the actual hash
    fetchSubmodules = true;
  };

  nativeBuildInputs = [
    cmake
    ninja
    python3
    pkg-config
    git
  ];

  buildInputs =
    [
      zlib
      libffi
      libedit
      libxml2
      ncurses
      elfutils
      libunwind
    ]
    ++ lib.optionals enableSYCL [
      spirv-llvm-translator
      opencl-headers
      ocl-icd
      level-zero
      tbb
      boost
    ];

  patches = [
    # Add any necessary patches here
  ];

  cmakeFlags =
    [
      "-DCMAKE_BUILD_TYPE=${if enableDebugInfo then "Debug" else "Release"}"
      "-DLLVM_ENABLE_PROJECTS=clang;clang-tools-extra;compiler-rt;lld;lldb"
      "-DLLVM_ENABLE_RUNTIMES=libcxx;libcxxabi;libunwind"
      "-DLLVM_TARGETS_TO_BUILD=${lib.concatStringsSep ";" targets}"
      "-DLLVM_INCLUDE_EXAMPLES=OFF"
      "-DLLVM_INCLUDE_TESTS=OFF"
      "-DLLVM_INCLUDE_BENCHMARKS=OFF"
      "-DLLVM_INCLUDE_DOCS=OFF"
      "-DLLVM_ENABLE_BINDINGS=OFF"
      "-DLLVM_ENABLE_TERMINFO=${if ncurses != null then "ON" else "OFF"}"
      "-DLLVM_ENABLE_ZLIB=${if zlib != null then "ON" else "OFF"}"
      "-DLLVM_ENABLE_LIBXML2=${if libxml2 != null then "ON" else "OFF"}"
      "-DLLVM_ENABLE_FFI=${if libffi != null then "ON" else "OFF"}"
      "-DLLVM_BUILD_SHARED_LIBS=${if enableSharedLibraries then "ON" else "OFF"}"
      "-DLLVM_ENABLE_RTTI=${if enableRTTI then "ON" else "OFF"}"
      "-DLLVM_ENABLE_ASSERTIONS=${if enableAssertions then "ON" else "OFF"}"
      "-DLLVM_INSTALL_UTILS=ON"
      "-DLLVM_OPTIMIZED_TABLEGEN=ON"
      "-DLLVM_PARALLEL_LINK_JOBS=1"
      "-DCMAKE_INSTALL_PREFIX=${placeholder "out"}"
    ]
    ++ lib.optionals enableSYCL [
      # SYCL-specific flags
      "-DSYCL_BUILD_PI_CUDA=OFF" # Disable CUDA by default
      "-DSYCL_BUILD_PI_HIP=OFF" # Disable HIP by default
      "-DSYCL_BUILD_PI_OPENCL=ON" # Enable OpenCL
      "-DSYCL_BUILD_PI_LEVEL_ZERO=ON" # Enable Level Zero
      "-DLLVM_EXTERNAL_PROJECTS=sycl"
      "-DLLVM_EXTERNAL_SYCL_SOURCE_DIR=${src}/sycl"
      "-DSYCL_ENABLE_WERROR=OFF"
      "-DSYCL_INCLUDE_TESTS=OFF"
    ];

  # Intel LLVM requires a lot of memory to build
  requiredSystemFeatures = [ "big-parallel" ];

  # Disable parallel building to avoid memory issues
  enableParallelBuilding = false;

  preConfigure =
    ''
      # Set up build environment
      export PYTHON_EXECUTABLE=${python3}/bin/python3

      # Create build directory
      mkdir -p build
      cd build
    ''
    + lib.optionalString enableSYCL ''
      # SYCL-specific setup
      export SYCL_BUILD_ROOT=$PWD
    '';

  postInstall =
    ''
      # Create symlinks for common tools
      ln -sf $out/bin/clang $out/bin/clang++
      ln -sf $out/bin/clang $out/bin/cc
      ln -sf $out/bin/clang++ $out/bin/c++
    ''
    + lib.optionalString enableSYCL ''
          # SYCL-specific post-install
          # Ensure SYCL headers and libraries are in the right place
          if [ -d $out/include/sycl ]; then
            ln -sf $out/include/sycl $out/include/CL
          fi

          # Create dpcpp wrapper script
          cat > $out/bin/dpcpp << 'EOF'
      #!/bin/bash
      exec $out/bin/clang++ -fsycl "$@"
      EOF
          chmod +x $out/bin/dpcpp
    '';

  # Tests are disabled by default to speed up build
  doCheck = false;

  meta = with lib; {
    description = "Intel LLVM-based compiler with SYCL support";
    longDescription = ''
      Intel's LLVM-based compiler toolchain with Data Parallel C++ (DPC++)
      and SYCL support for heterogeneous computing across CPUs, GPUs, and FPGAs.

      This package includes:
      - Clang/LLVM compiler with Intel extensions
      - DPC++ compiler (dpcpp) for SYCL development
      - Runtime libraries for Intel GPUs and other accelerators
      - Integration with Intel Level Zero and OpenCL
    '';
    homepage = "https://github.com/intel/llvm";
    license = with licenses; [ ncsa ];
    maintainers = with maintainers; [ ];
    platforms = platforms.linux;
    # This is a large build that requires significant resources
    hydraPlatforms = [ ];
  };

  passthru = {
    isClang = true;
    inherit enableSYCL;
    inherit version;

    # Provide access to common tools
    tools = {
      clang = "${placeholder "out"}/bin/clang";
      clangxx = "${placeholder "out"}/bin/clang++";
      dpcpp = lib.optionalString enableSYCL "${placeholder "out"}/bin/dpcpp";
      llc = "${placeholder "out"}/bin/llc";
      opt = "${placeholder "out"}/bin/opt";
    };
  };
}
