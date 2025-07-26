{ fetchFromGitHub }:
{
  vc-intrinsics = fetchFromGitHub {
    owner = "intel";
    repo = "vc-intrinsics";
    # See llvm/lib/SYCLLowerIR/CMakeLists.txt:17
    rev = "60cea7590bd022d95f5cf336ee765033bd114d69";
    sha256 = "sha256-1K16UEa6DHoP2ukSx58OXJdtDWyUyHkq5Gd2DUj1644=";
  };

  spirv-headers = fetchFromGitHub {
    owner = "KhronosGroup";
    repo = "SPIRV-Headers";
    # See llvm-spirv/spirv-headers-tag.conf
    rev = "c9aad99f9276817f18f72a4696239237c83cb775";
    sha256 = "sha256-/KfUxWDczLQ/0DOiFC4Z66o+gtoF/7vgvAvKyv9Z9OA=";
  };

  oneapi-ck = fetchFromGitHub {
    owner = "uxlfoundation";
    repo = "oneapi-construction-kit";
    # See llvm/lib/SYCLNativeCPUUtils/CMakeLists.txt:44
    rev = "d0a32d701e34b3285de7ce776ea36abfec673df7";
    sha256 = "sha256-d0uwd5bF+qhTjX/chrjew91QHuGANekpEdHSjQUOYUA=";
  };

  opencl-headers = fetchFromGitHub {
    owner = "KhronosGroup";
    repo = "OpenCL-Headers";
    # See opencl/CMakeLists.txt:23
    rev = "6eabe90aa7b6cff9c67800a2fe25a0cd88d8b749";
    sha256 = "sha256-6S9z6d09deODp5U62Ob8GOBGIV0cGpyG2jSYlv3uINw=";
  };

  opencl-icd-loader = fetchFromGitHub {
    owner = "KhronosGroup";
    repo = "OpenCL-ICD-Loader";
    # See opencl/CMakeLists.txt:24
    rev = "ddf6c70230a79cdb8fcccfd3c775b09e6820f42e";
    sha256 = "sha256-ixZU5Tln4qeJGUb5qcc/+HHpMTWtc17CQYrmVirlLoc=";
  };

  emhash = fetchFromGitHub {
    owner = "ktprime";
    repo = "emhash";
    # See sycl/cmake/modules/FetchEmhash.cmake:12
    rev = "3ba9abdfdc2e0430fcc2fd8993cad31945b6a02b";
    sha256 = "sha256-w/iW5n9BzdiieZfxnVBF5MJTpHtZoWCUomjZ0h4OGH8=";
  };

  parallel-hashmap = fetchFromGitHub {
    owner = "greg7mdp";
    repo = "parallel-hashmap";
    # See xptifw/src/CMakeLists.txt:15
    rev = "8a889d3699b3c09ade435641fb034427f3fd12b6";
    sha256 = "sha256-hcA5sjL0LHuddEJdJdFGRbaEXOAhh78wRa6csmxi4Rk=";
  };

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
}
