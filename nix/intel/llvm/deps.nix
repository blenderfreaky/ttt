{ fetchFromGitHub }:
{

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
}
