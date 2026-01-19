{pkgs}: rec {
  # mkl = pkgs.callPackage ./mkl.nix {};
  kokkos = pkgs.callPackage ./kokkos.nix {};
  kokkos-kernels = pkgs.callPackage ./kokkos-kernels.nix {inherit kokkos;};
  kokkos-tools = pkgs.callPackage ./kokkos-tools.nix {inherit kokkos;};
  # kokkos-with-toools = kokkos.override {
  #   env.KOKKOS_TOOLS_LIBS = kokkos-tools;
  # };

  mkTtt = {
    backend,
    extraBuildInputs ? [],
    extraNativeBuildInputs ? [],
    extraEnv ? {},
  }:
    pkgs.rustPlatform.buildRustPackage ({
        pname = "ttt";
        version = "0.1.0";
        src = ../burn;

        cargoLock = {
          lockFile = ../burn/Cargo.lock;
          allowBuiltinFetchGit = true;
        };

        nativeBuildInputs = with pkgs; [pkg-config cmake] ++ extraNativeBuildInputs;
        buildInputs = [pkgs.openssl] ++ extraBuildInputs;

        buildFeatures = [backend];
        buildNoDefaultFeatures = true;

        # Tests require GPU hardware and network access
        doCheck = false;
      }
      // extraEnv);

  mkTttDocker = {
    name,
    tttPkg,
    runtimeDeps ? [],
    extraEnv ? [],
  }:
    pkgs.dockerTools.buildLayeredImage {
      inherit name;
      tag = "latest";
      contents = with pkgs;
        [
          tttPkg
          cacert
          bashInteractive
          fish
          coreutils
          bintools
          file
          findutils

          ripgrep
          fd
          dust
          duf
          tmux
          stdenv.cc.cc.lib
          dockerTools.fakeNss

          (
            python313.withPackages
            (ps:
              with ps; [
                huggingface-hub
              ])
          )
        ]
        ++ runtimeDeps;
      fakeRootCommands = ''
        mkdir -p ./root ./tmp
        mkdir -p ./usr/lib/x86_64-linux-gnu
        mkdir -p ./usr/local/nvidia/lib64
        mkdir -p ./usr/lib64
        mkdir -p ./run/nvidia
        mkdir -p ./tmp ./root
        chmod 1777 ./tmp
      '';
      enableFakechroot = true;
      config = {
        Cmd = ["/bin/fish"];
        Env =
          [
            "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
            "HOME=/root"
            "USER=root"
            "TMPDIR=/tmp"
            "NVIDIA_VISIBLE_DEVICES=all"
            "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
            "LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/lib64"
          ]
          ++ extraEnv;
        WorkingDir = "/root";
      };
    };

  ttt-rocm = mkTtt {
    backend = "rocm";
    extraNativeBuildInputs = with pkgs.rocmPackages; [hipcc];
    extraBuildInputs = with pkgs.rocmPackages; [hip-common clr rocblas rocwmma rocm-device-libs];
    extraEnv = {
      ROCM_PATH = "${pkgs.rocmPackages.clr}";
      HIP_PATH = "${pkgs.rocmPackages.clr}";
      ROCM_DEVICE_LIB_PATH = "${pkgs.rocmPackages.rocm-device-libs}/amdgcn/bitcode";
    };
  };

  ttt-cuda = mkTtt {
    backend = "cuda";
    extraBuildInputs = [pkgs.cudaPackages_12.cudatoolkit];
    extraEnv.CUDA_PATH = "${pkgs.cudaPackages_12.cudatoolkit}";
  };

  ttt-rocm-docker = mkTttDocker {
    name = "ttt-rocm";
    tttPkg = ttt-rocm;
    runtimeDeps = with pkgs.rocmPackages; [clr rocblas rocwmma rocm-device-libs rocm-runtime] ++ [pkgs.nvtopPackages.amd];
    extraEnv = [
      "ROCM_PATH=${pkgs.rocmPackages.clr}"
      "HIP_PATH=${pkgs.rocmPackages.clr}"
      "ROCM_DEVICE_LIB_PATH=${pkgs.rocmPackages.rocm-device-libs}/amdgcn/bitcode"
    ];
  };

  ttt-cuda-docker = mkTttDocker {
    name = "ttt-cuda";
    tttPkg = ttt-cuda;
    runtimeDeps = with pkgs.cudaPackages_12; [cudatoolkit] ++ [pkgs.nvtopPackages.nvidia];
    extraEnv = [
      "CUDA_PATH=${pkgs.cudaPackages_12.cudatoolkit}"
    ];
  };
}
