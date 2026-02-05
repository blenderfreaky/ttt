{
  pkgs,
  pkgs-rocm,
  craneLib,
}: let
  # Use ROCm from older nixpkgs to match RunPod kernel driver (6.x)
  rocmPackages6 = pkgs-rocm.rocmPackages;
  rocmPackages7 = pkgs.rocmPackages;
  cudaPackages = pkgs.cudaPackages_12_8;

  fullSrc = craneLib.cleanCargoSource ../crates;

  cargoLock = ../crates/Cargo.lock;
  cargoToml = ../crates/Cargo.toml;
  cargoVendorDir = craneLib.vendorCargoDeps {inherit cargoLock cargoToml;};

  commonArgs = {
    src = fullSrc;
    inherit cargoVendorDir;
    strictDeps = true;
    pname = "ttt";
    version = "0.1.0";
    nativeBuildInputs = with pkgs; [pkg-config cmake];
    buildInputs = [pkgs.openssl pkgs.sqlite];
  };

  mkCargoArtifacts = {
    backend,
    extraBuildInputs ? [],
    extraNativeBuildInputs ? [],
    extraEnv ? {},
  }:
    craneLib.buildDepsOnly (commonArgs
      // {
        pname = "ttt-deps";
        nativeBuildInputs = commonArgs.nativeBuildInputs ++ extraNativeBuildInputs;
        buildInputs = commonArgs.buildInputs ++ extraBuildInputs;
        cargoExtraArgs = "--no-default-features --features ${backend} --all-targets";
        doCheck = false;
      }
      // extraEnv);

  mkTtt = {
    backend,
    binSuffix ? "",
    extraBuildInputs ? [],
    extraNativeBuildInputs ? [],
    extraEnv ? {},
  }: let
    cargoArtifacts = mkCargoArtifacts {inherit backend extraBuildInputs extraNativeBuildInputs extraEnv;};
  in
    craneLib.buildPackage (commonArgs
      // {
        pname = "ttt${binSuffix}";
        version = "0.1.0";
        inherit cargoArtifacts;
        nativeBuildInputs = commonArgs.nativeBuildInputs ++ extraNativeBuildInputs;
        buildInputs = commonArgs.buildInputs ++ extraBuildInputs;
        cargoExtraArgs = "--no-default-features --features ${backend}";
        doCheck = false;
      }
      // extraEnv
      // (
        if binSuffix != ""
        then {
          postInstall = ''for f in $out/bin/*; do mv "$f" "$f${binSuffix}"; done'';
        }
        else {}
      ));

  mkTttBench = {
    backend,
    binSuffix ? "",
    extraBuildInputs ? [],
    extraNativeBuildInputs ? [],
    extraEnv ? {},
  }: let
    cargoArtifacts = mkCargoArtifacts {inherit backend extraBuildInputs extraNativeBuildInputs extraEnv;};
  in
    craneLib.buildPackage (commonArgs
      // {
        pname = "ttt-bench${binSuffix}";
        version = "0.1.0";
        inherit cargoArtifacts;
        nativeBuildInputs = commonArgs.nativeBuildInputs ++ extraNativeBuildInputs;
        buildInputs = commonArgs.buildInputs ++ extraBuildInputs;
        cargoExtraArgs = "--no-default-features --features ${backend} --bench ttt-bench-criterion";
        installPhaseCommand = ''
          mkdir -p $out/bin
          find target/release/deps -name "ttt-bench-criterion-*" -type f -executable ! -name "*.d" -exec cp {} $out/bin/ttt-bench-criterion${binSuffix} \;
        '';
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
          bottom
          htop
          btop
          (python313.withPackages (ps: with ps; [huggingface-hub tokenizers datasets]))
        ]
        ++ runtimeDeps;
      fakeRootCommands = ''
        mkdir -p ./root ./tmp ./usr/lib/x86_64-linux-gnu ./usr/local/nvidia/lib64 ./usr/lib64 ./run/nvidia
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
            "LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/lib64:/lib"
          ]
          ++ extraEnv;
        WorkingDir = "/root";
      };
    };

  # Backend build configs
  mkRocmConfig = rocmPkgs: {
    extraNativeBuildInputs = with rocmPkgs; [hipcc];
    extraBuildInputs = with rocmPkgs; [hip-common clr rocblas rocwmma rocm-device-libs];
    extraEnv = {
      ROCM_PATH = "${rocmPkgs.clr}";
      HIP_PATH = "${rocmPkgs.clr}";
      ROCM_DEVICE_LIB_PATH = "${rocmPkgs.rocm-device-libs}/amdgcn/bitcode";
    };
  };

  cudaConfig = {
    extraNativeBuildInputs = [cudaPackages.cudatoolkit];
    extraBuildInputs = [cudaPackages.cudatoolkit];
    extraEnv.CUDA_PATH = "${cudaPackages.cudatoolkit}";
  };

  # Build a complete backend package set: {f32.{ttt,ttt-bench}, ttt-docker}
  mkBackendPkgs = {
    name,
    backend,
    buildConfig,
    runtimeDeps,
    dockerEnv,
    nvtop,
  }: let
    f32 = {
      ttt = mkTtt (buildConfig // {inherit backend;});
      ttt-bench = mkTttBench (buildConfig // {inherit backend;});
    };
    # bf16 = {
    #   ttt = mkTtt (buildConfig
    #     // {
    #       backend = "${backend},bf16";
    #       binSuffix = "-bf16";
    #     });
    #   ttt-bench = mkTttBench (buildConfig
    #     // {
    #       backend = "${backend},bf16";
    #       binSuffix = "-bf16";
    #     });
    # };
  in {
    inherit
      f32
      # bf16
      ;
    ttt-docker = mkTttDocker {
      inherit name;
      tttPkg = f32.ttt;
      runtimeDeps =
        runtimeDeps
        ++ [
          f32.ttt-bench
          # bf16.ttt
          # bf16.ttt-bench
          nvtop
        ];
      extraEnv = dockerEnv;
    };
    deps = mkCargoArtifacts (buildConfig // {inherit backend;});
  };

  mkRocmPkgs = {
    name,
    rocmPkgs,
  }:
    mkBackendPkgs {
      inherit name;
      backend = "rocm";
      buildConfig = mkRocmConfig rocmPkgs;
      runtimeDeps = with rocmPkgs; [clr rocblas rocwmma rocm-device-libs rocm-runtime];
      dockerEnv = [
        "ROCM_PATH=${rocmPkgs.clr}"
        "HIP_PATH=${rocmPkgs.clr}"
        "ROCM_DEVICE_LIB_PATH=${rocmPkgs.rocm-device-libs}/amdgcn/bitcode"
      ];
      nvtop = pkgs.nvtopPackages.amd;
    };
in {
  rocm6 = mkRocmPkgs {
    name = "ttt-rocm6";
    rocmPkgs = rocmPackages6;
  };
  rocm7 = mkRocmPkgs {
    name = "ttt-rocm7";
    rocmPkgs = rocmPackages7;
  };

  cuda = mkBackendPkgs {
    name = "ttt-cuda";
    backend = "cuda";
    buildConfig = cudaConfig;
    runtimeDeps = [cudaPackages.cudatoolkit];
    dockerEnv = [
      "NVIDIA_VISIBLE_DEVICES=all"
      "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
      "CUDA_PATH=${cudaPackages.cudatoolkit}"
    ];
    nvtop = pkgs.nvtopPackages.nvidia;
  };
}
