{
  pkgs,
  craneLib,
}: rec {
  # mkl = pkgs.callPackage ./mkl.nix {};
  kokkos = pkgs.callPackage ./kokkos.nix {};
  kokkos-kernels = pkgs.callPackage ./kokkos-kernels.nix {inherit kokkos;};
  kokkos-tools = pkgs.callPackage ./kokkos-tools.nix {inherit kokkos;};
  # kokkos-with-toools = kokkos.override {
  #   env.KOKKOS_TOOLS_LIBS = kokkos-tools;
  # };

  # Common source for crane - include burn and thundercube directories
  fullSrc = pkgs.lib.cleanSourceWith {
    src = ../.;
    filter = path: type:
      let
        baseName = baseNameOf path;
        relPath = pkgs.lib.removePrefix (toString ../. + "/") (toString path);
        # Include burn and thundercube directories
        isRelevant = (pkgs.lib.hasPrefix "burn" relPath) || (pkgs.lib.hasPrefix "thundercube" relPath);
        # Exclude common non-source files
        isExcluded = baseName == ".git" || baseName == "target" || baseName == "result" || baseName == ".worktree";
      in
        isRelevant && !isExcluded;
  };

  # Point crane to the Cargo.lock in the burn directory
  cargoLock = ../burn/Cargo.lock;
  cargoToml = ../burn/Cargo.toml;

  # Vendored dependencies (shared between all builds)
  cargoVendorDir = craneLib.vendorCargoDeps {
    inherit cargoLock cargoToml;
  };

  # Common args shared between dep and package builds
  commonArgs = {
    src = fullSrc;
    inherit cargoVendorDir;
    strictDeps = true;

    # Explicitly set these since Cargo.toml is in a subdirectory
    pname = "ttt";
    version = "0.1.0";

    nativeBuildInputs = with pkgs; [
      pkg-config
      cmake
    ];
    buildInputs = [pkgs.openssl];

    # Build from the burn subdirectory while keeping thundercube accessible
    # The sourceRoot is relative to the unpacked source directory
    postUnpack = ''
      # Move into the burn subdirectory but ensure thundercube is accessible
      sourceRoot="$sourceRoot/burn"
    '';
  };

  # Build dependencies separately for caching
  mkCargoArtifacts = {
    backend,
    extraBuildInputs ? [],
    extraNativeBuildInputs ? [],
    extraEnv ? {},
  }:
    craneLib.buildDepsOnly (
      commonArgs
      // {
        pname = "ttt-deps"; # Override pname for deps

        nativeBuildInputs = commonArgs.nativeBuildInputs ++ extraNativeBuildInputs;
        buildInputs = commonArgs.buildInputs ++ extraBuildInputs;

        # Build deps for both bin and bench targets (all-targets includes benchmarks)
        cargoExtraArgs = "--no-default-features --features ${backend} --all-targets";

        # Tests require GPU hardware and network access
        doCheck = false;
      }
      // extraEnv
    );

  mkTtt = {
    backend,
    extraBuildInputs ? [],
    extraNativeBuildInputs ? [],
    extraEnv ? {},
  }: let
    cargoArtifacts = mkCargoArtifacts {
      inherit backend extraBuildInputs extraNativeBuildInputs extraEnv;
    };
  in
    craneLib.buildPackage (
      commonArgs
      // {
        pname = "ttt";
        version = "0.1.0";
        inherit cargoArtifacts;

        nativeBuildInputs = commonArgs.nativeBuildInputs ++ extraNativeBuildInputs;
        buildInputs = commonArgs.buildInputs ++ extraBuildInputs;

        cargoExtraArgs = "--no-default-features --features ${backend}";

        # Tests require GPU hardware and network access
        doCheck = false;
      }
      // extraEnv
    );

  mkTttBench = {
    backend,
    extraBuildInputs ? [],
    extraNativeBuildInputs ? [],
    extraEnv ? {},
  }: let
    cargoArtifacts = mkCargoArtifacts {
      inherit backend extraBuildInputs extraNativeBuildInputs extraEnv;
    };
  in
    craneLib.buildPackage (
      commonArgs
      // {
        pname = "ttt-bench";
        version = "0.1.0";
        inherit cargoArtifacts;

        nativeBuildInputs = commonArgs.nativeBuildInputs ++ extraNativeBuildInputs;
        buildInputs = commonArgs.buildInputs ++ extraBuildInputs;

        # Build the benchmark binary
        cargoExtraArgs = "--no-default-features --features ${backend} --bench ttt_benchmark";

        # Install the benchmark binary (criterion benchmarks go to deps/)
        installPhaseCommand = ''
          mkdir -p $out/bin
          find target/release/deps -name "ttt_benchmark-*" -type f -executable ! -name "*.d" -exec cp {} $out/bin/ttt_benchmark \;
        '';

        # Tests require GPU hardware and network access
        doCheck = false;
      }
      // extraEnv
    );

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

          (python313.withPackages (
            ps:
              with ps; [
                huggingface-hub
              ]
          ))
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
            "LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/lib64:/lib"
          ]
          ++ extraEnv;
        WorkingDir = "/root";
      };
    };

  rocmBuildConfig = {
    extraNativeBuildInputs = with pkgs.rocmPackages; [hipcc];
    extraBuildInputs = with pkgs.rocmPackages; [
      hip-common
      clr
      rocblas
      rocwmma
      rocm-device-libs
    ];
    extraEnv = {
      ROCM_PATH = "${pkgs.rocmPackages.clr}";
      HIP_PATH = "${pkgs.rocmPackages.clr}";
      ROCM_DEVICE_LIB_PATH = "${pkgs.rocmPackages.rocm-device-libs}/amdgcn/bitcode";
    };
  };

  ttt-rocm = mkTtt (rocmBuildConfig // {backend = "rocm";});
  ttt-bench-rocm = mkTttBench (rocmBuildConfig // {backend = "rocm";});

  # Expose the cargo artifacts for debugging/caching visibility
  ttt-deps-rocm = mkCargoArtifacts (rocmBuildConfig // {backend = "rocm";});

  cudaPackages = pkgs.cudaPackages_12_8; # Match RunPod's CUDA 12.8.1

  cudaBuildConfig = {
    extraNativeBuildInputs = [cudaPackages.cudatoolkit]; # nvcc needed at build time for cudarc version detection
    extraBuildInputs = [cudaPackages.cudatoolkit];
    extraEnv.CUDA_PATH = "${cudaPackages.cudatoolkit}";
  };

  ttt-cuda = mkTtt (cudaBuildConfig // {backend = "cuda";});
  ttt-bench-cuda = mkTttBench (cudaBuildConfig // {backend = "cuda";});

  # Expose the cargo artifacts for debugging/caching visibility
  ttt-deps-cuda = mkCargoArtifacts (cudaBuildConfig // {backend = "cuda";});

  ttt-rocm-docker = mkTttDocker {
    name = "ttt-rocm";
    tttPkg = ttt-rocm;
    runtimeDeps = with pkgs.rocmPackages;
      [
        clr
        rocblas
        rocwmma
        rocm-device-libs
        rocm-runtime
        ttt-bench-rocm
      ]
      ++ [pkgs.nvtopPackages.amd];
    extraEnv = [
      "ROCM_PATH=${pkgs.rocmPackages.clr}"
      "HIP_PATH=${pkgs.rocmPackages.clr}"
      "ROCM_DEVICE_LIB_PATH=${pkgs.rocmPackages.rocm-device-libs}/amdgcn/bitcode"
    ];
  };

  ttt-cuda-docker = mkTttDocker {
    name = "ttt-cuda";
    tttPkg = ttt-cuda;
    runtimeDeps = with cudaPackages; [cudatoolkit] ++ [pkgs.nvtopPackages.nvidia ttt-bench-cuda];
    extraEnv = [
      "NVIDIA_VISIBLE_DEVICES=all"
      "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
      "CUDA_PATH=${cudaPackages.cudatoolkit}"
    ];
  };
}
