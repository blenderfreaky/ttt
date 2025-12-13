{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # nixpkgs-intel.url = "github:blenderfreaky/nixpkgs/package/intel-oneapi";
    intel-nix = {
      # url = "github:blenderfreaky/intel-nix/main";
      url = "path:/home/blenderfreaky/src/projects/intel-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    # nixpkgs-intel,
    intel-nix,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        intel-pkgs-original = intel-nix.packages.${system};
        intel-pkgs = {
          intel-llvm = intel-pkgs-original.src.llvm;
          oneDNN = intel-pkgs-original.src.oneDNN;
          oneMath = intel-pkgs-original.src.oneMath-rocm;
          intel-oneapi-basekit = intel-pkgs-original.toolkits.installer.base;
          intel-oneapi-hpckit = intel-pkgs-original.toolkits.installer.hpc;
        };
        pkgs =
          import
          nixpkgs
          {
            inherit system;
            nixpkgs.config = {
              rocmSupport = true;
              allowUnfree = true;
            };
            overlays = [
              (final: prev: {
                ccacheStdenv = prev.ccacheStdenv.override {
                  extraConfig = ''
                    export CCACHE_COMPRESS=1
                    #export CCACHE_DIR="$ {config.programs.ccache.cacheDir}"
                    export CCACHE_DIR="/var/cache/ccache"
                    export CCACHE_UMASK=007
                    export CCACHE_SLOPPINESS=random_seed
                    if [ ! -d "$CCACHE_DIR" ]; then
                      echo "====="
                      echo "Directory '$CCACHE_DIR' does not exist"
                      echo "Please create it with:"
                      echo "  sudo mkdir -m0770 '$CCACHE_DIR'"
                      echo "  sudo chown root:nixbld '$CCACHE_DIR'"
                      echo "====="
                      exit 1
                    fi
                    if [ ! -w "$CCACHE_DIR" ]; then
                      echo "====="
                      echo "Directory '$CCACHE_DIR' is not accessible for user $(whoami)"
                      echo "Please verify its access permissions"
                      echo "====="
                      exit 1
                    fi
                  '';
                };
              })
              (_: _: intel-pkgs)
            ];
          };
        own-pkgs = pkgs.callPackage ./nix {};
        # rocmMerged = pkgs:
        #   pkgs.symlinkJoin {
        #     name = "rocm-merged";
        #     paths = with pkgs.rocmPackages; [
        #       clr
        #       rocm-core
        #       rocm-device-libs
        #       rocm-runtime
        #       hip-common
        #     ];
        #     buildInputs = [pkgs.makeWrapper];
        #     postBuild = ''
        #       wrapProgram $out/bin/hipcc \
        #         --add-flags "--rocm-device-lib-path=$out/amdgcn/bitcode"
        #     '';
        #   };
      in {
        packages = own-pkgs // intel-pkgs;

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs;
            [
              stdenv.cc.cc.lib

              intel-llvm
              # oneDNN
              oneMath

              # spirv-tools
              lldb
              gdb

              stdenv.cc.cc.lib

              adaptivecpp

              cmake
              ninja

              just
              just-formatter

              # Wrap clangd to query the Intel LLVM compiler for include paths
              (writeShellScriptBin "clangd" ''
                exec ${clang-tools}/bin/clangd --query-driver="${intel-llvm}/bin/clang++" "$@"
              '')

              (python3.withPackages
                (ps:
                  with ps; [
                    numpy
                    torch
                    transformers
                    # causal-conv1d
                    safetensors
                  ]))
            ]
            ++ [
              # Memory profiling tools
              heaptrack
              valgrind
              perf
              hotspot
            ]
            ++ (with pkgs.rocmPackages; [
              hip-common
              clr
              rocblas
              rocwmma
              rocm-smi
              rocprofiler
            ]);

          LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.rocmPackages.clr}/lib";

          ROCM_PATH = "${pkgs.rocmPackages.clr}/bin";
          ROCM_DEVICE_LIB_PATH = "${pkgs.rocmPackages.rocm-device-libs}/amdgcn/bitcode";
          INTEL_LLVM_DIR = "${pkgs.intel-llvm}";
        };
      }
    );
}
