{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-rocm.url = "github:NixOS/nixpkgs/nixos-25.11"; # ROCm 6.x for RunPod
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
  };

  nixConfig = {
    extra-substituters = ["https://cache.nixos-cuda.org"];
    extra-trusted-public-keys = ["cache.nixos-cuda.org:74DUi4Ye579gUqzH4ziL9IyiJBlDpMRn9MBN8oNan9M="];
  };

  outputs = {
    self,
    nixpkgs,
    nixpkgs-rocm,
    flake-utils,
    crane,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            rocmSupport = true;
            allowUnfree = true;
          };
          overlays = [
            (final: prev: {
              rocmPackages = prev.rocmPackages // {
                rocprofiler = prev.rocmPackages.rocprofiler.overrideAttrs (oldAttrs: {
                  postInstall = oldAttrs.postInstall + ''
                    substituteInPlace $out/bin/rocprof --replace-fail '/bin/ls' '${final.coreutils}/bin/ls'
                  '';
                });
              };
            })
          ];
        };
        pkgs-rocm = import nixpkgs-rocm {
          inherit system;
          config = {
            rocmSupport = true;
            allowUnfree = true;
          };
        };
        craneLib = crane.mkLib pkgs;
        own-pkgs = pkgs.callPackage ./nix {inherit craneLib pkgs-rocm;};
        embeddingPython = pkgs.python3.withPackages (ps: with ps; [tokenizers datasets]);
      in {
        legacyPackages = own-pkgs // {inherit (pkgs.rocmPackages) rocprofiler;};

        apps.embedding = {
          type = "app";
          program = "${pkgs.writeShellScript "run-embedding" ''
            ${embeddingPython}/bin/python ${./ttt-embedding/emb.py}
          ''}";
        };

        devShells.default = pkgs.mkShell {
          shellHook = ''
            if command -v ttt &> /dev/null; then
              case "$SHELL" in
                */zsh)  eval "$(ttt completions zsh)" ;;
                */bash) eval "$(ttt completions bash)" ;;
                */fish) ttt completions fish | source ;;
              esac
            fi
          '';
          buildInputs = with pkgs; [
            stdenv.cc.cc.lib
            sqlite
            lldb
            gdb
            adaptivecpp
            cmake
            ninja
            just
            just-formatter
            cargo-nextest
            (python3.withPackages (ps: with ps; [numpy torch transformers safetensors]))
            heaptrack
            valgrind
            perf
            hotspot
          ] ++ (with rocmPackages; [hip-common clr rocblas rocwmma rocm-smi rocprofiler]);

          LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.rocmPackages.clr}/lib";
          ROCM_PATH = "${pkgs.rocmPackages.clr}/bin";
          ROCM_DEVICE_LIB_PATH = "${pkgs.rocmPackages.rocm-device-libs}/amdgcn/bitcode";
        };

        devShells.embedding = pkgs.mkShell {
          packages = [embeddingPython];
        };
      }
    );
}
