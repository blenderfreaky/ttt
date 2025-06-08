{
  description = "Kokkos Matrix Multiplication Example";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        packages.default = pkgs.stdenv.mkDerivation {
          pname = "kokkos-matmul";
          version = "0.1.0";
          src = ./.;

          nativeBuildInputs = with pkgs; [
            cmake
            ninja
            rocmPackages.hipcc
            rocmPackages.rocm-cmake
          ];

          buildInputs = with pkgs; [
            kokkos
            rocmPackages.clr
            rocmPackages.rocthrust
            rocmPackages.rocm-runtime
          ];

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=Release"
            "-DCMAKE_CXX_COMPILER=${pkgs.rocmPackages.hipcc}/bin/hipcc"
            "-DKokkos_ENABLE_HIP=ON"
            "-DKokkos_ENABLE_OPENMP=ON"
            "-DKokkos_ARCH_VEGA906=ON"
            "-DCMAKE_PREFIX_PATH=${pkgs.rocmPackages.rocthrust}/include"
          ];

          preConfigure = ''
            export ROCM_PATH=${pkgs.rocmPackages.clr}
            export HIP_PLATFORM=amd
            export ROCTHRUST_PATH=${pkgs.rocmPackages.rocthrust}
          '';

          installPhase = ''
            mkdir -p $out/bin
            cp matmul $out/bin/
          '';
        };

        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            cmake
            ninja
            gcc12
            rocmPackages.clr
            rocmPackages.hipcc
            rocmPackages.rocm-cmake
            rocmPackages.rocthrust
            just
            stdenv
          ];

          buildInputs = with pkgs; [
            kokkos
            rocmPackages.rocm-runtime
          ];

          shellHook = ''
            export CXX=${pkgs.rocmPackages.hipcc}/bin/hipcc
            export ROCM_PATH=${pkgs.rocmPackages.clr}
            export HIP_PLATFORM=amd
            export ROCTHRUST_PATH=${pkgs.rocmPackages.rocthrust}
            echo "Kokkos MatMul Development Shell"
          '';
        };
      });
}
