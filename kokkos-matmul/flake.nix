{
  description = "Kokkos Matrix Multiplication Example";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            cmake
            ninja
            gcc12
            just
          ];
          buildInputs = with pkgs; [
            (kokkos.override {
              rocmSupport = true;
              openmpSupport = true;
              enableShared = true;
            })
          ];
          shellHook = ''
            echo "Kokkos MatMul Development Shell"
          '';
        };

        packages.default = pkgs.stdenv.mkDerivation {
          pname = "kokkos-matmul";
          version = "0.1.0";
          src = ./.;

          nativeBuildInputs = with pkgs; [
            cmake
            ninja
            gcc12
          ];

          buildInputs = with pkgs; [
            (kokkos.override {
              rocmSupport = true;
              openmpSupport = true;
              enableShared = true;
            })
          ];

          configurePhase = ''
            cmake -B build -G Ninja \
              -DCMAKE_BUILD_TYPE=Release \
              -DKokkos_ENABLE_HIP=ON \
              -DKokkos_ARCH_VEGA90A=ON
          '';

          buildPhase = ''
            cmake --build build -v
          '';

          installPhase = ''
            mkdir -p $out/bin
            cp build/matmul $out/bin/
          '';
        };
      });
}