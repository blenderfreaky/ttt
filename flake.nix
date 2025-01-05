{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs @ {flake-parts, ...}:
    flake-parts.lib.mkFlake {inherit inputs;} {
      flake = {
      };
      systems = ["x86_64-linux"];
      perSystem = {
        config,
        self',
        inputs',
        pkgs,
        system,
        ...
      }: rec {
        devShells.default = devShells.kokkos;

        devShells.kokkos = pkgs.mkShell {
          packages = [
            pkgs.kokkos
          ];
        };

        packages.ttt-kokkos = pkgs.stdenv.mkDerivation {
          name = "kokkos";
          src = ./kokkos;
          buildInputs = [pkgs.cmake pkgs.kokkos];
        };
      };
    };
}
