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
          packages = with pkgs; with pkgs.rocmPackages; [cmake kokkos clr hip-common];
        };

        packages.ttt-kokkos = pkgs.stdenv.mkDerivation {
          name = "kokkos";
          src = ./kokkos;
          buildInputs = with pkgs; with pkgs.rocmPackages; [cmake kokkos clr hip-common];
        };
      };
    };
}
