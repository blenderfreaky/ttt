{
  inputs = {
    #nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    #nixpkgs.url = "/home/blenderfreaky/src/stuff/nixpkgs";
    nixpkgs.url = "github:LunNova/nixpkgs/rocm-update";

    #flake-parts.url = "github:hercules-ci/flake-parts";
    flakelight.url = "github:nix-community/flakelight";
  };

  outputs = {flakelight, ...}:
  #flake-parts.lib.mkFlake {inherit inputs;} {
  #  flake = {
  #  };
  #  systems = ["x86_64-linux"];
  #  perSystem = {
  #    config,
  #    self',
  #    inputs',
  #    system,
  #    ...
  #  }: let
    flakelight ./. (let
      rocm = pkgs:
        pkgs.symlinkJoin {
          name = "rocm-merged";
          paths = with pkgs.rocmPackages; [
            clr
            rocm-core
            rocm-device-libs
            rocm-runtime
          ];
          buildInputs = [pkgs.makeWrapper];
          postBuild = ''
            wrapProgram $out/bin/hipcc \
              --add-flags "--rocm-device-lib-path=$out/amdgcn/bitcode"
          '';
        };
    in {
      nixpkgs.config = {
        rocmSupport = true;
      };
      devShells.default = pkgs:
        pkgs.mkShell {
          packages = with pkgs; [(rocm pkgs) cmake ninja kokkos];
        };

      #devShells = {
      #  kokkos = pkgs.mkShell {
      #    packages = with pkgs; with pkgs.rocmPackages; [cmake kokkos clr hip-common];
      #  };
      #};

      packages = pkgs:
        (import ./nix {inherit pkgs;})
        // {
          ttt-kokkos = pkgs.stdenv.mkDerivation {
            name = "kokkos";
            src = ./kokkos;
            buildInputs = with pkgs; [cmake kokkos (rocm pkgs)];
          };
        };
    });
}
