{
  description = "SYCL Matrix Multiplication Project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            adaptivecpp
            cmake
            gcc12
            just
            ninja
          ];

          shellHook = ''
            echo "SYCL development environment ready!"
          '';
        };
      }
    );
}