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
            llvmPackages.openmp
          ];

          nativeBuildInputs = with pkgs; [
            pkg-config
          ];

          shellHook = ''
            export C_INCLUDE_PATH="${pkgs.llvmPackages_16.openmp}/include:$C_INCLUDE_PATH"
            export CPLUS_INCLUDE_PATH="${pkgs.llvmPackages_16.openmp}/include:$CPLUS_INCLUDE_PATH"
            export LIBRARY_PATH="${pkgs.llvmPackages_16.openmp}/lib:$LIBRARY_PATH"
            export LD_LIBRARY_PATH="${pkgs.llvmPackages_16.openmp}/lib:$LD_LIBRARY_PATH"
            echo "SYCL development environment ready!"
          '';
        };
      }
    );
}
