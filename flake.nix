{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {
    # self,
    nixpkgs,
    ...
  }: let
    systems = ["x86_64-linux"];
    forAllSys = nixpkgs.lib.genAttrs systems;
    forAllSysPkgs = f: forAllSys (sys: f nixpkgs.legacyPackages.${sys});
    rocm = pkgs:
      pkgs.symlinkJoin {
        name = "rocm-merged";
        paths = with pkgs.rocmPackages; [
          clr
          rocm-core
          rocm-device-libs
          rocm-runtime
          hip-common
        ];
        buildInputs = [pkgs.makeWrapper];
        postBuild = ''
          wrapProgram $out/bin/hipcc \
            --add-flags "--rocm-device-lib-path=$out/amdgcn/bitcode"
        '';
      };
    ownPackages = pkgs: import ./nix {inherit pkgs;};
    envPackages = pkgs:
      with pkgs;
        [
          (rocm pkgs)
          cmake
          ninja
          openssl
          pkg-config
          clang
          clang-tools
          clang
          libclang
          libclang.lib
          just
          llvmPackages.openmp
          gcc
          (adaptivecpp.override {llvmPackages_18 = llvmPackages;})
        ]
        ++ builtins.attrValues (ownPackages pkgs);
  in {
    nixpkgs.config = {
      rocmSupport = true;
      # Most of the intel stuff is closed source
      allowUnfree = true;
    };
    devShells = forAllSysPkgs (pkgs: {
      default = pkgs.mkShell {
        packages = envPackages pkgs;
        shellHook = ''
          # ROCm PR seems to set these to '-parallel-jobs=1' somehow, which breaks builds. I don't know why.
          # export CFLAGS=$CFLAGS -D__HIP_PLATFORM_AMD__
          # export CXXFLAGS=$CXXFLAGS -D__HIP_PLATFORM_AMD__
          export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.lib.makeLibraryPath (with pkgs; [
            vulkan-loader
            (rocm pkgs)
          ])}
          echo AMONGUS
          "'';
      };
    });

    packages = forAllSysPkgs (pkgs:
      (ownPackages pkgs)
      // {
        ttt-kokkos = pkgs.stdenv.mkDerivation {
          name = "ttt-kokkos";
          src = ./kokkos;
          buildInputs = envPackages pkgs;
        };
      });
  };
}
