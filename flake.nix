{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      # self,
      nixpkgs,
      ...
    }:
    let
      systems = [ "x86_64-linux" ];
      forAllSys = nixpkgs.lib.genAttrs systems;
      forAllSysPkgs = f: forAllSys (sys: f nixpkgs.legacyPackages.${sys});
      rocm =
        pkgs:
        pkgs.symlinkJoin {
          name = "rocm-merged";
          paths = with pkgs.rocmPackages; [
            clr
            rocm-core
            rocm-device-libs
            rocm-runtime
            hip-common
          ];
          buildInputs = [ pkgs.makeWrapper ];
          postBuild = ''
            wrapProgram $out/bin/hipcc \
              --add-flags "--rocm-device-lib-path=$out/amdgcn/bitcode"
          '';
        };
      ownPackages = pkgs: import ./nix { inherit pkgs; };
      envPackages =
        pkgs:
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
          (adaptivecpp.override { llvmPackages_18 = llvmPackages; })
        ]
        ++ (with (ownPackages pkgs); [
          kokkos
          kokkos-kernels
          kokkos-tools
        ]);

      syclEnvPackages =
        pkgs:
        envPackages pkgs
        ++ [
          # Additional packages for SYCL development
          pkgs.gdb
          pkgs.valgrind
          pkgs.opencl-headers
          pkgs.clinfo
        ];
    in
    {
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
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${
              pkgs.lib.makeLibraryPath (
                with pkgs;
                [
                  vulkan-loader
                  (rocm pkgs)
                ]
              )
            }
            echo AMONGUS
            "'';
        };

        # SYCL development shell with Intel OneAPI
        sycl-intel =
          let
            intelPkgs = (ownPackages pkgs).intelPackages;
          in
          pkgs.mkShell {
            packages = syclEnvPackages pkgs ++ [
              intelPkgs.intel-oneapi-basekit
              intelPkgs.intel-oneapi-hpckit
            ];
            shellHook = ''
              export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${
                pkgs.lib.makeLibraryPath (
                  with pkgs;
                  [
                    vulkan-loader
                    (rocm pkgs)
                  ]
                )
              }"

              # Intel OneAPI setup
              if [ -f "${intelPkgs.intel-oneapi-basekit}/opt/intel/oneapi/setvars.sh" ]; then
                source ${intelPkgs.intel-oneapi-basekit}/opt/intel/oneapi/setvars.sh --force
              else
                echo "Intel OneAPI basekit not found"
              fi
              if [ -f "${intelPkgs.intel-oneapi-hpckit}/opt/intel/oneapi/setvars.sh" ]; then
                source ${intelPkgs.intel-oneapi-hpckit}/opt/intel/oneapi/setvars.sh --force
              else
                echo "Intel OneAPI hpckit not found"
              fi

              echo "SYCL Intel OneAPI Development Environment"
              echo "========================================"
              echo "Available compilers:"
              echo "  - dpcpp (Intel DPC++)"
              echo "  - icpx (Intel C++ Compiler)"
              echo ""
              echo "To build SYCL project:"
              echo "  cd sycl-test"
              echo "  ./test.sh --intel"
            '';
          };

        # SYCL development shell with AdaptiveCpp
        sycl-adaptive = pkgs.mkShell {
          packages = syclEnvPackages pkgs;
          shellHook = ''
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${
              pkgs.lib.makeLibraryPath (
                with pkgs;
                [
                  vulkan-loader
                  (rocm pkgs)
                ]
              )
            }"

            # AdaptiveCpp setup
            export ACPP_TARGETS="omp"

            echo "SYCL AdaptiveCpp Development Environment"
            echo "======================================="
            echo "Available targets: $ACPP_TARGETS"
            echo ""
            echo "To build SYCL project:"
            echo "  cd sycl-test"
            echo "  ./test.sh --adaptive"
          '';
        };

        # Full SYCL development shell with both implementations
        sycl =
          let
            intelPkgs = (ownPackages pkgs).intelPackages;
          in
          pkgs.mkShell {
            packages = syclEnvPackages pkgs ++ [
              intelPkgs.intel-oneapi-basekit
              intelPkgs.intel-oneapi-hpckit
            ];
            shellHook = ''
              export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${
                pkgs.lib.makeLibraryPath (
                  with pkgs;
                  [
                    vulkan-loader
                    (rocm pkgs)
                  ]
                )
              }"

              # Intel OneAPI setup
              if [ -f "${intelPkgs.intel-oneapi-basekit}/setvars.sh" ]; then
                source ${intelPkgs.intel-oneapi-basekit}/setvars.sh --force
              fi
              if [ -f "${intelPkgs.intel-oneapi-hpckit}/setvars.sh" ]; then
                source ${intelPkgs.intel-oneapi-hpckit}/setvars.sh --force
              fi

              # AdaptiveCpp setup
              export ACPP_TARGETS="omp"

              echo "SYCL Full Development Environment"
              echo "================================="
              echo "Available implementations:"
              echo "  - Intel DPC++ (dpcpp)"
              echo "  - AdaptiveCpp (ACPP_TARGETS=$ACPP_TARGETS)"
              echo ""
              echo "To test both implementations:"
              echo "  cd sycl-test"
              echo "  ./test.sh"
            '';
          };
      });

      packages = forAllSysPkgs (
        pkgs:
        (ownPackages pkgs)
        // {
          ttt-kokkos = pkgs.stdenv.mkDerivation {
            name = "ttt-kokkos";
            src = ./kokkos;
            buildInputs = envPackages pkgs;
          };

          # SYCL test package with Intel DPC++
          sycl-test-intel =
            let
              intelPkgs = (ownPackages pkgs).intelPackages;
            in
            pkgs.stdenv.mkDerivation {
              name = "sycl-test-intel";
              src = ./sycl-test;
              nativeBuildInputs = [
                pkgs.cmake
                pkgs.ninja
              ];
              buildInputs = syclEnvPackages pkgs ++ [
                intelPkgs.intel-oneapi-basekit
                intelPkgs.intel-oneapi-hpckit
              ];

              configurePhase = ''
                # Source Intel OneAPI environment
                if [ -f "${intelPkgs.intel-oneapi-basekit}/setvars.sh" ]; then
                  source ${intelPkgs.intel-oneapi-basekit}/setvars.sh --force
                fi
                if [ -f "${intelPkgs.intel-oneapi-hpckit}/setvars.sh" ]; then
                  source ${intelPkgs.intel-oneapi-hpckit}/setvars.sh --force
                fi

                mkdir -p build
                cd build
                cmake .. -DUSE_INTEL_DPCPP=ON -DCMAKE_BUILD_TYPE=Release
              '';

              buildPhase = ''
                cd build
                make -j$NIX_BUILD_CORES
              '';

              installPhase = ''
                mkdir -p $out/bin
                cp sycl-test $out/bin/
              '';
            };

          # SYCL test package with AdaptiveCpp
          sycl-test-adaptive = pkgs.stdenv.mkDerivation {
            name = "sycl-test-adaptive";
            src = ./sycl-test;
            nativeBuildInputs = [
              pkgs.cmake
              pkgs.ninja
            ];
            buildInputs = syclEnvPackages pkgs;

            preConfigure = ''
              export ACPP_TARGETS="omp"
            '';

            configurePhase = ''
              mkdir -p build
              cd build
              cmake .. -DUSE_ADAPTIVECPP=ON -DCMAKE_BUILD_TYPE=Release
            '';

            buildPhase = ''
              cd build
              make -j$NIX_BUILD_CORES
            '';

            installPhase = ''
              mkdir -p $out/bin
              cp sycl-test $out/bin/
            '';
          };
        }
      );
    };
}
