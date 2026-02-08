# Reference implementation benchmark environments (JAX, PyTorch, Triton Kernels)
# Used for comparing against the Burn/CubeCL implementation.
{
  pkgs,
  patchesDir,
}: let
  # ============================================================
  # Source repositories
  # ============================================================

  ttt-lm-jax-src = pkgs.fetchFromGitHub {
    owner = "test-time-training";
    repo = "ttt-lm-jax";
    rev = "6f529b124c7fb5879b33c06926408b15add1d82f";
    hash = "sha256-jBGbvBL8nlFpPFAWI3do9hhxiUTkuX9tQO64GdoiwVc=";
  };

  ttt-lm-pytorch-src = pkgs.fetchFromGitHub {
    owner = "test-time-training";
    repo = "ttt-lm-pytorch";
    rev = "cd831db10c8c9a0f6340f02da5613316a8a92b67";
    hash = "sha256-g+MR4TQHptcfaWlFanBBzY1k0ip2aCHP9AL0a1qGj8k=";
  };

  ttt-lm-kernels-src = pkgs.fetchFromGitHub {
    owner = "test-time-training";
    repo = "ttt-lm-kernels";
    rev = "99851e6dcc44060952a5618f0131a5ca6d7f6519";
    hash = "sha256-+SqsWfQfOBHZVzfKugpZiFvAAU0IlpHQFpQxS04j9Ew=";
  };

  # ============================================================
  # JAX Implementation
  # ============================================================

  jaxPython = pkgs.python312.override {
    self = jaxPython;
    packageOverrides = pyfinal: pyprev: {
      jax = pyfinal.buildPythonPackage rec {
        pname = "jax";
        version = "0.7.1";
        format = "wheel";
        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/83/81/793d78c91b0546b3b1f08e55fdd97437174171cd7d70e46098f1a4d94b7b/jax-0.7.1-py3-none-any.whl";
          hash = "sha256:19nlvwcv4hjcf0sg1sy9wk69325c260z96an2835aijq1rp5fvh5";
        };
        dependencies = with pyfinal; [jaxlib ml-dtypes numpy opt-einsum scipy];
        pythonImportsCheck = ["jax"];
      };

      jaxlib = pyfinal.buildPythonPackage rec {
        pname = "jaxlib";
        version = "0.7.1";
        format = "wheel";
        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/0d/50/e37d02e250f5feb755112ec95b1c012a36d48a99209277267037d100f630/jaxlib-0.7.1-cp312-cp312-manylinux_2_27_x86_64.whl";
          hash = "sha256:1saj79rl23mpbfw78m56pg8k1i0np9fa649pvm029y4paw9x7avl";
        };
        nativeBuildInputs = [pkgs.autoPatchelfHook];
        buildInputs = [(pkgs.lib.getLib pkgs.stdenv.cc.cc)];
        dependencies = with pyfinal; [absl-py flatbuffers ml-dtypes scipy];
        pythonImportsCheck = ["jaxlib"];
      };

      flax = pyprev.flax.overridePythonAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or []) ++ [pyfinal.pythonRelaxDepsHook];
        pythonRelaxDeps = ["jax"];
        doCheck = false;
      });

      optax = pyprev.optax.overridePythonAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or []) ++ [pyfinal.pythonRelaxDepsHook];
        pythonRelaxDeps = ["jax"];
      });

      chex = pyprev.chex.overridePythonAttrs (old: {
        nativeBuildInputs = (old.nativeBuildInputs or []) ++ [pyfinal.pythonRelaxDepsHook];
        pythonRelaxDeps = ["jax"];
      });

      mlxu = pyfinal.buildPythonPackage rec {
        pname = "mlxu";
        version = "0.2.0";
        pyproject = true;
        src = pkgs.fetchPypi {
          inherit pname version;
          hash = "sha256-O85yWb36jjciRxlx5zB43L4uHfVK9HWDkjasBJQhOI8=";
        };
        build-system = with pyfinal; [setuptools];
        dependencies = with pyfinal; [
          absl-py
          ml-collections
          pyyaml
          requests
          gcsfs
          wandb
          cloudpickle
          numpy
        ];
        pythonImportsCheck = ["mlxu"];
      };
    };
  };

  # ROCm 7.1.1 JAX plugin wheels
  jax-rocm7-pjrt = jaxPython.pkgs.buildPythonPackage rec {
    pname = "jax-rocm7-pjrt";
    version = "0.7.1";
    format = "wheel";
    src = pkgs.fetchurl {
      url = "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/jax_rocm7_pjrt-0.7.1-py3-none-manylinux_2_28_x86_64.whl";
      hash = "sha256:0jz15c4379v6r2vp0byx2bz547g7am5336x1acmddgks5vi3437p";
    };
    nativeBuildInputs = [pkgs.autoPatchelfHook];
    buildInputs = [
      pkgs.rocmPackages.clr
      pkgs.rocmPackages.rocm-runtime
      pkgs.rocmPackages.rccl
      pkgs.rocmPackages.hipsolver
      pkgs.rocmPackages.rocsolver
      pkgs.rocmPackages.hipfft
      pkgs.rocmPackages.miopen
      pkgs.rocmPackages.rocprofiler-register
      pkgs.numactl
      (pkgs.lib.getLib pkgs.stdenv.cc.cc)
    ];
    autoPatchelfIgnoreMissingDeps = ["librocprofiler-sdk*.so*" "libhipfftw.so*" "libhipsolver_fortran.so*"];
    dontCheckPythonImports = true;
  };

  jax-rocm7-plugin = jaxPython.pkgs.buildPythonPackage rec {
    pname = "jax-rocm7-plugin";
    version = "0.7.1";
    format = "wheel";
    src = pkgs.fetchurl {
      url = "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.1.1/jax_rocm7_plugin-0.7.1-cp312-cp312-manylinux_2_28_x86_64.whl";
      hash = "sha256:11zcfrqjx48aajb6dqqzgs96s4v1xrs587flkllm5kl2b9kzgwaz";
    };
    nativeBuildInputs = [pkgs.autoPatchelfHook];
    buildInputs = [
      pkgs.rocmPackages.clr
      pkgs.rocmPackages.rocm-runtime
      pkgs.rocmPackages.rccl
      pkgs.rocmPackages.hipblas
      pkgs.rocmPackages.hipsparse
      pkgs.rocmPackages.hipsolver
      pkgs.rocmPackages.rocsolver
      pkgs.rocmPackages.hipfft
      pkgs.rocmPackages.miopen
      pkgs.rocmPackages.rocprofiler-register
      pkgs.numactl
      (pkgs.lib.getLib pkgs.stdenv.cc.cc)
    ];
    autoPatchelfIgnoreMissingDeps = ["librocprofiler-sdk*.so*" "libhipsparselt.so*" "libhipfftw.so*" "libhipsolver_fortran.so*"];
    dependencies = [jax-rocm7-pjrt];
    dontCheckPythonImports = true;
  };

  # Stub libraries for missing ROCm components
  rocm-stubs = pkgs.stdenv.mkDerivation {
    pname = "rocm-stubs";
    version = "0.0.1";
    dontUnpack = true;
    buildPhase = ''
      mkdir -p $out/lib
      cat > stub.c << 'EOF'
      #include <stdint.h>
      int rocprofiler_assign_callback_thread(void* a, void* b) { return -1; }
      int rocprofiler_configure_buffer_tracing_service(void* a, void* b, void* c) { return -1; }
      int rocprofiler_configure_callback_tracing_service(void* a, void* b, void* c, void* d) { return -1; }
      int rocprofiler_context_is_valid(void* a) { return 0; }
      int rocprofiler_create_buffer(void* a, void* b, void* c, void* d, void* e, void* f) { return -1; }
      int rocprofiler_create_callback_thread(void* a) { return -1; }
      int rocprofiler_create_context(void* a) { return -1; }
      int rocprofiler_force_configure(void* a) { return -1; }
      const char* rocprofiler_get_status_string(int s) { return "stub"; }
      int rocprofiler_get_timestamp(uint64_t* ts) { if(ts) *ts = 0; return 0; }
      int rocprofiler_iterate_callback_tracing_kind_operations(void* a, void* b, void* c) { return 0; }
      int rocprofiler_iterate_callback_tracing_kinds(void* a, void* b) { return 0; }
      int rocprofiler_query_available_agents(void* a, void* b, void* c, void* d) { return 0; }
      int rocprofiler_query_callback_tracing_kind_name(void* a, void* b, void* c) { return -1; }
      int rocprofiler_query_callback_tracing_kind_operation_name(void* a, void* b, void* c, void* d) { return -1; }
      int rocprofiler_start_context(void* a) { return -1; }
      int rocprofiler_stop_context(void* a) { return -1; }
      void roctxRangePushA(const char* s) {}
      int roctxRangePop() { return 0; }
      void roctxMarkA(const char* s) {}
      EOF
      $CC -shared -fPIC stub.c -o $out/lib/librocprofiler-sdk-attach.so.1
      $CC -shared -fPIC stub.c -o $out/lib/librocprofiler-sdk.so.1
      $CC -shared -fPIC stub.c -o $out/lib/librocprofiler-sdk-roctx.so.1
      $CC -shared -fPIC stub.c -o $out/lib/librocprofiler-sdk-rocpd.so.1
      $CC -shared -fPIC stub.c -o $out/lib/libhipfftw.so
      $CC -shared -fPIC stub.c -o $out/lib/libhipsolver_fortran.so.1
      $CC -shared -fPIC stub.c -o $out/lib/libhipsparselt.so
      ln -s librocprofiler-sdk-attach.so.1 $out/lib/librocprofiler-sdk-attach.so
      ln -s librocprofiler-sdk.so.1 $out/lib/librocprofiler-sdk.so
      ln -s librocprofiler-sdk-rocpd.so.1 $out/lib/librocprofiler-sdk-rocpd.so
      ln -s libhipsolver_fortran.so.1 $out/lib/libhipsolver_fortran.so
    '';
  };

  # ============================================================
  # Patched sources
  # ============================================================

  ttt-lm-jax-patched = pkgs.stdenv.mkDerivation {
    pname = "ttt-lm-jax-patched";
    version = "0.0.1";
    src = ttt-lm-jax-src;
    patches = [
      "${patchesDir}/jax-utils-api.patch"
      "${patchesDir}/jax-bfloat16-scan-carry.patch"
    ];
    installPhase = ''
      mkdir -p $out
      cp -r . $out/
      cp ${patchesDir}/jax-benchmark.py $out/benchmark.py
    '';
  };

  ttt-lm-pytorch-patched = pkgs.stdenv.mkDerivation {
    pname = "ttt-lm-pytorch-patched";
    version = "0.0.1";
    src = ttt-lm-pytorch-src;
    installPhase = ''
      mkdir -p $out
      cp -r . $out/
      cp ${patchesDir}/pytorch-benchmark.py $out/benchmark.py
    '';
  };

  ttt-lm-kernels-patched = pkgs.stdenv.mkDerivation {
    pname = "ttt-lm-kernels-patched";
    version = "0.0.1";
    src = ttt-lm-kernels-src;
    patches = [
      "${patchesDir}/kernels-triton-rocm.patch"
      "${patchesDir}/kernels-all-sizes.patch"
    ];
    installPhase = ''
      mkdir -p $out
      cp -r . $out/
      cp ${patchesDir}/kernels-benchmark.py $out/benchmark.py
    '';
  };

  # ============================================================
  # Python environments
  # ============================================================

  jaxPythonEnv = jaxPython.withPackages (ps:
    with ps; [
      jax
      jaxlib
      jax-rocm7-plugin
      numpy
      matplotlib
      tqdm
      transformers
      datasets
      einops
      scipy
      ml-collections
      mlxu
      flax
      optax
      torch
    ]);

  pytorchPythonEnv = pkgs.pkgsRocm.python313.withPackages (ps:
    with ps; [
      torch
      transformers
      numpy
      tqdm
      einops
      causal-conv1d
    ]);

  kernelsPythonEnv = pkgs.pkgsRocm.python313.withPackages (ps:
    with ps; [
      torch
      triton
      transformers
      numpy
      tqdm
      einops
      causal-conv1d
    ]);

  # ============================================================
  # Common ROCm shell hook
  # ============================================================

  rocmShellHook = ''
    # RX 6800 is gfx1030 (RDNA2) - set to 10.3.0
    export HSA_OVERRIDE_GFX_VERSION=10.3.0

    # JAX ROCm needs ROCM_PATH and lld
    export ROCM_PATH="${pkgs.rocmPackages.clr}"
    export PATH="${pkgs.llvmPackages.lld}/bin:$PATH"
    export TRITON_HIP_LLD_PATH="${pkgs.llvmPackages.lld}/bin/ld.lld"

    # ROCm library paths + stubs
    export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
      rocm-stubs
      pkgs.rocmPackages.clr
      pkgs.rocmPackages.rccl
      pkgs.rocmPackages.miopen
      pkgs.rocmPackages.hipblas
      pkgs.rocmPackages.hipfft
      pkgs.rocmPackages.hipsolver
      pkgs.rocmPackages.hipsparse
      pkgs.rocmPackages.rocsolver
    ]}:''${LD_LIBRARY_PATH:-}"
  '';
in {
  devShells = {
    bench-jax = pkgs.mkShell {
      buildInputs = [
        jaxPythonEnv
        pkgs.rocmPackages.clr
        pkgs.rocmPackages.rocm-smi
        pkgs.llvmPackages.lld
      ];
      shellHook =
        rocmShellHook
        + ''
          export PYTHONPATH="${ttt-lm-jax-patched}:$PYTHONPATH"
          cd ${ttt-lm-jax-patched}

          echo ""
          echo "TTT-LM JAX (ROCm) Benchmark Shell"
          echo "=================================="
          echo ""
          echo "JAX version: $(python -c 'import jax; print(jax.__version__)')"
          echo "JAX devices: $(python -c 'import jax; print(jax.devices())')"
          echo ""
          echo "Run benchmark:"
          echo "  python benchmark.py --model-size 125m --ttt-type linear"
        '';
    };

    bench-pytorch = pkgs.mkShell {
      buildInputs = [
        pytorchPythonEnv
        pkgs.rocmPackages.clr
        pkgs.rocmPackages.rocm-smi
      ];
      shellHook =
        rocmShellHook
        + ''
          export PYTHONPATH="${ttt-lm-pytorch-patched}:$PYTHONPATH"
          cd ${ttt-lm-pytorch-patched}

          echo ""
          echo "TTT-LM PyTorch (ROCm) Benchmark Shell"
          echo "======================================"
          echo ""
          echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
          echo "ROCm available: $(python -c 'import torch; print(torch.cuda.is_available())')"
          echo ""
          echo "Run benchmark:"
          echo "  python benchmark.py --model-size 125m --ttt-type linear"
        '';
    };

    bench-kernels = pkgs.mkShell {
      buildInputs = [
        kernelsPythonEnv
        pkgs.rocmPackages.clr
        pkgs.rocmPackages.rocm-smi
        pkgs.llvmPackages.lld
      ];
      shellHook =
        rocmShellHook
        + ''
          export PYTHONPATH="${ttt-lm-kernels-patched}:$PYTHONPATH"
          cd ${ttt-lm-kernels-patched}

          echo ""
          echo "TTT-LM Kernels (ROCm) Benchmark Shell"
          echo "======================================"
          echo ""
          echo "NOTE: Triton kernels patched for ROCm (fp32 casts for sqrt/sigmoid)"
          echo ""
          echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
          echo "Triton: $(python -c 'import triton; print(triton.__version__)' 2>/dev/null || echo 'N/A')"
          echo "ROCm available: $(python -c 'import torch; print(torch.cuda.is_available())')"
          echo ""
          echo "Run benchmark:"
          echo "  python benchmark.py --model-size 1b --ttt-type linear"
          echo "  python benchmark.py --model-size 1b --ttt-type linear --fast"
        '';
    };
  };
}
