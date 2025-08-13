{ pkgs }:
rec {
  llvm = pkgs.callPackage ./llvm { inherit unified-runtime; };
  unified-runtime = pkgs.callPackage ./unified-runtime.nix { inherit unified-memory-framework; };
  unified-memory-framework = pkgs.callPackage ./unified-memory-framework.nix { };

  # Expose the Intel LLVM package directly
  intel-llvm = llvm.intel-llvm;

  oneapi-construction-kit = pkgs.callPackage ./oneapi-construction-kit.nix { };

  # Convenience aliases
  dpcpp = intel-llvm;
  intel-clang = intel-llvm;
}
