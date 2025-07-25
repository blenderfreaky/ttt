{ pkgs }:
rec {
  llvm = pkgs.callPackage ./llvm { inherit unified-runtime; };
  unified-runtime = pkgs.callPackage ./unified-runtime.nix { inherit unified-memory-framework; };
  unified-memory-framework = pkgs.callPackage ./unified-memory-framework.nix { };

  # Expose the Intel LLVM package directly
  intel-llvm = llvm.intel-llvm;

  # Convenience aliases
  dpcpp = intel-llvm;
  intel-clang = intel-llvm;
}
