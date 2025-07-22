{ pkgs }:
rec {
  llvm = pkgs.callPackage ./llvm { };

  # Expose the Intel LLVM package directly
  intel-llvm = llvm.intel-llvm;

  # Convenience aliases
  dpcpp = intel-llvm;
  intel-clang = intel-llvm;
}
