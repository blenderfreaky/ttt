{ pkgs }:
{
  intel-llvm = pkgs.callPackage ./package.nix { };
}
