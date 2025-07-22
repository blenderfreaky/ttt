{ pkgs }:
rec {
  onednn = pkgs.callPackage ./dnn.nix { inherit onetbb; };
  onetbb = pkgs.callPackage ./tbb.nix { };

  acppStdenv = pkgs.overrideCC pkgs.stdenv pkgs.adaptivecpp;
}
