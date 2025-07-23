{ stdenv, fetchFromGitHub }:
let
  version = "";
in
stdenv.mkDerivation {
  name = "unified-runtime";
  inherit version;

  src = fetchFromGitHub {
    owner = "oneapi-src";
    repo = "unified-runtime";
    tag = version;
    sha256 = "";
  };
}
