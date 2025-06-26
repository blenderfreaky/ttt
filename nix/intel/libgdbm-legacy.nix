{
  lib,
  gdbm,
  stdenvNoCC,
}:
stdenvNoCC.mkDerivation {
  pname = "libgdbm-legacy";
  inherit (gdbm) version;

  dontUnpack = true;

  installPhase = ''
    runHook preInstall

    mkdir -p $out/lib

    cp ${lib.getLib gdbm}/lib/libgdbm_compat.so.4 $out/lib/libgdbm.so.4
    ln -s $out/lib/libgdbm.so{.4,}

    runHook postInstall
  '';
}
