#!/usr/bin/env bash
# Pipe the toolkit build log through this script to filter out
# which components are missing which dependencies
grep "error: auto-patchelf could not satisfy dependency (.+?) wanted by (.+)" -r '$2: $1' | \
      awk '
      BEGIN { FS = ": "; OFS = ": " }

      # Nearly everything requires libz.so.1, so we filter it out for brevity
      $2 == "libz.so.1" { next }

      # Group dependencies by the OneAPI component they are a part of
      # (like mpi, advisor, compiler, ...)
      {
        if (match($1, /\/opt\/intel\/oneapi\/([^\/]+)/, a)) {
          print a[1], $2
        } else {
          print "other", $2
        }
      }
    ' | \
      sort -u | \
      awk '
      BEGIN { FS = ": " }
      $1 != last_group {
        if (NR > 1) { print "" } # Add space between groups
        print $1 ":"
        last_group = $1
      }
      { print "  - " $2 }
