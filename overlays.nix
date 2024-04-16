{ lib, nix-filter, its-prep-overlay }:
rec {
  default = its-data;
  python-lib = its-data;

  its-data = lib.composeExtensions its-prep-overlay (
    final: prev: {
      pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
        (python-final: python-prev: {
          its-data = python-final.callPackage ./python-lib.nix { inherit nix-filter; };
        })
      ];
    }
  );
}
