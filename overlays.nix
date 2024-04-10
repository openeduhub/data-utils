{ lib, nix-filter }:
rec {
  default = its-data;
  python-lib = its-data;

  its-data = (
    final: prev: {
      pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
        (python-final: python-prev: {
          its-data = python-final.callPackage ./python-lib.nix { inherit nix-filter; };
        })
      ];
    }
  );
}
