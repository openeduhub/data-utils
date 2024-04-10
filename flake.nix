{
  description = "A Python package defined as a Nix Flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nix-filter.url = "github:numtide/nix-filter";
    its-prep.url = "github:openeduhub/its-prep";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }:
    {
      overlays = import ./overlays.nix {
        inherit (nixpkgs) lib;
        nix-filter = self.inputs.nix-filter.lib;
      };
    }
    // flake-utils.lib.eachDefaultSystem (
      system:
      let
        # import the packages from nixpkgs, adding additional dependencies and
        # its-data
        pkgs =
          (nixpkgs.legacyPackages.${system}.extend self.inputs.its-prep.overlays.default).extend
            self.outputs.overlays.default;
      in
      {
        # the packages that we can build
        packages = rec {
          inherit (pkgs.python3Packages) its-data;
          default = its-data;
          docs =
            pkgs.runCommand "docs"
              {
                buildInputs = [
                  ((its-data.override { withSphinx = true; }).overridePythonAttrs { doCheck = false; })
                ];
              }
              (
                pkgs.writeShellScript "docs.sh" ''
                  sphinx-build -b html ${./docs} $out
                ''
              );
        };
        apps = {
          download-data = {
            type = "app";
            program = "${pkgs.python3Packages.its-data}/bin/download-data";
          };
          publish-data = {
            type = "app";
            program = "${pkgs.python3Packages.its-data}/bin/publish-data";
          };
          find-test-data = {
            type = "app";
            program = "${pkgs.python3Packages.its-data}/bin/find-test-data";
          };
        };
        # the development environment
        devShells.default = pkgs.callPackage ./shell.nix { };
      }
    );
}
