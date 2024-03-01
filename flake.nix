{
  description = "A Python package defined as a Nix Flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";
    nix-filter.url = "github:numtide/nix-filter";
    nlprep = {
      url = "github:openeduhub/nlprep";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    let
      nix-filter = self.inputs.nix-filter.lib;

      ### create the python installation for the package
      python-packages-build = py-pkgs:
        with py-pkgs; [
          setuptools
          pandas
          numpy
          requests
          tqdm
          questionary
          (self.inputs.nlprep.lib.nlprep py-pkgs)
        ];

      python-packages-test = py-pkgs:
        with py-pkgs; [
          hypothesis
        ];

      python-packages-docs = py-pkgs:
        with py-pkgs; [
          sphinx
          sphinx-rtd-theme
        ];

      ### create the python installation for development
      # the development installation contains all build packages,
      # plus some additional ones we do not need to include in production.
      python-packages-devel = py-pkgs:
        with py-pkgs; [
          black
          pyflakes
          isort
          pytest
          pytest-cov
          mypy
          types-requests
          pandas-stubs
        ]
        ++ (python-packages-build py-pkgs)
        ++ (python-packages-test py-pkgs)
        ++ (python-packages-docs py-pkgs);

      ### create the python package
      get-data-utils = py-pkgs: py-pkgs.buildPythonPackage {
        pname = "data_utils";
        version = "0.1.0";
        /*
          only include files that are related to the application
          this will prevent unnecessary rebuilds
          */
        src = nix-filter {
          root = self;
          include = [
            # folders
            "data_utils"
            "test"
            # files
            ./setup.py
            ./requirements.txt
          ];
          exclude = [ (nix-filter.matchExt "pyc") ];
        };
        propagatedBuildInputs = (python-packages-build py-pkgs);
        nativeCheckInputs = [
          py-pkgs.pytestCheckHook
        ] ++ (python-packages-test py-pkgs);
      };

    in
    {
      lib = {
        data-utils = get-data-utils;
      };
      overlays.default = (final: prev: {
        pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
          (python-final: python-prev: {
            data-utils = self.outputs.lib.data-utils python-final;
          })
        ];
      });
    } //
    flake-utils.lib.eachDefaultSystem (system:
      let
        # import the packages from nixpkgs
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3;
      in
      {
        # the packages that we can build
        packages = rec {
          data-utils = self.outputs.lib.data-utils python.pkgs;
          default = data-utils;
          docs = pkgs.runCommand "docs"
            {
              buildInputs = [
                (python-packages-docs python.pkgs)
                (data-utils.override { doCheck = false; })
              ];
            }
            (pkgs.writeShellScript "docs.sh" ''
              sphinx-build -b html ${ ./docs} $out
            '');

        };
        apps = {
          download-data = {
            type = "app";
            program = "${self.outputs.packages.${system}.data-utils}/bin/download-data";
          };
          find-test-data = {
            type = "app";
            program = "${self.outputs.packages.${system}.data-utils}/bin/find-test-data";
          };
        };
        # the development environment
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # the development installation of python
            (python.withPackages python-packages-devel)
            # python lsp server
            pkgs.nodePackages.pyright
            # for automatically generating nix expressions, e.g. from PyPi
            pkgs.nix-template
            pkgs.nix-init
          ];
        };
      }
    );
}

