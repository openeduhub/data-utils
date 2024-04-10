{
  lib,
  stdenv,
  mkShell,
  python3,
  pyright,
  nix-template,
  nix-init,
  nix-tree
}:
mkShell {
  packages = [
    (python3.withPackages (
      py-pkgs:
      with py-pkgs;
      [
        black
        pyflakes
        isort
        pytest
        pytest-cov
        mypy
        types-requests
      ]
      # pandas-stubs appears to be broken on darwin
      ++ (lib.optionals (!stdenv.isDarwin) [ py-pkgs.pandas-stubs ])
      ++ (py-pkgs.its-data.override { withSphinx = true; }).propagatedBuildInputs
    ))
    pyright
    nix-template
    nix-init
    nix-tree
  ];
}
