{
  lib,
  nix-filter,
  buildPythonPackage,
  setuptools,
  pandas,
  numpy,
  requests,
  tqdm,
  questionary,
  its-prep,
  scikit-learn,
  pytestCheckHook,
  hypothesis,
  sphinx,
  sphinx-rtd-theme,
  withSphinx ? false,
}:
buildPythonPackage {
  pname = "its-data";
  version = "0.1.0";
  format = "setuptools";

  # only include files that are related to the application.
  # this will prevent unnecessary rebuilds
  src = nix-filter {
    root = ./.;
    include = [
      # folders
      "its_data"
      "test"
      # files
      ./setup.py
      ./requirements.txt
    ];
    exclude = [ (nix-filter.matchExt "pyc") ];
  };

  propagatedBuildInputs =
    [
      setuptools
      pandas
      numpy
      requests
      tqdm
      questionary
      its-prep
      scikit-learn
    ]
    ++ (lib.optionals withSphinx [
      sphinx
      sphinx-rtd-theme
    ]);

  nativeCheckInputs = [
    pytestCheckHook
    hypothesis
  ];
}
