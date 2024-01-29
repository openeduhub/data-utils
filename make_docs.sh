#!/bin/sh
git checkout gh-pages
git rebase origin/main
mkdir html
nix build .\#docs --out-link docs-result &&
    cp -rf $(readlink -f docs-result)/* html &&
    rm docs-result &&
    chmod -R 755 html
