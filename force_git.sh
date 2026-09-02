#!/bin/sh
# Hard-sync this checkout to the remote state of the branch it is ON.
# It used to reset to origin/master unconditionally, which silently kept the checkout on
# stale code whenever the work was on another branch.
# Discards local modifications to tracked files and untracked files, but NOT ignored ones
# -- results/ is gitignored, so simulation output survives. Do not add -x to the clean.
BRANCH=$(git rev-parse --abbrev-ref HEAD)
git fetch --all
git reset --hard "origin/$BRANCH"
git clean -fd
