#!/bin/bash -ev

# Run this script at project root by "./infra/linter.sh" before you commit

echo "Running isort ..."
isort -y -sp . --skip datasets --atomic

echo "Running black ..."
black -l 100 .

echo "Running flake8 ..."
if [ -x "$(command -v flake8-3)" ]; then
  flake8-3 .
else
  python3 -m flake8 .
fi

command -v arc > /dev/null && arc lint
