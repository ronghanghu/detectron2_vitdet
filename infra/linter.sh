#!/bin/bash -ev

# Run this script at project root by "./infra/linter.sh" before you commit

isort -y -sp . --skip datasets --atomic
black -l 100 .
flake8 .

command -v arc > /dev/null && arc lint
