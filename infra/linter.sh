#!/bin/bash -e

# Run this script at project root by "./infra/linter.sh" before you commit

isort -y -sp .flake8
black -l 100 .
flake8 .
