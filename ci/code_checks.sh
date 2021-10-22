#!/usr/bin/env bash

# If you change these, also change .circleci/config.yml.
SRC_FILES=(src/ tests/ setup.py)

set -x  # echo commands
set -e  # exit immediately on any error

echo "Source format checking"
flake8 ${SRC_FILES[@]}
black --check ${SRC_FILES}
codespell --skip='*.pyc' ${SRC_FILES[@]}

if [ -x "`which circleci`" ]; then
    circleci config validate
fi

if [ "$skipexpensive" != "true" ]; then
  echo "Type checking"
  pytype ${SRC_FILES[@]}
fi
