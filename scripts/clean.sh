#!/bin/bash

SRC_DIR="./src"
echo "Running isort ..."
isort $SRC_DIR --profile black --line-length 120
echo "Running black ..."
black --target-version py39 --line-length 120 $SRC_DIR
echo "Running autoflake ..."
autoflake --in-place --recursive --remove-all-unused-imports --expand-star-imports --remove-unused-variables $SRC_DIR
echo "Running flake8 ..."
flake8 --max-line-length 120 $SRC_DIR
