#!/usr/bin/env bash
set -e
PYTHON=${PYTHON:="python"}

pushd "$(dirname "$0"}"

$PYTHON test_callbacks.py
$PYTHON test_meters.py
