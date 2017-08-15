#!/usr/bin/env bash
set -e
PYTHON=${PYTHON:="python"}

$PYTHON test_callbacks.py
$PYTHON test_meters.py
