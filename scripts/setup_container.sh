#!/bin/bash

# This script installs dependencies.
# It's intended to run in a container prior to running tensorflow

set -e

# opencv dependencies
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

