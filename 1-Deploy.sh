#!/bin/sh
set -e

cd "$(dirname "$0")"

git submodule update --init --recursive

chmod +x "External/Packman/packman.sh" "2-Build.sh" "4-Clean.sh"

cmake -S . -B _Build -G "Ninja Multi-Config" "$@"
