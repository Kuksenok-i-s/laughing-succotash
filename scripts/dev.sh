#!/usr/bin/env bash
# Run a command against the project virtualenv with a clean environment.
#
# The Cursor AppImage exports ARGV0, APPDIR and LD_LIBRARY_PATH into the integrated terminal.
# zsh honours ARGV0 when exec'ing, so the venv interpreter is launched with the AppImage as its
# argv[0] and then fails to locate its own stdlib. Stripping those makes the venv behave normally.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec env -u ARGV0 -u APPDIR -u APPIMAGE -u LD_LIBRARY_PATH -u OWD \
    PYTHONPATH="${ROOT}/agent-core:${ROOT}/telegram-gateway:${ROOT}/gpu-transcriber${PYTHONPATH:+:${PYTHONPATH}}" \
    "${ROOT}/.venv/bin/python" "$@"
