#!/usr/bin/env bash
# Run every suite: the shared protocol, both deploy units, and the integration tests.
#
# Each unit is a separate pytest run because both have a top-level `tests` package; that is a
# consequence of them being independently deployable, not an accident worth working around.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEV="${ROOT}/scripts/dev.sh"

status=0
for target in "packages/pa-protocol/tests" "agent-core" "telegram-gateway" "tests"; do
    echo
    echo "=== ${target} ==="
    "${DEV}" -m pytest "${ROOT}/${target}" "$@" || status=$?
done

exit "${status}"
