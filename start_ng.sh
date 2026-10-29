#!/usr/bin/env sh
# voice-changer-ng launcher. Extra arguments are passed to the server.
set -e
cd "$(dirname "$0")"
uv sync
cd server
exec uv run --no-sync python MMVCServerSIO.py -p 18888 --https false "$@"
