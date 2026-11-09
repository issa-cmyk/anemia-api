#!/usr/bin/env bash
# One command to install and open the demo (macOS and Linux):
#   curl -fsSL https://raw.githubusercontent.com/Meditect/yako-anemia-eye/main/run.sh | bash
# Or, from a copy of the repository:  ./run.sh
# Everything goes into one folder (~/yako-anemia-eye) and a private Python environment;
# nothing is installed system-wide. Run it again to restart the demo.
set -euo pipefail

REPO=https://github.com/Meditect/yako-anemia-eye
DIR="${YAKO_DIR:-$HOME/yako-anemia-eye}"
PORT="${PORT:-8000}"
say() { printf '\n\033[1;32m==> %s\033[0m\n' "$1"; }
die() { printf '\n\033[1;31mError: %s\033[0m\n' "$1" >&2; exit 1; }

# 1. Python 3.10 or later
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then PY="$c"; break; fi
done
[ -n "$PY" ] || die "Python 3.10 or later is needed. Install it from https://www.python.org/downloads/ and run this command again."

# 2. The code: use this folder if the script runs from the repository, otherwise download it
SRC="${BASH_SOURCE[0]:-}"   # empty when the script is piped from curl
if [ -n "$SRC" ] && [ -f "$SRC" ] && [ -f "$(dirname "$SRC")/anemia_eye/pipeline.py" ]; then
  DIR="$(cd "$(dirname "$SRC")" && pwd)"
elif [ -d "$DIR/.git" ] && command -v git >/dev/null 2>&1; then
  say "Updating the code in $DIR"
  git -C "$DIR" pull -q --ff-only || true
elif [ ! -f "$DIR/anemia_eye/pipeline.py" ]; then
  say "Downloading the code into $DIR"
  if command -v git >/dev/null 2>&1; then
    git clone -q "$REPO.git" "$DIR"
  else
    mkdir -p "$DIR"
    curl -fsSL "$REPO/archive/refs/heads/main.tar.gz" | tar -xz -C "$DIR" --strip-components=1
  fi
fi
cd "$DIR"

# 3. Private Python environment and libraries (about 1 GB, a few minutes the first time only)
if [ ! -x .venv/bin/python ]; then
  say "Creating a private Python environment"
  "$PY" -m venv .venv
fi
if ! cmp -s requirements.txt .venv/.installed 2>/dev/null; then
  say "Installing the libraries (first time only, a few minutes)"
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r requirements.txt
  cp requirements.txt .venv/.installed
fi

# 4. Model weights (125 MB, first time only)
.venv/bin/python -c "from anemia_eye import download_weights; download_weights()"

# 5. Start the demo and open the browser once it answers (next free port if this one is taken)
PORT=$(.venv/bin/python -c "
import socket, sys
p = int(sys.argv[1])
while True:
    with socket.socket() as s:
        busy = s.connect_ex(('127.0.0.1', p)) == 0      # something already answers there
    if not busy:
        with socket.socket() as s:
            try: s.bind(('127.0.0.1', p)); break
            except OSError: pass
    p += 1
print(p)" "$PORT")
URL="http://localhost:$PORT"
say "Starting the demo on $URL (press Ctrl+C to stop)"
(
  for _ in $(seq 1 120); do
    if curl -s -o /dev/null "$URL/api/health"; then
      if command -v open >/dev/null 2>&1; then open "$URL"
      elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1
      else echo "Open $URL in your browser"; fi
      break
    fi
    sleep 1
  done
) &
exec .venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port "$PORT"
