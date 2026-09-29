#!/usr/bin/env bash
# Placeholder: produce a fresh /etc/kite-bot/access_token.txt for today.
# Kite tokens expire daily and login needs 2FA. Implement this with your own login flow
# (for example the scripts in the repository root) and check Zerodha's terms before automating it.
set -euo pipefail
test -s /etc/kite-bot/access_token.txt || { echo "no access_token.txt: refresh it first" >&2; exit 1; }
