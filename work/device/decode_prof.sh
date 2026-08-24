#!/usr/bin/env bash
# Decode qnn-net-run's binary profiling log into per-inference microseconds.
#
# trap #43: report LOAD and EXECUTE separately.  "QNN (load binary) time" is a one-off
# context load; "Accelerator (execute excluding wait) time" is the number to compare
# against a published per-inference latency.
# trap #10: the DSP throttles, so the headline number is the MINIMUM over inferences.
set -uo pipefail
source "$HOME/npuconvert/qnn_env.sh"
FF=${FF:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

for m in "$@"; do
  LOG=$FF/work/device/prof/$m/prof.log
  [ -f "$LOG" ] || { printf '%-12s no profiling log\n' "$m"; continue; }
  OUT=$FF/work/device/prof/$m/prof.csv
  qnn-profile-viewer --input_log "$LOG" --output "$OUT" >/dev/null 2>&1
  [ -f "$OUT" ] || { printf '%-12s profile-viewer produced nothing\n' "$m"; continue; }
  python - "$m" "$OUT" <<'PY'
import csv, sys
name, path = sys.argv[1], sys.argv[2]
# columns: Msg Timestamp, Message, Time, Unit, Timing Source, Event Level, Event Identifier
rows = [r for r in csv.reader(open(path, newline='', errors='replace')) if len(r) >= 7]
def collect(msg, ident):
    out = []
    for r in rows:
        if r[1].strip() == msg and r[6].strip() == ident and r[3].strip() == 'US':
            try: out.append(float(r[2]))
            except ValueError: pass
    return out
ex = collect('EXECUTE', 'Accelerator (execute excluding wait) time')
wall = collect('EXECUTE', 'QNN (execute) time')
ld = collect('INIT', 'QNN (load binary) time')
if ex:
    ex_ms = sorted(v/1000.0 for v in ex)
    w_ms = sorted(v/1000.0 for v in wall) if wall else ex_ms
    print('%-11s accel min %7.2f ms  med %7.2f ms | wall min %7.2f ms | load %6.0f ms | n=%d'
          % (name, ex_ms[0], ex_ms[len(ex_ms)//2], w_ms[0],
             (min(ld)/1000.0) if ld else -1, len(ex_ms)))
else:
    print('%-11s no EXECUTE rows in %s' % (name, path))
PY
done
