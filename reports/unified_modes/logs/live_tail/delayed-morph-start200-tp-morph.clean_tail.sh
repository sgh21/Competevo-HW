#!/usr/bin/env bash
set -euo pipefail
cd /home/user/Data/sgh_workstation/WorkSpace/Competevo-HW
LOG='tmp/robo-sumo-devants-v0/delayed-morph-start200-tp-morph-20260605_170014/log/robo-sumo-devants-v0-20260605_170014.log'
CLEAN='reports/unified_modes/logs/live_tail/delayed-morph-start200-tp-morph.clean.log'
tail -n 0 -F "$LOG" | perl -pe 'BEGIN{$|=1} s/\e\[[0-9;]*[A-Za-z]//g' >> "$CLEAN"
