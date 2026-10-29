#!/bin/bash
# Run a command but always exit 0, so a crashed pilot still lets its gate run (the gate then sees the missing
# output and picks the fallback). Usage: bash code/run_soft.sh <name> <command...>
name="$1"; shift
if "$@"; then
else
fi
exit 0
