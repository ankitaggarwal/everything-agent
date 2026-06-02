#!/bin/bash
# Launch the everything-agent on the Reachy Mini (run on the robot).
#   nohup bash scripts/run_on_robot.sh >/tmp/ea.log 2>&1 & disown
# Stop with:  pkill -f "[e]verything_agent"
cd /home/pollen/everything-agent || exit 1
export HOME=/home/pollen
export EVERYTHING_AGENT_CONFIG=config.reachy.yaml
exec /venvs/apps_venv/bin/python -u -m everything_agent
