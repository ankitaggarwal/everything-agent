#!/bin/bash
# Install the Everything Agent as a systemd service on the Reachy Mini.
# Run ON the robot, from the repo root:
#     bash deploy/install_service.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
UNIT=/etc/systemd/system/everything-agent.service

echo "Installing $UNIT ..."
sudo cp "$HERE/everything-agent.service" "$UNIT"
sudo systemctl daemon-reload
sudo systemctl enable everything-agent.service

cat <<'EOF'

Installed and enabled (starts on boot).

  start:   sudo systemctl start everything-agent
  stop:    sudo systemctl stop everything-agent
  status:  systemctl status everything-agent
  logs:    journalctl -u everything-agent -f
  disable: sudo systemctl disable --now everything-agent

Reminder: while it runs it holds the robot; stop it before running another app.
EOF
