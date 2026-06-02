# Deploying to a Reachy Mini Wireless

Everything here is **deployment glue**, deliberately kept *outside* the
`everything_agent` package — the package stays a clean, importable library; this
folder is how you put it on a robot.

## How a deploy works

The agent runs **on the robot**, inside the same Python venv the official Reachy
Mini apps use (`/venvs/apps_venv`), so it shares the on-board `reachy_mini` SDK.
The Reachy Mini **daemon** owns the hardware; the agent connects to it as a
client (see `everything_agent/robot/reachy_mini.py`).

```
your laptop                         Reachy Mini (Raspberry Pi 5)
-----------                         ----------------------------
repo  ──rsync──▶  /home/pollen/everything-agent
.env  ──scp────▶  /home/pollen/everything-agent/.env   (chmod 600)
                  /venvs/apps_venv/bin/python -m everything_agent
                          │  client of ──▶  reachy-mini-daemon (hardware + media)
```

## 1. Copy the code + secrets

```bash
# from the repo root on your laptop
rsync -az --exclude .git --exclude __pycache__ --exclude .venv \
      --exclude memory.json --exclude .env  ./  pollen@reachy-mini.local:/home/pollen/everything-agent/
scp .env pollen@reachy-mini.local:/home/pollen/everything-agent/.env
ssh pollen@reachy-mini.local 'chmod 600 /home/pollen/everything-agent/.env'
```

## 2. Install dependencies (once, on the robot)

```bash
# into the shared apps venv so we reuse the reachy_mini SDK
/venvs/apps_venv/bin/pip install anthropic python-dotenv pyyaml
# optional capabilities:
/venvs/apps_venv/bin/pip install openwakeword onnxruntime   # "Hey Reachy" wake word
/venvs/apps_venv/bin/pip install claude-agent-sdk           # real tool-using brain
# for the claude_sdk brain you also need Node + the Claude Code CLI:
sudo apt-get install -y nodejs npm && sudo npm install -g @anthropic-ai/claude-code
claude login        # one interactive step (device code in a browser)
```

## 3. Run it

Foreground (for debugging):
```bash
cd ~/everything-agent && bash scripts/run_on_robot.sh
```

As a boot service (recommended):
```bash
bash deploy/install_service.sh
sudo systemctl start everything-agent
journalctl -u everything-agent -f
```

Stop / disable:
```bash
sudo systemctl stop everything-agent
sudo systemctl disable --now everything-agent
```

> Only one app controls the robot at a time. Stop this service before running a
> different app from the dashboard, and vice-versa.
