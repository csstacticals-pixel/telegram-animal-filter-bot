# Telegram Animal-Image Filter Bot — Setup Guide

Deletes photos that only contain animals; keeps photos containing a person
or a vehicle (car, truck, bus, motorcycle, bicycle). Built on YOLOv8 object
detection, running in polling mode (no public server/SSL required).

## Files

- `bot.py` — the bot
- `requirements.txt` — Python dependencies
- `Procfile` — tells Railway (or Heroku-style platforms) how to start the bot
- `railway.json` — Railway build/deploy config (auto-restart on crash, etc.)
- `.python-version` — pins the Python version Railway builds with
- `.gitignore` — keeps venv/cache/model files out of your repo
- `nixpacks.toml` — installs system libraries (`libgl1`, `libglib2.0-0`) that
  `opencv`/`ultralytics` need but aren't present on Railway's base image by
  default (without this, the deploy crashes with `ImportError: libGL.so.1`)

## 1. Create the bot on Telegram

1. Open a chat with **@BotFather** in Telegram.
2. Send `/newbot`, follow the prompts, and copy the API token it gives you.

## 2. Add the bot to your group/channel

1. Add the bot to the target group or channel.
2. Promote it to **Admin**.
3. Make sure **"Delete Messages"** permission is enabled for the bot — without
   this, it can see photos but cannot remove them.

> Note: Telegram bots only see messages sent **after** they join. It cannot
> retroactively scan/delete older photos already in the chat history.

## 3. Install and run

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

export TELEGRAM_BOT_TOKEN="paste-your-token-here"   # Windows: set TELEGRAM_BOT_TOKEN=...
python3 bot.py
```

The first run downloads the YOLOv8 nano model automatically (~6 MB) — this
can take a minute depending on your connection. After that it starts polling
immediately.

Leave the script running (or deploy it — see below) and it will process
every new photo sent to the chat from then on.

## 4. Tuning (optional, edit `bot.py`)

- `MODEL_NAME` — swap `yolov8n.pt` for `yolov8s.pt`/`yolov8m.pt` for better
  accuracy at the cost of speed (nano is fine for most use cases).
- `CONFIDENCE_THRESHOLD` — raise if you're seeing false deletes, lower if
  animal-only photos are slipping through.
- `KEEP_CLASSES` / `ANIMAL_CLASSES` — add or remove COCO class names as needed.
- `KEEP_ON_UNCERTAIN` — set to `False` if you'd rather delete anything the
  model can't confidently classify (not recommended to start with).

## 5. Deploying to Railway (push and deploy — recommended, fastest)

The repo is already pre-configured with `Procfile` and `railway.json`, so
Railway auto-detects everything. No manual start-command setup needed.

**Push to GitHub:**

```bash
git init
git add .
git commit -m "Telegram animal-image filter bot"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

(Create the empty repo on github.com first if you haven't.)

**Deploy on Railway:**

1. Go to railway.app → sign in with GitHub.
2. **New Project → Deploy from GitHub repo** → select your repo.
3. Railway reads `railway.json`/`Procfile` and installs `requirements.txt`
   automatically — no build config needed from you.
4. Go to the service's **Variables** tab → add:
   - `TELEGRAM_BOT_TOKEN` = your token from BotFather
5. Click **Deploy**. Once the build finishes, check the **Deployments → Logs**
   tab for `Bot started. Polling for new photos...` — that confirms it's live.

From here it runs continuously in the cloud; you don't need your computer on.

**Alternative: run it yourself (VPS, Pi, local machine)**

```bash
nohup python3 bot.py > bot.log 2>&1 &
```

Or wrap it in a systemd service / Docker container for production use.

## Known limitations

- Classification is only as good as YOLOv8's COCO-trained detector — unusual
  angles, low light, or partially obscured subjects can be misclassified.
  `KEEP_ON_UNCERTAIN = True` is the safety net for that.
- No history backfill (see note above).
- Runs on CPU by default; fine for personal/small-group volume. For a
  high-traffic channel, run on a machine with a GPU or reduce the model size.
