"""
Telegram Animal-Image Filter Bot
---------------------------------
Watches a Telegram group/channel it's an admin of. When a photo is posted:
  - Runs it through a YOLOv8 object detector.
  - If a "person" or vehicle (car/truck/bus/motorcycle/bicycle) is detected -> KEEP.
  - If only animal classes are detected (no person/vehicle) -> DELETE.
  - If nothing relevant is detected at all (empty/unclear result) -> KEEP by default
    (safer than risking deletion of a non-animal photo; see KEEP_ON_UNCERTAIN below).

Requires the bot to be added as an ADMIN with "Delete Messages" permission in the
target chat. Telegram bots cannot read message history from before they joined,
so this only affects new photos posted after the bot is added and running.

Run mode: polling (no public URL/SSL needed — just run this script on a server
or your own machine and leave it running).
"""

import logging
import os
import tempfile

from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters
from telegram.request import HTTPXRequest
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------------

# Get this from @BotFather after running /newbot
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "PASTE_YOUR_BOT_TOKEN_HERE")

# YOLOv8 model. "yolov8n.pt" (nano) is fastest but least accurate. "yolov8s.pt"
# (small) is meaningfully more accurate at a modest speed cost — a single
# photo still processes in well under a second on typical Railway hardware.
# Use "yolov8m.pt" for even higher accuracy if you don't mind it being slower.
MODEL_NAME = "yolov8s.pt"

# Minimum confidence to trust a detection. Lowered from 0.45 so more animals
# (and more people/vehicles) get picked up instead of falling through as
# "uncertain". If you start seeing false deletes (a person/car photo removed
# by mistake), raise this back up in steps of 0.05.
CONFIDENCE_THRESHOLD = 0.30

# Classes (from the COCO dataset, which YOLOv8 is trained on) that mean
# "keep this image" if detected.
KEEP_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}

# Classes that count as "animal" for logging/clarity (not strictly required
# for the delete decision — anything that isn't in KEEP_CLASSES is treated
# as a candidate for deletion as long as at least one animal was found).
ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe",
}

# If the detector finds NEITHER a keep-class NOR an animal-class object
# (e.g. blurry photo, landscape, empty/unclear image), should we keep or
# delete? Set to False to also clear out empty/ambiguous photos, alongside
# clear animal shots. Trade-off: a person/car photo the model fails to
# recognize (bad angle, poor lighting) would also get deleted under this
# setting, since it no longer defaults to the safe "keep" behavior.
KEEP_ON_UNCERTAIN = False

# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("animal-filter-bot")

logger.info("Loading YOLO model (%s)...", MODEL_NAME)
model = YOLO(MODEL_NAME)


def classify_image(image_path: str) -> str:
    """
    Returns one of: "keep", "delete", "uncertain"
    """
    results = model(image_path, verbose=False)[0]

    found_keep = False
    found_animal = False

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONFIDENCE_THRESHOLD:
            continue
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]

        if cls_name in KEEP_CLASSES:
            found_keep = True
        elif cls_name in ANIMAL_CLASSES:
            found_animal = True

    if found_keep:
        return "keep"
    if found_animal:
        return "delete"
    return "uncertain"


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message or not message.photo:
        return

    # Highest-resolution version of the photo
    photo = message.photo[-1]
    file = await context.bot.get_file(photo.file_id)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        await file.download_to_drive(tmp.name)
        decision = classify_image(tmp.name)

    chat_id = message.chat_id
    msg_id = message.message_id

    if decision == "delete":
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
            logger.info("Deleted animal-only image (chat %s, msg %s)", chat_id, msg_id)
        except Exception as exc:
            logger.warning(
                "Failed to delete message %s in chat %s: %s. "
                "Check the bot has admin + 'Delete Messages' rights.",
                msg_id, chat_id, exc,
            )
    elif decision == "keep":
        logger.info("Kept image with human/vehicle (chat %s, msg %s)", chat_id, msg_id)
    else:  # uncertain
        if KEEP_ON_UNCERTAIN:
            logger.info("Uncertain classification, keeping (chat %s, msg %s)", chat_id, msg_id)
        else:
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
                logger.info("Uncertain classification, deleted per config (chat %s, msg %s)", chat_id, msg_id)
            except Exception as exc:
                logger.warning("Failed to delete uncertain message %s: %s", msg_id, exc)


def main() -> None:
    if BOT_TOKEN == "PASTE_YOUR_BOT_TOKEN_HERE":
        raise SystemExit(
            "Set your bot token first: either edit BOT_TOKEN in bot.py, "
            "or set the TELEGRAM_BOT_TOKEN environment variable."
        )

    # Longer timeouts than the library default reduce the occasional
    # "TimedOut" errors seen when downloading larger photos or on a slow
    # connection between Railway and Telegram's file servers. These are
    # non-fatal either way (the bot recovers and keeps polling), but fewer
    # timeouts means fewer skipped/retried classifications.
    request = HTTPXRequest(connect_timeout=15.0, read_timeout=30.0, write_timeout=15.0)

    app = Application.builder().token(BOT_TOKEN).request(request).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("Bot started. Polling for new photos...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
