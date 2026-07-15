"""
Telegram Animal-Image Filter Bot
---------------------------------
Watches a Telegram group/channel it's an admin of. When a photo is posted:
  - Runs it through a YOLOv8 object detector (in a background thread, so it
    never blocks the bot from handling other incoming photos concurrently).
  - Buffers photos that arrive close together into a "burst" (cameras here
    typically send 2 images per motion event) and waits briefly for the
    rest of the burst before acting — see BURST_WINDOW_SECONDS/MAX_BURST_SIZE.
  - If ANY photo in a burst shows a person or vehicle -> the WHOLE burst is
    kept. This favors recall: a single frame catching an intrusion is enough
    to preserve the full event, even if a companion frame in the same burst
    missed it.
  - If NO photo in the burst shows a person/vehicle (animal-only or
    unclear) -> each photo in the burst is deleted per KEEP_ON_UNCERTAIN.

No tagging/reply messages are sent — the bot only acts silently (deletes or
leaves photos alone).

Requires the bot to be added as an ADMIN with "Delete Messages" permission in the
target chat. Telegram bots cannot read message history from before they joined,
so this only affects new photos posted after the bot is added and running.

Run mode: polling (no public URL/SSL needed — just run this script on a server
or your own machine and leave it running).
"""

import asyncio
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters
from telegram.request import HTTPXRequest
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------------

# Get this from @BotFather after running /newbot
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "PASTE_YOUR_BOT_TOKEN_HERE")

# YOLOv8 model. This is a security/intrusion-monitoring use case, so accuracy
# is weighted higher than raw speed. Using the large model for maximum
# detection accuracy.
#   yolov8n.pt - fastest, least accurate
#   yolov8s.pt - good balance
#   yolov8m.pt - noticeably more accurate
#   yolov8l.pt - highest accuracy of this set  <- current
MODEL_NAME = "yolov8l.pt"

# Minimum confidence to trust a detection. Kept low so weaker/partial/distant
# detections (common in wide fisheye or wildlife-camera shots) still count.
CONFIDENCE_THRESHOLD = 0.25

# Classes (from the COCO dataset, which YOLOv8 is trained on) that mean
# "keep this image" if detected.
KEEP_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}

# Classes that count as "animal" for the delete decision — anything that
# isn't in KEEP_CLASSES is treated as a candidate for deletion as long as at
# least one animal was found.
ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe",
}

# If a single photo finds NEITHER a keep-class NOR an animal-class object
# (blurry, empty, unclear), should it count as delete-eligible? False = also
# delete empty/ambiguous photos alongside clear animal shots.
KEEP_ON_UNCERTAIN = False

# --- Multi-frame burst confirmation -----------------------------------------
# Cameras here send photos in short bursts per motion event (commonly 2
# images). Rather than deciding on each photo alone, the bot groups photos
# from the same chat that arrive close together and decides for the whole
# group at once: if ANY frame in the burst shows a person/vehicle, the whole
# burst is kept — reducing the chance a real intrusion is lost just because
# one of several frames missed the detection.

# How long to wait after the last photo in a chat before finalizing the
# burst (in case a companion frame is still on its way).
BURST_WINDOW_SECONDS = 4.0

# Finalize immediately once this many photos have arrived for a chat,
# without waiting for the full window — matches cameras that reliably send
# exactly 2 images per event.
MAX_BURST_SIZE = 2

# --- Health check -----------------------------------------------------------
# Posts a short "still running" message to a chat every N hours, so rangers
# have a quick way to notice if the bot has silently stopped working (e.g.
# stalled/crashed container) rather than only finding out when a real
# intrusion photo doesn't get filtered.
#
# Defaults to the same group the bot already monitors (seen in deploy logs
# as chat -1002504583469). Override via the HEALTH_CHECK_CHAT_ID environment
# variable if you ever want the heartbeat posted somewhere else instead, or
# set it to an empty string to disable the health check entirely.
_health_check_chat_id_raw = os.environ.get("HEALTH_CHECK_CHAT_ID", "-1002504583469")
HEALTH_CHECK_CHAT_ID = int(_health_check_chat_id_raw) if _health_check_chat_id_raw else None
HEALTH_CHECK_INTERVAL_SECONDS = 6 * 60 * 60  # every 6 hours
HEALTH_CHECK_MESSAGE = "🟢 AI running"

# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("animal-filter-bot")

logger.info("Loading YOLO model (%s)...", MODEL_NAME)
model = YOLO(MODEL_NAME)

# Single background worker thread for YOLO inference. Inference is CPU-bound
# and would otherwise block the asyncio event loop (and every other Telegram
# update) for the duration of each call. Running it in an executor frees the
# event loop to keep downloading/handling other photos concurrently while
# inference happens in the background. max_workers=1 keeps inference calls
# serialized (simplest/safest for a single YOLO model instance) while still
# unblocking the event loop.
_inference_executor = ThreadPoolExecutor(max_workers=1)

# Per-chat burst buffers: chat_id -> {"items": [(msg_id, decision), ...], "timer": asyncio.Task}
_pending_bursts: dict = {}
_pending_lock = asyncio.Lock()


def classify_image(image_path: str) -> str:
    """
    Synchronous, CPU-bound. Returns one of: "keep", "delete", "uncertain".
    Called via the thread pool executor — do not call directly from the
    event loop.
    """
    # imgsz=1280 (up from the default 640) runs inference at higher resolution,
    # which meaningfully helps detect small/distant subjects.
    results = model(image_path, imgsz=1280, verbose=False)[0]

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


async def classify_image_async(image_path: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_inference_executor, classify_image, image_path)


async def finalize_burst(context: ContextTypes.DEFAULT_TYPE, chat_id: int, items: list) -> None:
    decisions = [d for _, d in items]
    msg_ids = [m for m, _ in items]

    if "keep" in decisions:
        logger.info(
            "Kept burst of %d photo(s) with human/vehicle detected in at least one frame "
            "(chat %s, msgs %s)",
            len(items), chat_id, msg_ids,
        )
        return

    # No frame in the burst showed a person/vehicle — decide each photo
    # individually based on its own animal/uncertain classification.
    for msg_id, decision in items:
        should_delete = decision == "delete" or (decision == "uncertain" and not KEEP_ON_UNCERTAIN)
        if should_delete:
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
                logger.info(
                    "Deleted photo (burst-confirmed no human/vehicle; frame decision=%s) "
                    "(chat %s, msg %s)",
                    decision, chat_id, msg_id,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to delete message %s in chat %s: %s. "
                    "Check the bot has admin + 'Delete Messages' rights.",
                    msg_id, chat_id, exc,
                )
        else:
            logger.info(
                "Kept photo (uncertain, KEEP_ON_UNCERTAIN=True) (chat %s, msg %s)",
                chat_id, msg_id,
            )


async def _finalize_after_delay(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
    try:
        await asyncio.sleep(BURST_WINDOW_SECONDS)
    except asyncio.CancelledError:
        return

    async with _pending_lock:
        bucket = _pending_bursts.pop(chat_id, None)

    if bucket and bucket["items"]:
        await finalize_burst(context, chat_id, bucket["items"])


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message or not message.photo:
        return

    # Highest-resolution version of the photo
    photo = message.photo[-1]
    file = await context.bot.get_file(photo.file_id)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        await file.download_to_drive(tmp.name)
        decision = await classify_image_async(tmp.name)

    chat_id = message.chat_id
    msg_id = message.message_id

    async with _pending_lock:
        bucket = _pending_bursts.setdefault(chat_id, {"items": [], "timer": None})
        bucket["items"].append((msg_id, decision))

        if bucket["timer"] is not None:
            bucket["timer"].cancel()

        if len(bucket["items"]) >= MAX_BURST_SIZE:
            items = bucket["items"]
            del _pending_bursts[chat_id]
            finalize_now = True
        else:
            bucket["timer"] = asyncio.create_task(_finalize_after_delay(context, chat_id))
            finalize_now = False

    if finalize_now:
        await finalize_burst(context, chat_id, items)


async def health_check_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        await context.bot.send_message(chat_id=HEALTH_CHECK_CHAT_ID, text=HEALTH_CHECK_MESSAGE)
        logger.info("Health check sent to chat %s", HEALTH_CHECK_CHAT_ID)
    except Exception as exc:
        logger.warning("Failed to send health check to chat %s: %s", HEALTH_CHECK_CHAT_ID, exc)


def main() -> None:
    if BOT_TOKEN == "PASTE_YOUR_BOT_TOKEN_HERE":
        raise SystemExit(
            "Set your bot token first: either edit BOT_TOKEN in bot.py, "
            "or set the TELEGRAM_BOT_TOKEN environment variable."
        )

    # Longer timeouts than the library default reduce occasional "TimedOut"
    # errors when downloading larger photos or on a slow connection between
    # Railway and Telegram's file servers.
    request = HTTPXRequest(connect_timeout=15.0, read_timeout=30.0, write_timeout=15.0)

    app = Application.builder().token(BOT_TOKEN).request(request).build()
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    if HEALTH_CHECK_CHAT_ID:
        app.job_queue.run_repeating(
            health_check_job,
            interval=HEALTH_CHECK_INTERVAL_SECONDS,
            first=HEALTH_CHECK_INTERVAL_SECONDS,
        )
        logger.info(
            "Health check enabled: posting to chat %s every %d hours",
            HEALTH_CHECK_CHAT_ID, HEALTH_CHECK_INTERVAL_SECONDS // 3600,
        )
    else:
        logger.info("Health check disabled (HEALTH_CHECK_CHAT_ID not set)")

    logger.info("Bot started. Polling for new photos...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
