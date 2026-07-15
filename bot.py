"""
Telegram Animal-Image Filter Bot — MegaDetector edition
--------------------------------------------------------
Watches a Telegram group/channel it's an admin of. When a photo is posted:
  - Enhances the image internally (contrast/low-light correction) to help
    the detector, without altering the photo actually left in the chat.
  - Runs it through MegaDetectorV6 (PytorchWildlife) — a detector built
    specifically for camera-trap imagery, with a generic 3-class taxonomy:
    animal / person / vehicle. Unlike a general COCO model, it doesn't need
    to recognize specific species (oryx, springbok, kudu, etc.) — anything
    non-human/non-vehicle just falls under "animal".
  - Runs inference in a background thread (never blocks the bot from
    handling other incoming photos concurrently).
  - Buffers photos that arrive close together into a "burst" (cameras here
    typically send 2 images per motion event) and waits briefly for the
    rest of the burst before acting — see BURST_WINDOW_SECONDS/MAX_BURST_SIZE.
  - If ANY photo in a burst shows a person or vehicle -> the WHOLE burst is
    kept. If NO photo in the burst shows a person/vehicle -> every photo in
    the burst is deleted.

No tagging/reply messages are sent — the bot only acts silently (deletes or
leaves photos alone), aside from the periodic health-check heartbeat.

Requires the bot to be added as an ADMIN with "Delete Messages" permission in
the target chat. Telegram bots cannot read message history from before they
joined, so this only affects new photos posted after the bot is added.

Run mode: polling (no public URL/SSL needed — just run this script on a
server or your own machine and leave it running).

--- IMPORTANT NOTE ON THIS VERSION ---
The MegaDetector/PytorchWildlife integration below is built from Microsoft's
official PyPI README and model-zoo docs, but their documentation site is
JavaScript-rendered so I could not confirm every exact detail of the
single_image_detection() return format from here. The result-parsing code
is written defensively (tries several plausible shapes and logs clearly if
none match) specifically so that if the exact format differs, the very
first deploy log will show us precisely what to adjust — the same pattern
we used to fix the earlier OpenCV/libGL issue on Railway. Expect to send me
the first deploy log after this change so we can confirm/adjust quickly.
"""

import asyncio
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from telegram import Update
from telegram.ext import Application, MessageHandler, ContextTypes, filters
from telegram.request import HTTPXRequest
from PytorchWildlife.models import detection as pw_detection

# ---------------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------------

# Get this from @BotFather after running /newbot
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "PASTE_YOUR_BOT_TOKEN_HERE")

# MegaDetectorV6 version. "MDV6-yolov10-c" is the smallest/fastest variant
# (2.3M params, no NMS post-processing step) — chosen here to keep
# per-photo processing quick. If detection quality isn't good enough,
# "MDV6-yolov10-e" (29.5M params) trades speed for meaningfully higher
# accuracy — see https://microsoft.github.io/CameraTraps/model_zoo/megadetector/
MEGADETECTOR_VERSION = "MDV6-yolov10-c"

# Minimum confidence to trust a detection.
CONFIDENCE_THRESHOLD = 0.25

# If a photo finds NEITHER a person/vehicle NOR an animal (blurry, empty,
# unclear), should it count as delete-eligible? False = also delete
# empty/ambiguous photos alongside clear animal shots.
KEEP_ON_UNCERTAIN = False

# --- Image enhancement (for detection accuracy only) ------------------------
# Camera-trap footage is often low-contrast, hazy, or dim (dawn/dusk/night).
# When enabled, each photo is enhanced in memory before being handed to the
# detector — the photo left in the chat is untouched.
ENHANCE_IMAGES = True

# Denoising (on top of contrast correction) further helps grainy/noisy night
# shots, but is noticeably slower per photo. Off by default.
ENHANCE_DENOISE = False

# --- Multi-frame burst confirmation -----------------------------------------
# Cameras here send photos in short bursts per motion event (commonly 2
# images). Rather than deciding on each photo alone, the bot groups photos
# from the same chat that arrive close together and decides for the whole
# group at once: if ANY frame in the burst shows a person/vehicle, the whole
# burst is kept.
BURST_WINDOW_SECONDS = 4.0
MAX_BURST_SIZE = 2

# --- Health check -----------------------------------------------------------
# Posts a short "still running" message to a chat every N hours. Defaults to
# the same group the bot already monitors (seen in deploy logs as chat
# -1002504583469). Override via HEALTH_CHECK_CHAT_ID, or set it to an empty
# string to disable.
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

logger.info("Loading MegaDetectorV6 (%s)...", MEGADETECTOR_VERSION)
detector = pw_detection.MegaDetectorV6(version=MEGADETECTOR_VERSION)

# MegaDetector's fixed taxonomy. PytorchWildlife typically exposes this as
# detector.CLASS_NAMES; falling back to the documented default order
# (0=animal, 1=person, 2=vehicle) if that attribute isn't present under
# this version, so a mismatch here is easy to spot and fix from the logs.
CLASS_NAMES = getattr(detector, "CLASS_NAMES", {0: "animal", 1: "person", 2: "vehicle"})
logger.info("MegaDetector class map: %s", CLASS_NAMES)

KEEP_LABELS = {"person", "vehicle"}
ANIMAL_LABELS = {"animal"}

# Single background worker thread for detector inference — frees the asyncio
# event loop to keep handling other photos while inference runs.
_inference_executor = ThreadPoolExecutor(max_workers=1)

# Per-chat burst buffers: chat_id -> {"items": [(msg_id, decision), ...], "timer": asyncio.Task}
_pending_bursts: dict = {}
_pending_lock = asyncio.Lock()


def enhance_image(image_path: str):
    """
    Loads the image and applies CLAHE (contrast-limited adaptive histogram
    equalization) to correct low-contrast/dim shots. Returns a numpy array
    (BGR) on success, or the original path on failure so detection can
    still fall back to reading the file directly.
    """
    img = cv2.imread(image_path)
    if img is None:
        return image_path

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    if ENHANCE_DENOISE:
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)

    return enhanced


def _extract_detections(result):
    """
    Defensively pulls (label, confidence) pairs out of whatever shape
    single_image_detection() returns. Tries the documented "detections"
    object (supervision.Detections-style, with .class_id / .confidence
    arrays) first, then falls back to a couple of other plausible shapes.
    Logs the raw result once if nothing matches, so we can see exactly
    what came back and adjust quickly.
    """
    pairs = []

    detections = None
    if isinstance(result, dict):
        detections = result.get("detections")
    else:
        detections = getattr(result, "detections", None)

    if detections is not None:
        class_ids = getattr(detections, "class_id", None)
        confidences = getattr(detections, "confidence", None)
        if class_ids is not None and confidences is not None:
            for cls_id, conf in zip(class_ids, confidences):
                label = CLASS_NAMES.get(int(cls_id), str(cls_id))
                pairs.append((label, float(conf)))
            return pairs

    # Fallback: a list of dicts each with e.g. "category"/"label" and "conf"/"confidence".
    if isinstance(result, dict) and "detections" in result and isinstance(result["detections"], list):
        for det in result["detections"]:
            label = det.get("label") or det.get("category") or det.get("class")
            conf = det.get("confidence", det.get("conf", 0.0))
            if label is not None:
                pairs.append((str(label), float(conf)))
        if pairs:
            return pairs

    logger.warning(
        "Could not parse MegaDetector result into (label, confidence) pairs — "
        "raw result: %r. Detection will be treated as 'uncertain' until this "
        "is fixed; send this log line so the parsing can be corrected.",
        result,
    )
    return pairs


def classify_image(image_path: str) -> str:
    """
    Synchronous, CPU-bound. Returns one of: "keep", "delete", "uncertain".
    Called via the thread pool executor — do not call directly from the
    event loop.
    """
    detection_input = enhance_image(image_path) if ENHANCE_IMAGES else image_path

    if isinstance(detection_input, np.ndarray):
        img_rgb = cv2.cvtColor(detection_input, cv2.COLOR_BGR2RGB)
    else:
        img_bgr = cv2.imread(detection_input)
        if img_bgr is None:
            logger.warning("Could not read image for detection: %s", detection_input)
            return "uncertain"
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # PytorchWildlife's quick-start example passes a channels-first
    # (3, H, W) array — matching that shape here.
    img_chw = np.transpose(img_rgb, (2, 0, 1))

    result = detector.single_image_detection(img_chw)
    detections = _extract_detections(result)

    found_keep = False
    found_animal = False
    for label, conf in detections:
        if conf < CONFIDENCE_THRESHOLD:
            continue
        if label in KEEP_LABELS:
            found_keep = True
        elif label in ANIMAL_LABELS:
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
