"""
Telegram Animal-Image Filter Bot — MegaDetector edition
--------------------------------------------------------
Watches a Telegram group/channel it's an admin of. When a photo is posted:
  - Enhances the image internally (contrast/low-light correction) to help
    the detector, without altering the photo actually left in the chat.
  - Runs it through MegaDetectorV6 (PytorchWildlife) — a detector built
    specifically for camera-trap imagery, with a generic 3-class taxonomy:
    animal / person / vehicle.
  - Runs inference in a background thread (never blocks the bot from
    handling other incoming photos concurrently).
  - Buffers photos that arrive close together into a "burst" (cameras here
    typically send 2 images per motion event) and waits briefly for the
    rest of the burst before acting — see BURST_WINDOW_SECONDS/MAX_BURST_SIZE.
  - If ANY photo in a burst shows a person or vehicle -> the WHOLE burst is
    provisionally kept. If NO photo in the burst shows a person/vehicle ->
    every photo in the burst is deleted.
  - On top of that, kept bursts go through a rolling dedupe window
    (DEDUPE_WINDOW_SECONDS): a later kept burst is only treated as a
    duplicate of an earlier one — and the weaker one deleted — if BOTH of
    these hold:
      1. Same scene: a perceptual image-hash comparison
         (SIMILARITY_HAMMING_THRESHOLD).
      2. Same source (when it can be determined): the camera name/timestamp
         stamp attached to the photo matches. This is read from the
         Telegram caption/text first (cheap, always on), and — if enabled —
         from OCR of any watermark burned directly into the image pixels
         (see OCR_ENABLED below). If source can't be determined for either
         photo, the check is skipped and the decision falls back to the
         scene-similarity result alone, so unrelated cameras never get
         silently merged just because two frames happen to look similar.
    Two different vehicles/people that both land inside the window are NOT
    merged just because they share a class label — only matching scene +
    matching source counts as a duplicate. The window rolls forward on
    every new kept burst, whether it wins, loses, or is judged distinct.
  - Tracks daily counts (received / kept / deleted) and posts a summary once
    a day — see DAILY_SUMMARY_HOUR_UTC below.

No per-photo tagging/reply messages are sent — the bot only acts silently
(deletes or leaves photos alone), aside from the health-check heartbeat and
the daily summary.

Requires the bot to be added as an ADMIN with "Delete Messages" permission in
the target chat. Telegram bots cannot read message history from before they
joined, so this only affects new photos posted after the bot is added.

Run mode: polling (no public URL/SSL needed — just run this script on a
server or your own machine and leave it running).
"""

import asyncio
import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import time as dt_time

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

# MegaDetectorV6 version. "MDV6-yolov10-c" is the smallest/fastest variant.
MEGADETECTOR_VERSION = "MDV6-yolov10-c"

# Minimum confidence to trust a detection.
CONFIDENCE_THRESHOLD = 0.25

# If a photo finds NEITHER a person/vehicle NOR an animal (blurry, empty,
# unclear), should it count as delete-eligible? False = also delete
# empty/ambiguous photos alongside clear animal shots.
KEEP_ON_UNCERTAIN = False

# --- Image enhancement (for detection accuracy only) ------------------------
ENHANCE_IMAGES = True
ENHANCE_DENOISE = False

# --- Multi-frame burst confirmation -----------------------------------------
# Cameras here typically fire 2 images per motion event; group photos that
# arrive close together and decide on the whole group at once.
BURST_WINDOW_SECONDS = 4.0
MAX_BURST_SIZE = 2

# --- Rolling dedupe window for kept (human/vehicle) bursts -------------------
# If another kept burst lands within this many seconds of the last one (for
# the same chat), it's only treated as a duplicate (weaker one deleted) if
# it also passes the scene-similarity AND source-match checks below.
# Set to 0 to disable dedupe entirely.
DEDUPE_WINDOW_SECONDS = int(os.environ.get("DEDUPE_WINDOW_SECONDS", str(5 * 60)))

# How visually similar two photos must be (perceptual/average-hash Hamming
# distance out of 64 bits) to be treated as the SAME scene/event, rather than
# two different vehicles or people that both happen to fall inside the
# dedupe window. Lower = stricter (fewer merges, more kept photos). Higher =
# looser (more aggressive deduping). 0-6 ~= near-identical frame. 7-12 ~=
# same scene, some movement/lighting change. Above ~16 is usually a
# genuinely different subject/framing.
SIMILARITY_HAMMING_THRESHOLD = int(os.environ.get("SIMILARITY_HAMMING_THRESHOLD", "10"))

# --- Source/camera-stamp matching for dedupe ---------------------------------
# Two "kept" photos are only ever merged as duplicates if their source can't
# be told apart from those below, OR the sources actually match.
#
# 1) Caption/text stamp (always on, zero extra dependencies): if the photo
#    arrives with a caption or accompanying text — e.g. camera exports like
#    "EtoshaMushara Camera 4-Region-2 - Mushara Water Hole" — that text
#    (with date/time portions stripped out) is used as the camera's source
#    fingerprint.
#
# 2) Burned-in image watermark via OCR (OFF by default): some cameras stamp
#    the name/timestamp directly onto the photo's pixels instead of (or as
#    well as) sending a caption. Reading that requires OCR, which needs the
#    `pytesseract` + `Pillow` Python packages AND the `tesseract-ocr` system
#    binary — a new dependency we intentionally do NOT install automatically
#    given how much trouble third-party package installs have caused on
#    Railway before. To enable it:
#      - add `tesseract-ocr` to the `aptPkgs` (or install phase) in
#        nixpacks.toml
#      - add `pytesseract` and `Pillow` to requirements.txt
#      - set OCR_ENABLED=true in Railway's Variables tab
#    Until then, this code path is a no-op (it detects the missing packages
#    and silently skips OCR, relying on the caption stamp alone).
OCR_ENABLED = os.environ.get("OCR_ENABLED", "false").strip().lower() == "true"
# Bottom fraction of the image to run OCR on (where watermarks usually sit).
OCR_CROP_BOTTOM_FRACTION = float(os.environ.get("OCR_CROP_BOTTOM_FRACTION", "0.15"))

try:
    if OCR_ENABLED:
        import pytesseract
        from PIL import Image
        _OCR_AVAILABLE = True
    else:
        _OCR_AVAILABLE = False
except ImportError:
    _OCR_AVAILABLE = False

# --- Health check -----------------------------------------------------------
_health_check_chat_id_raw = os.environ.get("HEALTH_CHECK_CHAT_ID", "-1002504583469")
HEALTH_CHECK_CHAT_ID = int(_health_check_chat_id_raw) if _health_check_chat_id_raw else None
HEALTH_CHECK_INTERVAL_SECONDS = 6 * 60 * 60  # every 6 hours
HEALTH_CHECK_MESSAGE = "🟢 AI running"

# --- Daily summary digest ----------------------------------------------------
# Posts one message per day with counts of everything the bot processed.
# Reuses HEALTH_CHECK_CHAT_ID by default (same monitored group) — override
# with DAILY_SUMMARY_CHAT_ID if you want it posted somewhere else instead.
_daily_summary_chat_id_raw = os.environ.get("DAILY_SUMMARY_CHAT_ID", _health_check_chat_id_raw)
DAILY_SUMMARY_CHAT_ID = int(_daily_summary_chat_id_raw) if _daily_summary_chat_id_raw else None

# What time (UTC) to post the digest. Default 06:00 UTC — adjust to your
# rangers' shift start via the DAILY_SUMMARY_HOUR_UTC env var if needed.
DAILY_SUMMARY_HOUR_UTC = int(os.environ.get("DAILY_SUMMARY_HOUR_UTC", "6"))

# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("animal-filter-bot")

if OCR_ENABLED and not _OCR_AVAILABLE:
    logger.warning(
        "OCR_ENABLED=true but pytesseract/Pillow (or the tesseract-ocr binary) are not "
        "installed — falling back to caption-only source matching. See the OCR_ENABLED "
        "comment in bot.py for how to enable it."
    )

logger.info("Loading MegaDetectorV6 (%s)...", MEGADETECTOR_VERSION)
detector = pw_detection.MegaDetectorV6(version=MEGADETECTOR_VERSION)

CLASS_NAMES = getattr(detector, "CLASS_NAMES", {0: "animal", 1: "person", 2: "vehicle"})
logger.info("MegaDetector class map: %s", CLASS_NAMES)

KEEP_LABELS = {"person", "vehicle"}
ANIMAL_LABELS = {"animal"}

_inference_executor = ThreadPoolExecutor(max_workers=1)

_pending_bursts: dict = {}
_pending_lock = asyncio.Lock()

# Rolling dedupe state per chat:
# {"msg_ids": [...], "confidence": float, "timestamp": float, "hash": int|None, "source_key": str|None}
_recent_keeps: dict = {}
_recent_keep_lock = asyncio.Lock()

# Daily counters — reset after each digest is sent. Only ever touched from
# the asyncio event loop (no await between read and write), so plain ints
# are safe without an extra lock.
_stats = {
    "received": 0,
    "kept": 0,
    "deleted": 0,
}

_DATE_TIME_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}"          # 2026-07-15
    r"|\d{1,2}/\d{1,2}/\d{2,4}"   # 15/07/2026
    r"|\d{1,2}:\d{2}(:\d{2})?"    # 15:33 or 15:33:06
)


def normalize_source_text(text: str):
    """
    Strips date/time-like substrings and normalizes whitespace/case, so the
    remaining text is a stable "camera identity" fingerprint even though the
    timestamp portion differs between every photo from the same camera.
    Returns None for empty/whitespace-only results.
    """
    if not text:
        return None
    cleaned = _DATE_TIME_PATTERN.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned or None


def extract_caption_source_key(message):
    caption = getattr(message, "caption", None) or getattr(message, "text", None)
    return normalize_source_text(caption)


def extract_ocr_source_key(image_path: str):
    """
    Best-effort OCR of any camera watermark burned into the bottom strip of
    the image. Returns None if OCR isn't enabled/available, or nothing
    readable was found. Never raises — a failure here should never break
    classification.
    """
    if not OCR_ENABLED or not _OCR_AVAILABLE:
        return None
    try:
        pil_img = Image.open(image_path)
        width, height = pil_img.size
        crop_top = int(height * (1 - OCR_CROP_BOTTOM_FRACTION))
        crop = pil_img.crop((0, crop_top, width, height))
        text = pytesseract.image_to_string(crop)
        return normalize_source_text(text)
    except Exception as exc:
        logger.warning("OCR source extraction failed for %s: %s", image_path, exc)
        return None


def enhance_image(image_path: str):
    """
    Loads the image and applies CLAHE to correct low-contrast/dim shots.
    Returns a numpy array (BGR, uint8) on success, or the original path on
    failure so detection can still fall back to reading the file directly.
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


def compute_image_hash(img_bgr):
    """
    Cheap 64-bit average-hash (aHash) of an image, used only to tell whether
    two kept photos are the same scene/event or genuinely different subjects.
    Not a security/detection signal — purely for dedupe comparison.
    """
    if img_bgr is None:
        return None
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
        avg = resized.mean()
        bits = (resized > avg).flatten()
        h = 0
        for bit in bits:
            h = (h << 1) | int(bit)
        return h
    except Exception as exc:
        logger.warning("Failed to compute image hash: %s", exc)
        return None


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _extract_detections(result):
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
        "raw result: %r.",
        result,
    )
    return pairs


def classify_image(image_path: str):
    """
    Synchronous, CPU-bound. Returns a (decision, confidence, image_hash,
    ocr_source_key) tuple, where decision is one of: "keep", "delete",
    "uncertain". confidence is the strongest matching detection's confidence
    (0.0 for "uncertain"). image_hash is a 64-bit perceptual hash used only
    for dedupe-window scene comparison (None if it couldn't be computed).
    ocr_source_key is the normalized camera-watermark text read via OCR, if
    OCR_ENABLED (None otherwise). Called via the thread pool executor — do
    not call directly from the event loop.
    """
    raw_img_bgr = cv2.imread(image_path)
    image_hash = compute_image_hash(raw_img_bgr)
    ocr_source_key = extract_ocr_source_key(image_path)

    detection_input = enhance_image(image_path) if ENHANCE_IMAGES else image_path

    if isinstance(detection_input, np.ndarray):
        img_rgb = cv2.cvtColor(detection_input, cv2.COLOR_BGR2RGB)
    else:
        img_bgr = raw_img_bgr if raw_img_bgr is not None else cv2.imread(detection_input)
        if img_bgr is None:
            logger.warning("Could not read image for detection: %s", detection_input)
            return "uncertain", 0.0, image_hash, ocr_source_key
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Pass the array in standard (height, width, channels) order — this is
    # an Ultralytics-backed model under the hood, which expects HWC.
    result = detector.single_image_detection(img_rgb)
    detections = _extract_detections(result)

    found_keep = False
    found_animal = False
    keep_confidence = 0.0
    animal_confidence = 0.0

    for label, conf in detections:
        if conf < CONFIDENCE_THRESHOLD:
            continue
        if label in KEEP_LABELS:
            found_keep = True
            keep_confidence = max(keep_confidence, conf)
        elif label in ANIMAL_LABELS:
            found_animal = True
            animal_confidence = max(animal_confidence, conf)

    if found_keep:
        return "keep", keep_confidence, image_hash, ocr_source_key
    if found_animal:
        return "delete", animal_confidence, image_hash, ocr_source_key
    return "uncertain", 0.0, image_hash, ocr_source_key


async def classify_image_async(image_path: str):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_inference_executor, classify_image, image_path)


async def _delete_messages(context: ContextTypes.DEFAULT_TYPE, chat_id: int, msg_ids: list, reason: str) -> int:
    """Deletes each message id, logging + counting successes. Returns count deleted."""
    deleted_count = 0
    for msg_id in msg_ids:
        try:
            await context.bot.delete_message(chat_id=chat_id, message_id=msg_id)
            deleted_count += 1
            logger.info("Deleted photo (%s) (chat %s, msg %s)", reason, chat_id, msg_id)
        except Exception as exc:
            logger.warning(
                "Failed to delete message %s in chat %s: %s. "
                "Check the bot has admin + 'Delete Messages' rights.",
                msg_id, chat_id, exc,
            )
    return deleted_count


async def resolve_keep_with_dedupe(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    new_msg_ids: list,
    new_confidence: float,
    new_hash,
    new_source_key,
) -> None:
    """
    Applies the rolling dedupe window on top of a provisionally-kept burst.
    A previously-kept burst only "beats" this one if it's (a) still within
    the dedupe window, (b) visually similar enough (same scene/event), AND
    (c) from the same source when source can be determined for both sides.
    The window always rolls forward to "now" on every new kept burst,
    whether it wins, loses, or is judged distinct.
    """
    if DEDUPE_WINDOW_SECONDS <= 0:
        _stats["kept"] += len(new_msg_ids)
        logger.info(
            "Kept burst of %d photo(s) with human/vehicle detected in at least one frame "
            "(chat %s, msgs %s, confidence=%.2f)",
            len(new_msg_ids), chat_id, new_msg_ids, new_confidence,
        )
        return

    now = time.monotonic()

    async with _recent_keep_lock:
        record = _recent_keeps.get(chat_id)
        window_active = record is not None and (now - record["timestamp"]) < DEDUPE_WINDOW_SECONDS

        distance = None
        same_scene = False
        if window_active and record.get("hash") is not None and new_hash is not None:
            distance = hamming_distance(record["hash"], new_hash)
            same_scene = distance <= SIMILARITY_HAMMING_THRESHOLD

        old_source = record.get("source_key") if record else None
        source_known = bool(old_source) and bool(new_source_key)
        same_source = (old_source == new_source_key) if source_known else True

        should_merge = window_active and same_scene and same_source

        if should_merge and record["confidence"] >= new_confidence:
            outcome = "new_superseded"
            to_delete = new_msg_ids
            _recent_keeps[chat_id] = {
                "msg_ids": record["msg_ids"],
                "confidence": record["confidence"],
                "timestamp": now,
                "hash": record["hash"],
                "source_key": old_source,
            }
        elif should_merge:
            outcome = "old_superseded"
            to_delete = record["msg_ids"]
            _recent_keeps[chat_id] = {
                "msg_ids": new_msg_ids,
                "confidence": new_confidence,
                "timestamp": now,
                "hash": new_hash,
                "source_key": new_source_key,
            }
        else:
            # No active window, different scene, or (when determinable) a
            # different camera source — treat as a distinct event. Nothing
            # gets deleted; the record moves forward to track this newest
            # event for future comparisons.
            outcome = "distinct_event" if window_active else "first_in_window"
            to_delete = []
            _recent_keeps[chat_id] = {
                "msg_ids": new_msg_ids,
                "confidence": new_confidence,
                "timestamp": now,
                "hash": new_hash,
                "source_key": new_source_key,
            }

    if outcome == "new_superseded":
        deleted = await _delete_messages(
            context, chat_id, to_delete,
            "duplicate human/vehicle detection within dedupe window; same scene/source, lower/equal confidence",
        )
        _stats["deleted"] += deleted
        logger.info(
            "New kept burst (chat %s, msgs %s, confidence=%.2f, hash_distance=%s, source_match=%s) "
            "judged a duplicate of an earlier, higher-confidence photo within the %ds dedupe "
            "window — deleted.",
            chat_id, new_msg_ids, new_confidence, distance, same_source, DEDUPE_WINDOW_SECONDS,
        )
        return

    if outcome == "old_superseded":
        deleted = await _delete_messages(
            context, chat_id, to_delete,
            "superseded by a higher-confidence duplicate of the same scene/source within dedupe window",
        )
        _stats["kept"] = max(0, _stats["kept"] - deleted)
        _stats["deleted"] += deleted
        logger.info(
            "Replaced previously kept photo(s) with a higher-confidence duplicate of the same "
            "scene/source within the %ds dedupe window (chat %s, hash_distance=%s).",
            DEDUPE_WINDOW_SECONDS, chat_id, distance,
        )

    elif outcome == "distinct_event":
        logger.info(
            "New kept burst (chat %s, msgs %s) fell inside the %ds dedupe window but was judged a "
            "different scene/source (hash_distance=%s, threshold=%d, source_match=%s) — kept as a "
            "separate event.",
            chat_id, new_msg_ids, DEDUPE_WINDOW_SECONDS, distance, SIMILARITY_HAMMING_THRESHOLD, same_source,
        )

    _stats["kept"] += len(new_msg_ids)
    logger.info(
        "Kept burst of %d photo(s) with human/vehicle detected in at least one frame "
        "(chat %s, msgs %s, confidence=%.2f)",
        len(new_msg_ids), chat_id, new_msg_ids, new_confidence,
    )


async def finalize_burst(context: ContextTypes.DEFAULT_TYPE, chat_id: int, items: list) -> None:
    """items is a list of (msg_id, decision, confidence, image_hash, source_key) tuples."""
    decisions = [d for _, d, _, _, _ in items]
    msg_ids = [m for m, _, _, _, _ in items]

    if "keep" in decisions:
        keep_items = [(m, c, h, s) for m, d, c, h, s in items if d == "keep"]
        # Use the frame with the strongest keep-confidence as the burst's
        # representative confidence/hash/source for dedupe comparison.
        _, keep_confidence, keep_hash, keep_source = max(keep_items, key=lambda t: t[1])
        # Prefer a source key from any frame in the burst if the top-confidence
        # frame didn't have one (e.g. caption was on the other photo).
        if not keep_source:
            keep_source = next((s for (_, _, _, s) in keep_items if s), None)
        await resolve_keep_with_dedupe(context, chat_id, msg_ids, keep_confidence, keep_hash, keep_source)
        return

    for msg_id, decision, _, _, _ in items:
        should_delete = decision == "delete" or (decision == "uncertain" and not KEEP_ON_UNCERTAIN)
        if should_delete:
            deleted = await _delete_messages(
                context, chat_id, [msg_id], f"burst-confirmed no human/vehicle; frame decision={decision}"
            )
            _stats["deleted"] += deleted
        else:
            _stats["kept"] += 1
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

    _stats["received"] += 1

    caption_source_key = extract_caption_source_key(message)

    photo = message.photo[-1]
    file = await context.bot.get_file(photo.file_id)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        await file.download_to_drive(tmp.name)
        decision, confidence, image_hash, ocr_source_key = await classify_image_async(tmp.name)

    # Prefer the caption stamp (cheap, reliable) over OCR (best-effort).
    source_key = caption_source_key or ocr_source_key

    chat_id = message.chat_id
    msg_id = message.message_id

    async with _pending_lock:
        bucket = _pending_bursts.setdefault(chat_id, {"items": [], "timer": None})
        bucket["items"].append((msg_id, decision, confidence, image_hash, source_key))

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


async def daily_summary_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    received = _stats["received"]
    kept = _stats["kept"]
    deleted = _stats["deleted"]

    text = (
        "📊 Daily Summary\n"
        f"Total images received: {received}\n"
        f"Classified as human/vehicle (kept): {kept}\n"
        f"Filtered out as animal/empty/duplicate (deleted): {deleted}"
    )

    try:
        await context.bot.send_message(chat_id=DAILY_SUMMARY_CHAT_ID, text=text)
        logger.info(
            "Daily summary sent to chat %s (received=%d, kept=%d, deleted=%d)",
            DAILY_SUMMARY_CHAT_ID, received, kept, deleted,
        )
    except Exception as exc:
        logger.warning("Failed to send daily summary to chat %s: %s", DAILY_SUMMARY_CHAT_ID, exc)

    # Reset for the next day regardless of whether the send succeeded, so a
    # transient failure doesn't cause double-counting into tomorrow's report.
    _stats["received"] = 0
    _stats["kept"] = 0
    _stats["deleted"] = 0


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

    if DAILY_SUMMARY_CHAT_ID:
        app.job_queue.run_daily(
            daily_summary_job,
            time=dt_time(hour=DAILY_SUMMARY_HOUR_UTC, minute=0),
        )
        logger.info(
            "Daily summary enabled: posting to chat %s at %02d:00 UTC",
            DAILY_SUMMARY_CHAT_ID, DAILY_SUMMARY_HOUR_UTC,
        )
    else:
        logger.info("Daily summary disabled (DAILY_SUMMARY_CHAT_ID not set)")

    if DEDUPE_WINDOW_SECONDS > 0:
        logger.info(
            "Rolling dedupe enabled: kept photos within %ds of each other in the same chat are "
            "collapsed to the single highest-confidence photo ONLY if they look like the same "
            "scene (hash distance <= %d/64) AND match on source (caption%s). Distinct scenes or "
            "sources are both kept.",
            DEDUPE_WINDOW_SECONDS, SIMILARITY_HAMMING_THRESHOLD,
            " + OCR" if OCR_ENABLED and _OCR_AVAILABLE else ", OCR disabled",
        )
    else:
        logger.info("Rolling dedupe disabled (DEDUPE_WINDOW_SECONDS=0)")

    logger.info("Bot started. Polling for new photos...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
