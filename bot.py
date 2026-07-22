"""
Telegram Animal-Image Filter Bot — MegaDetector edition
--------------------------------------------------------
Watches a Telegram group/channel via TWO listeners that feed the same
classification pipeline:

  1. The main bot (python-telegram-bot / Bot API) — sees photos sent by
     humans and by itself.
  2. A "userbot" listener (Telethon, using a real Telegram user account) —
     sees photos sent by OTHER bots in the group.

Why two listeners: Telegram's Bot API deliberately does not deliver a bot
messages authored by a DIFFERENT bot (to prevent bot-loops). Etosha's camera
alerts are posted into the main group by a separate Google Apps Script bot,
so our filter bot would otherwise never see them at all — not a detection
bug, a hard platform restriction. The Telethon listener, running as a normal
user account, has no such restriction and sees everything; it hands off
detected photos into the exact same classify -> burst -> dedupe pipeline.

Deletion routing: the same bot-isolation restriction that blocks a bot from
SEEING another bot's messages also appears to block it from being able to
DELETE one (Telegram's deleteMessage returned "message not found" 100% of
the time for Apps-Script-bot-authored photos, even with admin + Delete
Messages rights, while deletes of human-authored photos always succeeded).
So deletion is routed by whichever listener originally spotted the photo:
photos seen by the main Bot API bot are deleted by that bot; photos seen by
the Telethon userbot are deleted by the userbot account itself. The userbot
account therefore needs its own admin + Delete Messages rights in the group
(no longer just plain membership).

Everything below the two entry points (handle_photo for the Bot API side,
handle_userbot_photo for the Telethon side) is shared, unchanged pipeline
logic:
  - CLAHE image enhancement for detection accuracy.
  - MegaDetectorV6 (PytorchWildlife) classification: animal / person /
    vehicle, run on a background thread.
  - Multi-frame burst confirmation (cameras send ~2 photos per event).
  - Rolling dedupe window: a later kept burst is only treated as a
    duplicate of an earlier one if it matches BOTH on visual scene
    (perceptual hash) and on source (caption text, or OCR of a burned-in
    watermark if enabled) — never merged purely because two events share a
    class label.
  - Health-check heartbeat and daily summary digest.

No per-photo tagging/reply messages are sent — the bot only acts silently
(deletes or leaves photos alone), aside from the health-check heartbeat and
the daily summary.

Run mode: polling (no public URL/SSL needed).
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
BURST_WINDOW_SECONDS = 4.0
MAX_BURST_SIZE = 2

# --- Rolling dedupe window for kept (human/vehicle) bursts -------------------
DEDUPE_WINDOW_SECONDS = int(os.environ.get("DEDUPE_WINDOW_SECONDS", str(5 * 60)))
SIMILARITY_HAMMING_THRESHOLD = int(os.environ.get("SIMILARITY_HAMMING_THRESHOLD", "10"))

# --- Source/camera-stamp matching for dedupe ---------------------------------
OCR_ENABLED = os.environ.get("OCR_ENABLED", "false").strip().lower() == "true"
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
HEALTH_CHECK_MESSAGE = "AI running"

# --- Daily summary digest ----------------------------------------------------
_daily_summary_chat_id_raw = os.environ.get("DAILY_SUMMARY_CHAT_ID", _health_check_chat_id_raw)
DAILY_SUMMARY_CHAT_ID = int(_daily_summary_chat_id_raw) if _daily_summary_chat_id_raw else None
DAILY_SUMMARY_HOUR_UTC = int(os.environ.get("DAILY_SUMMARY_HOUR_UTC", "6"))

# --- Userbot listener (Telethon) — sees photos posted by OTHER bots --------
# Needed because Telegram's Bot API never delivers a bot messages authored
# by a different bot. Get TELETHON_API_ID / TELETHON_API_HASH once, for
# free, at https://my.telegram.org (API Development Tools). Generate
# TELETHON_SESSION_STRING once via the separate generate_telethon_session.py
# helper script, run locally/interactively (never on Railway) — see that
# script's docstring for the one-time login steps. The account used needs
# its own admin + Delete Messages rights in the monitored group: it deletes
# the photos it detects itself, since the main bot cannot delete messages
# authored by another bot even with admin rights (see module docstring).
TELETHON_API_ID = int(os.environ.get("TELETHON_API_ID", "0") or 0)
TELETHON_API_HASH = os.environ.get("TELETHON_API_HASH", "")
TELETHON_SESSION_STRING = os.environ.get("TELETHON_SESSION_STRING", "")
_userbot_chat_id_raw = os.environ.get("USERBOT_CHAT_ID", _health_check_chat_id_raw)
USERBOT_CHAT_ID = int(_userbot_chat_id_raw) if _userbot_chat_id_raw else None

USERBOT_LISTENER_ENABLED = bool(
    TELETHON_API_ID and TELETHON_API_HASH and TELETHON_SESSION_STRING and USERBOT_CHAT_ID
)

try:
    OWN_BOT_USER_ID = int(BOT_TOKEN.split(":")[0])
except (ValueError, IndexError):
    OWN_BOT_USER_ID = None

# ---------------------------------------------------------------------------

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("animal-filter-bot")

if OCR_ENABLED and not _OCR_AVAILABLE:
    logger.warning(
        "OCR_ENABLED=true but pytesseract/Pillow (or the tesseract-ocr binary) are not "
        "installed — falling back to caption-only source matching."
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

_recent_keeps: dict = {}
_recent_keep_lock = asyncio.Lock()

_stats = {
    "received": 0,
    "kept": 0,
    "deleted": 0,
}

# Set once each client is up. Deletion is routed by which listener spotted
# the photo: _ptb_bot_ref deletes Bot-API-seen (human) photos, _userbot_client_ref
# deletes Telethon-seen (other-bot) photos — see module docstring for why.
_ptb_bot_ref = None
_userbot_client_ref = None

_DATE_TIME_PATTERN = re.compile(
    r"\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|\d{1,2}:\d{2}(:\d{2})?"
)


def normalize_source_text(text):
    if not text:
        return None
    cleaned = _DATE_TIME_PATTERN.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned or None


def extract_caption_source_key(message):
    caption = getattr(message, "caption", None) or getattr(message, "text", None)
    return normalize_source_text(caption)


def extract_ocr_source_key(image_path: str):
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
        "Could not parse MegaDetector result into (label, confidence) pairs — raw result: %r.",
        result,
    )
    return pairs


def classify_image(image_path: str):
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


async def _delete_one(chat_id: int, msg_id: int, listener: str) -> bool:
    """Deletes a single message using whichever client actually SAW it.
    Photos spotted by the main Bot API bot are deleted by that bot; photos
    spotted by the Telethon userbot are deleted by the userbot account
    itself. Mixing these up is exactly what caused the "message to delete
    not found" failures seen on every Apps-Script-bot-authored photo — the
    main bot appears unable to delete a message it was never shown, even
    with admin + Delete Messages rights."""
    try:
        if listener == "userbot":
            if _userbot_client_ref is None:
                logger.warning(
                    "Cannot delete msg %s in chat %s via userbot: userbot client not ready. "
                    "Falling back to the main bot (may fail with 'not found').",
                    msg_id, chat_id,
                )
                await _ptb_bot_ref.delete_message(chat_id=chat_id, message_id=msg_id)
            else:
                await _userbot_client_ref.delete_messages(chat_id, msg_id, revoke=True)
        else:
            await _ptb_bot_ref.delete_message(chat_id=chat_id, message_id=msg_id)
        return True
    except Exception as exc:
        logger.warning(
            "Failed to delete message %s in chat %s (via %s): %s. "
            "Check that account has admin + 'Delete Messages' rights.",
            msg_id, chat_id, listener, exc,
        )
        return False


async def _delete_messages(chat_id: int, items: list, reason: str) -> int:
    """items is a list of (msg_id, listener) tuples."""
    deleted_count = 0
    for msg_id, listener in items:
        if await _delete_one(chat_id, msg_id, listener):
            deleted_count += 1
            logger.info("Deleted photo (%s) (chat %s, msg %s, via %s)", reason, chat_id, msg_id, listener)
    return deleted_count


async def resolve_keep_with_dedupe(chat_id: int, new_items: list, new_confidence: float, new_hash, new_source_key) -> None:
    """new_items is a list of (msg_id, listener) tuples for this kept burst."""
    new_msg_ids = [m for m, _ in new_items]

    if DEDUPE_WINDOW_SECONDS <= 0:
        _stats["kept"] += len(new_items)
        logger.info(
            "Kept burst of %d photo(s) with human/vehicle detected in at least one frame "
            "(chat %s, msgs %s, confidence=%.2f)",
            len(new_items), chat_id, new_msg_ids, new_confidence,
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
            to_delete = new_items
            _recent_keeps[chat_id] = {
                "msg_ids": record["msg_ids"], "confidence": record["confidence"],
                "timestamp": now, "hash": record["hash"], "source_key": old_source,
            }
        elif should_merge:
            outcome = "old_superseded"
            to_delete = record["msg_ids"]
            _recent_keeps[chat_id] = {
                "msg_ids": new_items, "confidence": new_confidence,
                "timestamp": now, "hash": new_hash, "source_key": new_source_key,
            }
        else:
            outcome = "distinct_event" if window_active else "first_in_window"
            to_delete = []
            _recent_keeps[chat_id] = {
                "msg_ids": new_items, "confidence": new_confidence,
                "timestamp": now, "hash": new_hash, "source_key": new_source_key,
            }

    if outcome == "new_superseded":
        deleted = await _delete_messages(
            chat_id, to_delete,
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
            chat_id, to_delete,
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

    _stats["kept"] += len(new_items)
    logger.info(
        "Kept burst of %d photo(s) with human/vehicle detected in at least one frame "
        "(chat %s, msgs %s, confidence=%.2f)",
        len(new_items), chat_id, new_msg_ids, new_confidence,
    )


async def finalize_burst(chat_id: int, items: list) -> None:
    """items is a list of (msg_id, decision, confidence, image_hash, source_key, listener) tuples."""
    decisions = [d for _, d, _, _, _, _ in items]

    if "keep" in decisions:
        keep_items = [(m, c, h, s, l) for m, d, c, h, s, l in items if d == "keep"]
        _, keep_confidence, keep_hash, keep_source, _ = max(keep_items, key=lambda t: t[1])
        if not keep_source:
            keep_source = next((s for (_, _, _, s, _) in keep_items if s), None)
        keep_msg_items = [(m, l) for m, _, _, _, l in keep_items]
        await resolve_keep_with_dedupe(chat_id, keep_msg_items, keep_confidence, keep_hash, keep_source)
        return

    for msg_id, decision, _, _, _, listener in items:
        should_delete = decision == "delete" or (decision == "uncertain" and not KEEP_ON_UNCERTAIN)
        if should_delete:
            deleted = await _delete_messages(
                chat_id, [(msg_id, listener)], f"burst-confirmed no human/vehicle; frame decision={decision}"
            )
            _stats["deleted"] += deleted
        else:
            _stats["kept"] += 1
            logger.info("Kept photo (uncertain, KEEP_ON_UNCERTAIN=True) (chat %s, msg %s)", chat_id, msg_id)


async def _finalize_after_delay(chat_id: int) -> None:
    try:
        await asyncio.sleep(BURST_WINDOW_SECONDS)
    except asyncio.CancelledError:
        return

    async with _pending_lock:
        bucket = _pending_bursts.pop(chat_id, None)

    if bucket and bucket["items"]:
        await finalize_burst(chat_id, bucket["items"])


async def _ingest_classified_photo(chat_id: int, msg_id: int, decision: str, confidence: float, image_hash, source_key, listener: str) -> None:
    """Shared burst-bucketing logic used by BOTH the Bot API photo handler
    and the Telethon userbot listener, so a photo is treated identically no
    matter which listener spotted it. `listener` ("bot" or "userbot") is
    carried through so the eventual delete call uses the client that can
    actually see/act on that specific message."""
    async with _pending_lock:
        bucket = _pending_bursts.setdefault(chat_id, {"items": [], "timer": None})
        bucket["items"].append((msg_id, decision, confidence, image_hash, source_key, listener))

        if bucket["timer"] is not None:
            bucket["timer"].cancel()

        if len(bucket["items"]) >= MAX_BURST_SIZE:
            items = bucket["items"]
            del _pending_bursts[chat_id]
            finalize_now = True
        else:
            bucket["timer"] = asyncio.create_task(_finalize_after_delay(chat_id))
            finalize_now = False

    if finalize_now:
        await finalize_burst(chat_id, items)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Bot API entry point — sees photos sent by humans and by this bot itself."""
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

    source_key = caption_source_key or ocr_source_key

    await _ingest_classified_photo(
        message.chat_id, message.message_id, decision, confidence, image_hash, source_key, listener="bot"
    )


async def handle_userbot_photo(event) -> None:
    """Telethon entry point — sees photos posted by OTHER bots (e.g. the
    Google Apps Script relay bot), which the Bot API side can never see.

    Registered WITHOUT a chats= filter on purpose: Telethon's internal chat-ID
    representation doesn't always match the "-100..." form the Bot API uses
    for the same chat, and if a chats= filter doesn't match, the handler
    silently never fires at all (no error, nothing in the logs). Filtering
    manually here instead means a mismatch is loud and diagnosable rather
    than silent.
    """
    chat_id = event.chat_id

    if USERBOT_CHAT_ID is not None and chat_id != USERBOT_CHAT_ID:
        if event.photo:
            chat_title = getattr(event.chat, "title", None)
            logger.info(
                "Userbot saw a photo in a chat that doesn't match USERBOT_CHAT_ID "
                "(configured=%s, actual=%s, chat_title=%r). If %r is actually the "
                "intended group, set USERBOT_CHAT_ID=%s and redeploy.",
                USERBOT_CHAT_ID, chat_id, chat_title, chat_title, chat_id,
            )
        return

    if not event.photo:
        return

    sender = await event.get_sender()
    is_bot = bool(getattr(sender, "bot", False))
    if not is_bot:
        return  # human-sent photos are already handled by the Bot API side

    sender_id = getattr(sender, "id", None)
    if OWN_BOT_USER_ID and sender_id == OWN_BOT_USER_ID:
        return  # our own bot's own photos are already handled by the Bot API side

    if _ptb_bot_ref is None:
        logger.warning("Userbot listener saw a photo but the main bot isn't ready yet; skipping.")
        return

    _stats["received"] += 1
    caption_source_key = normalize_source_text(getattr(event.message, "message", None))

    tmp_path = tempfile.mktemp(suffix=".jpg")
    try:
        await event.download_media(file=tmp_path)
        decision, confidence, image_hash, ocr_source_key = await classify_image_async(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    source_key = caption_source_key or ocr_source_key

    await _ingest_classified_photo(
        event.chat_id, event.message.id, decision, confidence, image_hash, source_key, listener="userbot"
    )


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
        "Daily Summary\n"
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

    _stats["received"] = 0
    _stats["kept"] = 0
    _stats["deleted"] = 0


async def run_bot() -> None:
    global _ptb_bot_ref, _userbot_client_ref

    if BOT_TOKEN == "PASTE_YOUR_BOT_TOKEN_HERE":
        raise SystemExit(
            "Set your bot token first: either edit BOT_TOKEN in bot.py, "
            "or set the TELEGRAM_BOT_TOKEN environment variable."
        )

    # connection_pool_size bumped from the httpx default: under bursty
    # activity (several deletes firing near-simultaneously), the default
    # pool size was previously observed to exhaust and cause a delete
    # attempt to fail outright ("Pool timeout: All connections in the
    # connection pool are occupied"), leaving that one photo un-deleted.
    request = HTTPXRequest(
        connect_timeout=15.0, read_timeout=30.0, write_timeout=15.0,
        connection_pool_size=20,
    )

    application = Application.builder().token(BOT_TOKEN).request(request).build()
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    if HEALTH_CHECK_CHAT_ID:
        application.job_queue.run_repeating(
            health_check_job, interval=HEALTH_CHECK_INTERVAL_SECONDS, first=HEALTH_CHECK_INTERVAL_SECONDS,
        )
        logger.info(
            "Health check enabled: posting to chat %s every %d hours",
            HEALTH_CHECK_CHAT_ID, HEALTH_CHECK_INTERVAL_SECONDS // 3600,
        )
    else:
        logger.info("Health check disabled (HEALTH_CHECK_CHAT_ID not set)")

    if DAILY_SUMMARY_CHAT_ID:
        application.job_queue.run_daily(daily_summary_job, time=dt_time(hour=DAILY_SUMMARY_HOUR_UTC, minute=0))
        logger.info(
            "Daily summary enabled: posting to chat %s at %02d:00 UTC",
            DAILY_SUMMARY_CHAT_ID, DAILY_SUMMARY_HOUR_UTC,
        )
    else:
        logger.info("Daily summary disabled (DAILY_SUMMARY_CHAT_ID not set)")

    if DEDUPE_WINDOW_SECONDS > 0:
        logger.info(
            "Rolling dedupe enabled: kept photos within %ds of each other in the same chat are "
            "collapsed to the single highest-confidence photo ONLY if they match on scene "
            "(hash distance <= %d/64) AND source (caption%s).",
            DEDUPE_WINDOW_SECONDS, SIMILARITY_HAMMING_THRESHOLD,
            " + OCR" if OCR_ENABLED and _OCR_AVAILABLE else ", OCR disabled",
        )
    else:
        logger.info("Rolling dedupe disabled (DEDUPE_WINDOW_SECONDS=0)")

    await application.initialize()
    await application.start()
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    _ptb_bot_ref = application.bot
    logger.info("Bot started. Polling for new photos...")

    userbot_client = None
    if USERBOT_LISTENER_ENABLED:
        from telethon import TelegramClient, events
        from telethon.sessions import StringSession

        userbot_client = TelegramClient(
            StringSession(TELETHON_SESSION_STRING), TELETHON_API_ID, TELETHON_API_HASH
        )
        # No chats= filter here on purpose — see handle_userbot_photo's
        # docstring for why. The chat match is done manually inside the
        # handler instead, so a chat-ID mismatch is visible in the logs
        # rather than silently dropping every event.
        userbot_client.add_event_handler(handle_userbot_photo, events.NewMessage())
        await userbot_client.start()
        _userbot_client_ref = userbot_client
        me = await userbot_client.get_me()
        logger.info(
            "Userbot listener active as %s (id=%s), watching chat %s for photos posted by OTHER bots. "
            "Deletions of those photos will be performed by this account directly (requires admin + "
            "Delete Messages rights in the group).",
            getattr(me, "username", None) or getattr(me, "first_name", "unknown"), me.id, USERBOT_CHAT_ID,
        )
    else:
        logger.info(
            "Userbot listener disabled (set TELETHON_API_ID, TELETHON_API_HASH, "
            "TELETHON_SESSION_STRING, and USERBOT_CHAT_ID to enable it)."
        )

    try:
        await asyncio.Event().wait()  # run forever
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        logger.info("Shutting down...")
        if userbot_client is not None:
            await userbot_client.disconnect()
        await application.updater.stop()
        await application.stop()
        await application.shutdown()


def main() -> None:
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
