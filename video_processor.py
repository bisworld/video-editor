"""Underwater video analysis and montage (OpenCV + MoviePy + YOLO)."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional, Set, Tuple

import cv2
import numpy as np
from moviepy import VideoFileClip, concatenate_videoclips

ProgressCallback = Optional[Callable[[float, str], None]]

SEGMENT_DURATION = 3.0  # seconds per candidate clip

# COCO land / above-water classes → hard reject at conf >= 0.30
LAND_CLASS_NAMES: Set[str] = {
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}

# Optional aquatic names (COCO yolov8n has no "fish"; used if custom weights present)
FISH_CLASS_NAMES: Set[str] = {
    "fish",
    "animal",
}

# Diver gear / silhouette proxies (tanks often misclassified as backpack)
DIVER_GEAR_NAMES: Set[str] = {"backpack"}
DIVER_GEAR_CLASS_IDS: Set[int] = {24}  # COCO backpack

LAND_CONF = 0.30
PERSON_CONF = 0.08  # 8% — catch distant / partial divers
DIVER_GEAR_CONF = 0.10  # 10% — tanks / backpack proxy
YOLO_IMGSZ = 1280
PERSON_BUFFER_SEC = 2.0  # block ±2s around each diver hit (≈5s window)
UNDERWATER_MIN_RATIO = 0.60
PERSON_CLASS_ID = 0  # COCO "person"
DARK_V_MAX = 25
DARK_MOTION_AREA_RATIO = 0.08
BUBBLE_V_MIN = 220
BUBBLE_S_MAX = 35
BUBBLE_AREA_RATIO = 0.015  # 1.5% — localized exhaust plume

_yolo_model = None


@dataclass
class SegmentScore:
    start: float
    end: float
    sharpness: float
    fish_score: float
    landscape_score: float
    total: float
    has_people: bool = False


RESOLUTION_MAP = {
    "Оригинальное": None,
    "1080p (1920x1080) — Горизонтальное": (1920, 1080),
    "720p (1280x720) — Горизонтальное": (1280, 720),
    "1080x1920 (Shorts/Reels) — Вертикальное": (1080, 1920),
}


def _notify(cb: ProgressCallback, value: float, text: str) -> None:
    if cb:
        cb(min(max(value, 0.0), 1.0), text)


def get_yolo():
    """Lazy-load local YOLOv8n (weights cached as yolov8n.pt)."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO

        weights = os.path.join(os.path.dirname(__file__), "yolov8n.pt")
        if not os.path.isfile(weights):
            weights = "yolov8n.pt"
        _yolo_model = YOLO(weights)
    return _yolo_model


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def apply_clahe_bgr(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def underwater_ratio(frame: np.ndarray) -> float:
    """Blue/teal underwater pixels (HSV H:75–140, S:40–255, V:30–255)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (h >= 75) & (h <= 140) & (s >= 40) & (v >= 30)
    return float(mask.sum()) / float(mask.size)


def is_underwater(frame: np.ndarray, min_ratio: float = UNDERWATER_MIN_RATIO) -> bool:
    return underwater_ratio(frame) >= min_ratio


def yolo_scan(frame: np.ndarray, model) -> Tuple[bool, bool, float]:
    """
    Returns (has_land, has_diver, fish_yolo_score).
    has_diver: person (conf>=0.08) or backpack/tanks (conf>=0.10).
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(
        source=rgb,
        conf=min(PERSON_CONF, DIVER_GEAR_CONF),
        verbose=False,
        imgsz=YOLO_IMGSZ,
    )
    has_land = False
    has_diver = False
    fish_score = 0.0
    names = model.names

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            label = str(names.get(cls_id, "")).lower().strip()

            if (cls_id == PERSON_CLASS_ID or label == "person") and conf >= PERSON_CONF:
                has_diver = True
            elif (
                cls_id in DIVER_GEAR_CLASS_IDS or label in DIVER_GEAR_NAMES
            ) and conf >= DIVER_GEAR_CONF:
                has_diver = True
            elif label in LAND_CLASS_NAMES and conf >= LAND_CONF:
                has_land = True
            elif label in FISH_CLASS_NAMES and conf >= 0.20:
                fish_score += conf * 25.0

    return has_land, has_diver, fish_score


def detect_dark_wetsuit(frame: np.ndarray, fg_mask: np.ndarray) -> bool:
    """
    Large, monolithically dark moving blob (wetsuit): HSV V < 25 on MOG2
    motion covering > 8% of the frame.
    """
    h, w = fg_mask.shape[:2]
    total = float(h * w)
    if total < 1:
        return False

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_bin = motion > 0
    if not np.any(motion_bin):
        return False

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    dark_motion = motion_bin & (v < DARK_V_MAX)
    if float(dark_motion.sum()) / total >= DARK_MOTION_AREA_RATIO:
        return True

    # Contour-level: one large dark mover (sideways diver)
    contours, _ = cv2.findContours(
        motion.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for c in contours:
        area = float(cv2.contourArea(c))
        if area / total < DARK_MOTION_AREA_RATIO:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [c], -1, 255, thickness=-1)
        region = mask > 0
        if not np.any(region):
            continue
        if float(v[region].mean()) < DARK_V_MAX:
            return True
    return False


def detect_bubble_trail(frame: np.ndarray, fg_mask: np.ndarray) -> bool:
    """
    Dense exhaust bubble plume: near-white (V>220, S<35) moving pixels/contours
    covering > 1.5% of the frame. Overrides fish-motion scoring.
    """
    h, w = frame.shape[:2]
    total = float(h * w)
    if total < 1:
        return False

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    white = (v > BUBBLE_V_MIN) & (s < BUBBLE_S_MAX)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    motion = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_bin = motion > 0

    # Prefer moving whites; fall back to bright whites if MOG2 still warming
    if np.any(motion_bin):
        bubble_pix = white & motion_bin
    else:
        bubble_pix = white

    bubble_ratio = float(bubble_pix.sum()) / total
    if bubble_ratio >= BUBBLE_AREA_RATIO:
        return True

    # Localized cluster of small bright ovals (trail density)
    bubble_u8 = (bubble_pix.astype(np.uint8) * 255)
    bubble_u8 = cv2.morphologyEx(bubble_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(bubble_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_single = total * 0.04  # ignore huge specular glare panels
    min_single = max(4.0, total * 0.00005)
    cluster_area = 0.0
    small_count = 0
    for c in contours:
        area = float(cv2.contourArea(c))
        if min_single <= area <= max_single:
            x, y, bw, bh = cv2.boundingRect(c)
            aspect = bw / max(bh, 1)
            if 0.25 <= aspect <= 4.0:
                cluster_area += area
                small_count += 1

    if small_count >= 6 and (cluster_area / total) >= BUBBLE_AREA_RATIO:
        return True
    return False


def person_buffer_ranges(person_times: List[float], duration: float) -> List[Tuple[float, float]]:
    """Merge ±PERSON_BUFFER_SEC windows around each diver / gear / wetsuit / bubble hit."""
    if not person_times:
        return []
    windows = [
        (max(0.0, t - PERSON_BUFFER_SEC), min(duration, t + PERSON_BUFFER_SEC))
        for t in sorted(person_times)
    ]
    merged: List[Tuple[float, float]] = [windows[0]]
    for start, end in windows[1:]:
        prev_s, prev_e = merged[-1]
        if start <= prev_e:
            merged[-1] = (prev_s, max(prev_e, end))
        else:
            merged.append((start, end))
    return merged


def overlaps_any(start: float, end: float, ranges: List[Tuple[float, float]]) -> bool:
    for r0, r1 in ranges:
        if not (end <= r0 or start >= r1):
            return True
    return False


def motion_fish_score(fg_mask: np.ndarray, sensitivity: int) -> Tuple[float, np.ndarray]:
    """Medium-sized moving blobs (fish-like); ignore noise and huge camera sway."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    h, w = cleaned.shape[:2]
    total = float(h * w)
    min_area = max(35.0, 220.0 - sensitivity * 18.0)
    max_area = total * (0.015 + sensitivity * 0.004)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    area_sum = 0.0
    for c in contours:
        area = float(cv2.contourArea(c))
        if min_area <= area <= max_area:
            count += 1
            area_sum += area

    score = count * (8.0 + sensitivity) + (area_sum / total) * 120.0
    return float(score), cleaned


def landscape_score(frame: np.ndarray, motion_mask: Optional[np.ndarray] = None) -> float:
    """Clear blue-cyan underwater palette in low-motion zones."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    still = np.ones(h.shape, dtype=bool)
    if motion_mask is not None and motion_mask.shape[:2] == h.shape:
        still = motion_mask < 127
    if not np.any(still):
        return 0.0

    still_n = float(still.sum())
    blue_cyan = still & (h >= 75) & (h <= 140) & (s >= 40) & (v >= 30)
    purity = float(blue_cyan.sum()) / still_n

    if np.any(blue_cyan):
        sat = float(s[blue_cyan].mean()) / 255.0
        val = float(v[blue_cyan].mean()) / 255.0
    else:
        sat = 0.0
        val = 0.0

    coral = still & (((h <= 15) | (h >= 165)) & (s >= 70) & (v >= 50))
    coral_ratio = float(coral.sum()) / still_n

    return purity * 55.0 + sat * 35.0 + val * 5.0 + coral_ratio * 15.0


def crop_and_resize(frame: np.ndarray, target: Optional[Tuple[int, int]]) -> np.ndarray:
    if target is None:
        return frame

    tw, th = target
    h, w = frame.shape[:2]
    target_aspect = tw / th
    src_aspect = w / h

    if abs(src_aspect - target_aspect) < 0.01:
        cropped = frame
    elif src_aspect > target_aspect:
        new_w = int(h * target_aspect)
        x0 = max(0, (w - new_w) // 2)
        cropped = frame[:, x0 : x0 + new_w]
    else:
        new_h = int(w / target_aspect)
        y0 = max(0, (h - new_h) // 2)
        cropped = frame[y0 : y0 + new_h, :]

    return cv2.resize(cropped, (tw, th), interpolation=cv2.INTER_AREA)


def rank_score(
    fish: float,
    landscape: float,
    sharpness: float,
    priority: str,
) -> float:
    sharp_n = min(sharpness / 200.0, 1.0) * 20.0
    if priority == "Больше рыб":
        return fish * 2.5 + landscape * 0.5 + sharp_n
    if priority == "Больше пейзажей":
        return landscape * 2.5 + fish * 0.4 + sharp_n
    return fish * 1.2 + landscape * 1.2 + sharp_n


def analyze_video(
    video_path: str,
    blur_filter: int,
    fish_sensitivity: int,
    exclude_people: bool,
    priority: str,
    progress: ProgressCallback = None,
) -> List[SegmentScore]:
    """
    Hard rule: diver safety first. Fish/landscape scores are computed ONLY
    for frames that pass person/backpack/bubble/wetsuit checks. Blacklisted
    timestamps (±2s buffer) never enter the montage, regardless of fish score.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видеофайл.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count else 0.0
    step = max(1, int(round(fps / 8.0)))
    yolo_every = max(step, int(round(fps / 1.0)))
    safety_every = max(1, int(round(fps / 3.0))) if exclude_people else yolo_every

    _notify(progress, 0.03, "Шаг 1/3: Загрузка YOLO (yolov8n)...")
    model = get_yolo()

    sharpness_threshold = float(blur_filter) * 1.5
    mog_var = max(8.0, 40.0 - fish_sensitivity * 2.5)

    segments: List[SegmentScore] = []
    blacklist_times: List[float] = []
    t = 0.0
    seg_idx = 0
    total_segs = max(1, int(duration / SEGMENT_DURATION))

    while t + 1.5 < duration:
        end = min(t + SEGMENT_DURATION, duration)
        start_frame = int(t * fps)
        end_frame = max(start_frame + 1, int(end * fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        subtractor = cv2.createBackgroundSubtractorMOG2(
            history=60,
            varThreshold=mog_var,
            detectShadows=False,
        )

        samples_sharp: List[float] = []
        samples_fish: List[float] = []
        samples_land: List[float] = []
        underwater_ok = 0
        underwater_fail = 0
        hard_reject = False
        blacklisted = False
        warmed = 0
        local_i = 0

        while True:
            cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if cur >= end_frame:
                break
            ok, frame = cap.read()
            if not ok:
                break

            frame_time = cur / fps
            if local_i % step != 0:
                local_i += 1
                continue

            run_safety_yolo = (local_i % safety_every == 0) or warmed == 0
            run_land_yolo = (local_i % yolo_every == 0) or warmed == 0
            local_i += 1

            if not is_underwater(frame, UNDERWATER_MIN_RATIO):
                underwater_fail += 1
                if underwater_fail >= 2:
                    hard_reject = True
                    break
                continue
            underwater_ok += 1

            fg = subtractor.apply(frame)
            warmed += 1
            if warmed < 2:
                continue

            # 1) SAFETY FIRST — fish scores only after this passes
            if exclude_people:
                if detect_bubble_trail(frame, fg) or detect_dark_wetsuit(frame, fg):
                    blacklist_times.append(frame_time)
                    blacklisted = True
                    continue

                if run_safety_yolo:
                    has_land, has_diver, _ = yolo_scan(frame, model)
                    if has_diver:
                        blacklist_times.append(frame_time)
                        blacklisted = True
                        continue
                    if has_land:
                        hard_reject = True
                        break
            else:
                if run_land_yolo:
                    has_land, _, _ = yolo_scan(frame, model)
                    if has_land:
                        hard_reject = True
                        break

            if blacklisted:
                continue

            # 2) SAFE ONLY: sharpness + fish + landscape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = laplacian_variance(gray)
            samples_sharp.append(sharp)
            if sharp < sharpness_threshold:
                continue

            fish_motion, motion_mask = motion_fish_score(fg, fish_sensitivity)
            land = landscape_score(frame, motion_mask)
            fish_yolo = 0.0
            if run_land_yolo:
                _, _, fish_yolo = yolo_scan(frame, model)

            samples_fish.append(fish_motion + fish_yolo)
            samples_land.append(land)

        if blacklisted or hard_reject:
            seg_idx += 1
            _notify(
                progress,
                0.05 + 0.55 * (seg_idx / max(total_segs, 1)),
                "Шаг 1/3: Блок дайвера / суши (балл = 0)...",
            )
            t = end
            continue

        uw_total = underwater_ok + underwater_fail
        uw_ratio = (underwater_ok / uw_total) if uw_total else 0.0
        avg_sharp = float(np.mean(samples_sharp)) if samples_sharp else 0.0

        if (
            uw_ratio < UNDERWATER_MIN_RATIO
            or underwater_ok == 0
            or avg_sharp < sharpness_threshold
            or not samples_fish
        ):
            pass
        else:
            fish = float(np.mean(samples_fish))
            land = float(np.mean(samples_land))
            total = rank_score(fish, land, avg_sharp, priority)
            segments.append(
                SegmentScore(
                    start=t,
                    end=end,
                    sharpness=avg_sharp,
                    fish_score=fish,
                    landscape_score=land,
                    total=total,
                    has_people=False,
                )
            )

        seg_idx += 1
        _notify(
            progress,
            0.05 + 0.55 * (seg_idx / max(total_segs, 1)),
            "Шаг 1/3: YOLO + анализ рыб и сюжета...",
        )
        t = end

    cap.release()

    # ±2s buffer: neighbors forced out even if they only showed fish
    if exclude_people and blacklist_times:
        banned = person_buffer_ranges(blacklist_times, duration)
        segments = [
            s for s in segments
            if not overlaps_any(s.start, s.end, banned) and s.total > 0
        ]

    segments.sort(key=lambda s: s.total, reverse=True)
    return segments


def select_segments(candidates: List[SegmentScore], target_duration: float) -> List[SegmentScore]:
    selected: List[SegmentScore] = []
    used: List[Tuple[float, float]] = []
    total = 0.0

    for seg in candidates:
        if seg.total <= 0:
            continue
        if total >= target_duration:
            break
        overlap = any(not (seg.end <= u0 or seg.start >= u1) for u0, u1 in used)
        if overlap:
            continue
        selected.append(seg)
        used.append((seg.start, seg.end))
        total += seg.end - seg.start

    selected.sort(key=lambda s: s.start)
    return selected


def _process_frame_rgb(
    frame_rgb: np.ndarray,
    enhance: bool,
    target: Optional[Tuple[int, int]],
) -> np.ndarray:
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if enhance:
        bgr = apply_clahe_bgr(bgr)
    bgr = crop_and_resize(bgr, target)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_montage(
    video_path: str,
    segments: List[SegmentScore],
    resolution_label: str,
    enhance_color: bool,
    output_path: str,
    progress: ProgressCallback = None,
) -> str:
    if not segments:
        raise RuntimeError(
            "Не найдено подходящих фрагментов. Ослабьте фильтр мутности "
            "или отключите «Исключить людей»."
        )

    target = RESOLUTION_MAP.get(resolution_label)
    _notify(progress, 0.65, "Шаг 2/3: Цветокоррекция и склеивание...")

    source = VideoFileClip(video_path)
    clips = []
    n = len(segments)
    max_t = max(0.0, float(source.duration or 0.0) - 1e-3)

    try:
        for i, seg in enumerate(segments):
            start = max(0.0, min(float(seg.start), max_t))
            end = max(start + 0.05, min(float(seg.end), max_t))
            if end <= start:
                continue
            sub = source.subclipped(start, end).without_audio()

            def transform(frame, _enhance=enhance_color, _target=target):
                return _process_frame_rgb(frame, _enhance, _target)

            sub = sub.image_transform(transform)
            clips.append(sub)
            _notify(
                progress,
                0.65 + 0.25 * ((i + 1) / n),
                "Шаг 2/3: Цветокоррекция и склеивание...",
            )

        if not clips:
            raise RuntimeError("Не удалось вырезать фрагменты из видео.")

        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(
            output_path,
            codec="libx264",
            audio=False,
            fps=source.fps or 25,
            preset="medium",
            logger=None,
        )
        final.close()
    finally:
        for c in clips:
            try:
                c.close()
            except Exception:
                pass
        source.close()

    _notify(progress, 1.0, "Готово!")
    return output_path


def process_video(
    video_path: str,
    target_duration: int,
    resolution: str,
    priority: str,
    fish_sensitivity: int,
    exclude_people: bool,
    enhance_color: bool,
    blur_filter: int,
    output_path: Optional[str] = None,
    progress: ProgressCallback = None,
) -> str:
    _notify(progress, 0.02, "Шаг 1/3: Анализ YOLO, резкости и сюжета...")

    candidates = analyze_video(
        video_path=video_path,
        blur_filter=blur_filter,
        fish_sensitivity=fish_sensitivity,
        exclude_people=exclude_people,
        priority=priority,
        progress=progress,
    )
    selected = select_segments(candidates, float(target_duration))

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".mp4", prefix="underwater_out_")
        os.close(fd)

    return build_montage(
        video_path=video_path,
        segments=selected,
        resolution_label=resolution,
        enhance_color=enhance_color,
        output_path=output_path,
        progress=progress,
    )
