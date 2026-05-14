"""
Colony Counting and Hemolysis Detection API
Version: 1.9.2 - OpenCFU-inspired pipeline with density-channel architecture

Algorithm based on OpenCFU (Geissmann 2013, PLOS ONE):
1. Local background subtraction (large Gaussian) — handles uneven illumination
2. Threshold ladder with score-map accumulation — robust multi-level detection
3. Density pre-scan → route to LOW / MEDIUM / HIGH channel
4. Channel-specific extraction, splitting, and filtering
5. Robust circularity filtering (moments-based for small colonies)

Detection mode ('sensitive', 'strict', 'auto') maps to a continuous
sensitivity parameter [0, 1] that makes small adjustments. The core
algorithm is the same regardless of mode.

v1.9.1 changes:
- Density-channel architecture: LOW / MEDIUM / HIGH channels with independent params
- LOW channel: hardcoded 0.19 threshold, 250px watershed, edge/center/contrast noise filter, min_circ 0.40
- MEDIUM channel: v1.9.0 settings preserved (0.85 threshold, 80px watershed)
- HIGH channel: unchanged from v1.9.0
v1.9.2: LOW channel tuning — hardcode score_threshold 0.19, undersized 40% median, contrast filter 15 lum units
"""

import os
import logging
import traceback
import base64
import uuid
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
from scipy.ndimage import (
    median_filter, gaussian_filter, label, find_objects,
    distance_transform_edt, maximum_filter,
    binary_closing, binary_opening, binary_fill_holes,
    binary_erosion, binary_dilation,
)
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Supabase client (optional — app still runs without it)
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_KEY')

supabase = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        logger.info("Supabase client initialized")
    except Exception as e:
        logger.error("Failed to initialize Supabase client: %s", e)
else:
    logger.warning("Supabase not configured — image retention disabled")


# Reliable image MIME → extension mapping (avoids mimetypes' .jpe/.jfif quirks)
_IMAGE_EXTENSIONS = {
    'image/jpeg': '.jpg',
    'image/jpg':  '.jpg',
    'image/pjpeg': '.jpg',
    'image/png':  '.png',
    'image/tiff': '.tiff',
    'image/tif':  '.tiff',
    'image/webp': '.webp',
    'image/gif':  '.gif',
    'image/bmp':  '.bmp',
    'image/heic': '.heic',
    'image/heif': '.heif',
    'image/avif': '.avif',
}
_DEFAULT_CONTENT_TYPE = 'image/jpeg'
_DEFAULT_EXTENSION = '.jpg'


def _resolve_content_type(content_type):
    """Normalize a content_type string to (content_type, extension).
    Falls back to image/jpeg / .jpg when detection fails.
    """
    if not content_type:
        return _DEFAULT_CONTENT_TYPE, _DEFAULT_EXTENSION
    # Strip parameters like "; charset=..." and normalize case
    ct = content_type.split(';', 1)[0].strip().lower()
    ext = _IMAGE_EXTENSIONS.get(ct)
    if ext:
        return ct, ext
    return _DEFAULT_CONTENT_TYPE, _DEFAULT_EXTENSION


def _parse_image_base64(b64_str):
    """Decode a possibly-data-URL base64 string.
    Returns (image_bytes, detected_content_type_or_None).
    Accepts both raw base64 and 'data:<mime>;base64,<payload>' forms.
    """
    detected = None
    if isinstance(b64_str, str) and b64_str.startswith('data:'):
        header, sep, body = b64_str.partition(',')
        if sep:
            # header looks like 'data:image/png;base64'
            meta = header[5:]
            detected = meta.split(';', 1)[0] or None
            b64_str = body
    return base64.b64decode(b64_str), detected


def store_upload(image_bytes, count, density_channel, user_email=None,
                 image_metadata=None, content_type=None):
    """Persist uploaded image to Supabase Storage and log the analysis.
    The filename extension and stored Content-Type are derived from the
    caller-provided content_type, defaulting to image/jpeg.
    """
    resolved_ct, ext = _resolve_content_type(content_type)
    filename = f"{uuid.uuid4()}{ext}"

    supabase.storage.from_("colony-uploads").upload(
        path=filename,
        file=image_bytes,
        file_options={"content-type": resolved_ct}
    )

    supabase.table("colony_analyses").insert({
        "user_email": user_email,
        "image_path": filename,
        "colony_count": count,
        "density_channel": density_channel,
        "image_metadata": image_metadata,
    }).execute()


app = Flask(__name__)
CORS(app)

# Configuration
MAX_IMAGE_SIZE = 1600
FALLBACK_IMAGE_SIZE = 800
SUPPORTED_MEDIA_TYPES = ['blood_agar', 'nutrient_agar', 'macconkey_agar']
DETECTION_MODES = ['sensitive', 'strict', 'auto']
COLONY_SIZES = ['small', 'medium', 'large', 'auto']

# Unified pipeline parameters
PIPELINE_PARAMS = {
    'bg_sigma_fraction': 0.08,
    'n_thresholds': 30,
    'ladder_min_area': 6,
    'ladder_max_area': 6000,
    'ladder_min_circularity': 0.12,
    'score_threshold_fraction': 0.19,
    'watershed_footprint': 7,
    'watershed_min_distance': 2.0,
    'min_colony_area': 8,
    'max_colony_area': 5000,
    'min_circularity': 0.30,
}

# Backward-compatible mode mapping
MODE_SENSITIVITY = {
    'sensitive': 0.7,
    'strict': 0.3,
    'auto': 0.5,
}


def _adjust_params_for_sensitivity(sensitivity, colony_size='auto'):
    """Adjust pipeline parameters based on sensitivity and colony_size hint."""
    p = dict(PIPELINE_PARAMS)
    p['score_threshold_fraction'] = 0.29 - 0.20 * sensitivity
    p['min_colony_area'] = int(14 - 8 * sensitivity)
    p['min_circularity'] = 0.35 - 0.13 * sensitivity
    p['oversized_segment_threshold'] = 80
    p['plate_radius_factor'] = 0.88

    if colony_size == 'small':
        p['min_colony_area'] = max(3, int(p['min_colony_area'] * 0.6))
        p['score_threshold_fraction'] = max(0.05, p['score_threshold_fraction'] - 0.03)
        p['oversized_segment_threshold'] = 60
        p['plate_radius_factor'] = 0.93
    elif colony_size == 'large':
        p['min_colony_area'] = int(p['min_colony_area'] * 1.3)
        p['score_threshold_fraction'] = p['score_threshold_fraction'] + 0.02
        p['oversized_segment_threshold'] = 200

    return p


# === IMAGE LOADING ===

def load_and_resize_image(image_bytes, max_size=None):
    """Load image and resize to manageable size"""
    if max_size is None:
        max_size = MAX_IMAGE_SIZE

    img = Image.open(BytesIO(image_bytes))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.info("Resized image from %dx%d to %dx%d (max_size=%d)", w, h, new_w, new_h, max_size)

    return np.array(img)


# === COLOR SPACE ===

def apply_noise_reduction(gray_image, kernel_size=3):
    """Apply median filter for noise reduction"""
    return median_filter(gray_image, size=kernel_size)


def rgb_to_hsv(rgb_array):
    """Convert RGB to HSV color space"""
    rgb = rgb_array.astype(float) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    diff = max_c - min_c

    v = max_c
    s = np.where(max_c != 0, diff / max_c, 0)

    h = np.zeros_like(max_c)
    mask = diff != 0

    idx = (max_c == r) & mask
    h[idx] = (60 * ((g[idx] - b[idx]) / diff[idx]) + 360) % 360
    idx = (max_c == g) & mask
    h[idx] = (60 * ((b[idx] - r[idx]) / diff[idx]) + 120) % 360
    idx = (max_c == b) & mask
    h[idx] = (60 * ((r[idx] - g[idx]) / diff[idx]) + 240) % 360

    return np.stack([h, s * 255, v * 255], axis=-1)


def get_luminance(rgb_array):
    """Calculate luminance from RGB"""
    return 0.299 * rgb_array[:, :, 0] + 0.587 * rgb_array[:, :, 1] + 0.114 * rgb_array[:, :, 2]


# === PLATE DETECTION ===

def detect_plate_region(rgb_array):
    """Detect blood agar plate using HSV color space"""
    hsv = rgb_to_hsv(rgb_array)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    is_red_expanded = ((h <= 45) | (h >= 315)) & (s > 40) & (v > 30) & (v < 250)

    agar_mask = binary_closing(is_red_expanded, iterations=8)
    agar_mask = binary_opening(agar_mask, iterations=4)
    agar_mask = binary_fill_holes(agar_mask)

    return agar_mask


def create_plate_mask(shape, rgb_array=None, radius_factor=0.88):
    """Create plate mask combining circular and color detection"""
    h, w = shape[:2]
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 2 - 5

    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    circular_mask = dist <= radius * radius_factor

    if rgb_array is not None:
        color_mask = detect_plate_region(rgb_array)
        combined = circular_mask & color_mask
        if np.sum(combined) > np.sum(circular_mask) * 0.25:
            return combined, (center_x, center_y), radius

    return circular_mask, (center_x, center_y), radius


# === STEP 1: LOCAL BACKGROUND SUBTRACTION ===

def estimate_local_background(gray, plate_mask, sigma_fraction=0.15):
    """
    Estimate local background using large-kernel Gaussian blur.
    Sigma is set as a fraction of the image diagonal so it adapts
    to different image sizes. Smooths out colony-scale detail while
    preserving the illumination gradient.
    """
    h, w = gray.shape
    diagonal = np.sqrt(h ** 2 + w ** 2)
    sigma = max(20.0, sigma_fraction * diagonal)

    plate_median = np.median(gray[plate_mask])
    filled = np.where(plate_mask, gray, plate_median)

    background = gaussian_filter(filled, sigma=sigma)
    return background


def subtract_background(gray, plate_mask):
    """Subtract local background to produce foreground signal (>= 0)."""
    background = estimate_local_background(gray, plate_mask)
    foreground = gray.astype(float) - background.astype(float)
    foreground = np.maximum(foreground, 0.0)
    foreground = foreground * plate_mask
    return foreground


def compute_foreground_signal(rgb_array, plate_mask):
    """
    Compute foreground signal: median filter for noise reduction,
    then local background subtraction on luminance.
    """
    luminance = get_luminance(rgb_array).astype(float)
    denoised = median_filter(luminance, size=3)
    foreground = subtract_background(denoised, plate_mask)

    return foreground


# === STEP 2: THRESHOLD LADDER WITH SCORE MAP ===

def _fast_circularity(component_local, area):
    """
    O(1) circularity estimate using bounding-box aspect ratio and fill ratio.
    Used inside the threshold ladder inner loop for speed.
    """
    h, w = component_local.shape
    if h == 0 or w == 0:
        return 0.0

    aspect = min(h, w) / max(h, w)
    fill = area / (h * w)
    circularity = aspect * (fill / 0.785)
    return min(circularity, 1.0)


def build_score_map(foreground, plate_mask, n_thresholds=30,
                    min_area=6, max_area=6000, min_circularity=0.15):
    """
    OpenCFU-style threshold ladder. Sweeps threshold values from high to low,
    finds valid (circular, correctly-sized) connected components at each level,
    and accumulates a score map counting how many levels each pixel was valid.
    Colony centers score high (valid at many levels), noise scores low.
    """
    fg_values = foreground[plate_mask & (foreground > 0)]
    if len(fg_values) == 0:
        return np.zeros_like(foreground, dtype=np.int32)

    t_high = np.percentile(fg_values, 97)
    t_low = np.percentile(fg_values, 2)

    if t_high <= t_low:
        t_high = fg_values.max()
        t_low = max(1.0, fg_values.min())

    thresholds = np.linspace(t_high, t_low, n_thresholds)

    score_map = np.zeros_like(foreground, dtype=np.int32)
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity

    for T in thresholds:
        binary = (foreground >= T) & plate_mask
        labeled, n_components = label(binary, structure=structure)

        if n_components == 0:
            continue

        slices = find_objects(labeled)

        for i, slc in enumerate(slices):
            if slc is None:
                continue

            component_local = (labeled[slc] == (i + 1))
            area = int(np.sum(component_local))

            if area < min_area or area > max_area:
                continue

            circ = _fast_circularity(component_local, area)

            if circ >= min_circularity:
                score_map[slc] += component_local.astype(np.int32)

    return score_map


# === STEP 3: COLONY EXTRACTION FROM SCORE MAP ===

def _classify_density(score_map, plate_mask):
    """
    Pre-scan to classify plate density into LOW / MEDIUM / HIGH.
    Returns (channel_name, high_score_fraction).
    """
    max_score = score_map.max()
    if max_score == 0:
        return 'LOW', 0.0

    high_score_fraction = float(np.sum(
        (score_map > 0.5 * max_score) & plate_mask
    )) / max(1, int(np.sum(plate_mask)))

    if high_score_fraction > 0.15:
        return 'HIGH', high_score_fraction
    elif high_score_fraction >= 0.02:
        return 'MEDIUM', high_score_fraction
    else:
        return 'LOW', high_score_fraction


def extract_colonies_from_score_map(score_map, plate_mask,
                                    score_threshold_fraction=0.3):
    """Threshold the score map to get a binary colony mask."""
    max_score = score_map.max()
    if max_score == 0:
        return np.zeros_like(plate_mask, dtype=bool)

    threshold = max(1, int(score_threshold_fraction * max_score))
    binary_mask = (score_map >= threshold) & plate_mask

    binary_mask = binary_opening(binary_mask, iterations=1)
    binary_mask = binary_fill_holes(binary_mask)

    return binary_mask


# === STEP 4: PROPER WATERSHED SPLITTING ===

def _voronoi_split(binary_mask, markers, n_markers, min_area):
    """Voronoi partition by markers, then label connected components per territory."""
    marker_mask = (markers > 0)
    _, nearest_idx = distance_transform_edt(~marker_mask, return_indices=True)
    nearest_label = markers[nearest_idx[0], nearest_idx[1]]
    segmented = np.where(binary_mask, nearest_label, 0).astype(np.int32)

    slices = find_objects(segmented)
    if not slices:
        labeled, n_features = label(binary_mask)
        return labeled.astype(np.int32), n_features

    output = np.zeros_like(binary_mask, dtype=np.int32)
    next_id = 0

    for i, slc in enumerate(slices):
        if slc is None:
            continue
        region = (segmented[slc] == (i + 1))
        if not np.any(region):
            continue
        sub_labeled, n_sub = label(region)
        for j in range(1, n_sub + 1):
            component = (sub_labeled == j)
            if int(np.sum(component)) >= min_area:
                next_id += 1
                output[slc][component] = next_id

    if next_id == 0:
        labeled, n_features = label(binary_mask)
        return labeled.astype(np.int32), n_features

    return output, next_id


def split_touching_colonies(binary_mask, foreground, min_area=6,
                            oversized_threshold=250):
    """
    Two-pass colony splitting:
    Pass 1: EDT peaks with Voronoi partition (standard splitting)
    Pass 2: For oversized segments (>oversized_threshold px), apply
            conservative foreground intensity peaks (footprint 9, 70th
            percentile) to split only the most obvious merged clusters.
    """
    if np.sum(binary_mask) == 0:
        return np.zeros_like(binary_mask, dtype=np.int32), 0

    distance = distance_transform_edt(binary_mask)

    # Pass 1: EDT peaks
    local_max = maximum_filter(distance, size=7)
    peaks = (distance == local_max) & (distance >= 2.0) & binary_mask
    markers, n_markers = label(peaks)

    if n_markers <= 1:
        labeled, n_features = label(binary_mask)
    else:
        labeled, n_features = _voronoi_split(binary_mask, markers, n_markers, min_area)

    # Pass 2: split oversized segments using conservative foreground peaks
    fg_smooth = gaussian_filter(foreground * binary_mask, sigma=1.5)
    fg_vals = foreground[binary_mask]
    fg_threshold = np.percentile(fg_vals, 70) if len(fg_vals) > 0 else 0

    slices = find_objects(labeled)
    if not slices:
        return labeled.astype(np.int32), n_features

    output = np.zeros_like(labeled)
    next_id = 0

    for i, slc in enumerate(slices):
        if slc is None:
            continue
        seg_mask = (labeled[slc] == (i + 1))
        area = int(np.sum(seg_mask))
        if area < min_area:
            continue

        if area > oversized_threshold:
            # Try foreground peak splitting on this oversized segment
            local_fg = fg_smooth[slc] * seg_mask
            fg_local_max = maximum_filter(local_fg, size=9)
            fg_peaks = (local_fg == fg_local_max) & (local_fg > fg_threshold) & seg_mask
            sub_markers, n_sub_markers = label(fg_peaks)

            if n_sub_markers >= 2:
                sub_split, n_sub = _voronoi_split(seg_mask, sub_markers, n_sub_markers, min_area)

                # Circularity check: if any fragment has circularity < 0.3,
                # the split was bad — merge everything back as one colony
                bad_split = False
                for k in range(1, n_sub + 1):
                    fragment = (sub_split == k)
                    frag_area = int(np.sum(fragment))
                    if frag_area >= min_area:
                        frag_circ = calculate_circularity_robust(fragment)
                        if frag_circ < 0.3:
                            bad_split = True
                            break

                if not bad_split:
                    for k in range(1, n_sub + 1):
                        next_id += 1
                        output[slc][sub_split == k] = next_id
                    continue
                # else: fall through and keep segment as-is

        # Keep segment as-is
        next_id += 1
        output[slc][seg_mask] = next_id

    if next_id == 0:
        return labeled.astype(np.int32), n_features

    return output, next_id


# === STEP 5: ROBUST CIRCULARITY AND FILTERING ===

def calculate_circularity_robust(component_mask):
    """
    Robust circularity that works well at all scales.
    - Small components (area < 25px): moments-based (inertia tensor eigenvalue ratio)
    - Larger components: erosion-based perimeter with Cauchy-Crofton correction
    """
    area = int(np.sum(component_mask))
    if area == 0:
        return 0.0

    if area < 25:
        ys, xs = np.where(component_mask)
        if len(ys) < 2:
            return 0.8

        cy, cx = np.mean(ys), np.mean(xs)

        mu20 = np.mean((ys - cy) ** 2)
        mu02 = np.mean((xs - cx) ** 2)
        mu11 = np.mean((ys - cy) * (xs - cx))

        trace = mu20 + mu02
        det = mu20 * mu02 - mu11 ** 2
        discriminant = max(0, trace ** 2 - 4 * det)

        lambda1 = (trace + np.sqrt(discriminant)) / 2
        lambda2 = (trace - np.sqrt(discriminant)) / 2

        if lambda1 <= 0:
            return 0.8

        elongation = np.sqrt(max(0, lambda2) / lambda1)

        h_bb = ys.max() - ys.min() + 1
        w_bb = xs.max() - xs.min() + 1
        fill = area / (h_bb * w_bb) if (h_bb * w_bb) > 0 else 0

        circularity = elongation * min(1.0, fill / 0.785)
        return min(circularity, 1.0)
    else:
        interior = binary_erosion(component_mask)
        border = component_mask & ~interior
        perimeter = np.sum(border)

        if perimeter == 0:
            return 0.0

        perimeter_corrected = perimeter * (np.pi / 4)
        circularity = 4 * np.pi * area / (perimeter_corrected ** 2)
        return min(circularity, 1.0)


def filter_colonies(labeled, min_area, max_area, min_circularity):
    """Filter labeled regions by size and circularity, return re-labeled array."""
    if labeled.max() == 0:
        return labeled, 0, [], []

    slices = find_objects(labeled)
    filtered = np.zeros_like(labeled)
    count = 0
    sizes = []
    circs = []

    for i, slc in enumerate(slices):
        if slc is None:
            continue

        component = (labeled[slc] == (i + 1))
        area = int(np.sum(component))

        if area < min_area or area > max_area:
            continue

        circ = calculate_circularity_robust(component)

        if circ >= min_circularity:
            count += 1
            filtered[slc][component] = count
            sizes.append(area)
            circs.append(circ)

    return filtered, count, sizes, circs


# === LOW-DENSITY NOISE FILTER ===

def _filter_edge_and_center_noise(labeled, plate_mask, center, radius, colony_sizes_px,
                                   luminance=None):
    """
    Post-detection noise filter for low-density plates.
    Rejects detections that are:
    - Within 20px of plate edge
    - Within center 15% of plate (where label text sits)
    - Below 40% of median colony size
    - Low contrast: luminance difference between colony and 5px surround < 15 units
    Returns (labeled, count, sizes, circs, rejection_counts).
    """
    rejected_edge = 0
    rejected_center = 0
    rejected_undersized = 0
    rejected_contrast = 0

    if labeled.max() == 0 or not colony_sizes_px:
        return labeled, 0, [], [], {'edge': 0, 'center': 0, 'undersized': 0, 'contrast': 0}

    cx, cy = center
    median_size = float(np.median(colony_sizes_px))
    min_size_threshold = median_size * 0.40

    logger.info("[COLONY-DEBUG] Noise filter params: median_size=%.1f, "
                "min_size_threshold=%.1f (40%% median), radius=%d, center=(%d,%d)",
                median_size, min_size_threshold, radius, cx, cy)

    slices = find_objects(labeled)
    output = np.zeros_like(labeled)
    count = 0
    kept_sizes = []
    kept_circs = []

    for i, slc in enumerate(slices):
        if slc is None:
            continue
        component = (labeled[slc] == (i + 1))
        area = int(np.sum(component))
        if area == 0:
            continue

        # Compute centroid of this colony
        ys, xs = np.where(component)
        # Offset to global coordinates
        global_y = ys + slc[0].start
        global_x = xs + slc[1].start
        centroid_y = np.mean(global_y)
        centroid_x = np.mean(global_x)

        # Distance from plate center
        dist_from_center = np.sqrt((centroid_x - cx) ** 2 + (centroid_y - cy) ** 2)

        # Reject if within 20px of plate edge
        if radius > 0 and dist_from_center > (radius - 20):
            rejected_edge += 1
            continue

        # Reject if within center 15% (label text zone)
        if radius > 0 and dist_from_center < (radius * 0.15):
            rejected_center += 1
            continue

        # Reject if below 40% of median colony size
        if area < min_size_threshold:
            rejected_undersized += 1
            continue

        # Contrast filter: colony vs 5px surrounding ring must differ by >= 15 lum units
        if luminance is not None:
            # Expand slice bounds by 5px for the surrounding ring
            pad = 5
            h_img, w_img = luminance.shape
            y_start = max(0, slc[0].start - pad)
            y_stop = min(h_img, slc[0].stop + pad)
            x_start = max(0, slc[1].start - pad)
            x_stop = min(w_img, slc[1].stop + pad)
            expanded_slc = (slice(y_start, y_stop), slice(x_start, x_stop))

            # Place component in expanded frame
            comp_in_expanded = np.zeros((y_stop - y_start, x_stop - x_start), dtype=bool)
            cy_off = slc[0].start - y_start
            cx_off = slc[1].start - x_start
            comp_h, comp_w = component.shape
            comp_in_expanded[cy_off:cy_off + comp_h, cx_off:cx_off + comp_w] = component

            # 5px dilation ring minus the colony itself
            dilated = binary_dilation(comp_in_expanded, iterations=pad)
            ring = dilated & ~comp_in_expanded

            lum_patch = luminance[expanded_slc]
            colony_lum = float(np.mean(lum_patch[comp_in_expanded]))
            ring_pixels = lum_patch[ring]
            if len(ring_pixels) > 0:
                surround_lum = float(np.mean(ring_pixels))
                contrast = abs(colony_lum - surround_lum)
                logger.info("[COLONY-DEBUG] Colony #%d: area=%d, contrast=%.1f (colony_lum=%.1f, surround=%.1f)",
                            i + 1, area, contrast, colony_lum, surround_lum)
                if contrast < 15.0:
                    rejected_contrast += 1
                    continue

        circ = calculate_circularity_robust(component)
        count += 1
        output[slc][component] = count
        kept_sizes.append(area)
        kept_circs.append(circ)

    rejections = {'edge': rejected_edge, 'center': rejected_center,
                  'undersized': rejected_undersized, 'contrast': rejected_contrast}
    return output, count, kept_sizes, kept_circs, rejections


# === MAIN DETECTION PIPELINE ===

def colony_detection_opencfu(rgb_array, luminance, plate_mask,
                             sensitivity=0.5, colony_size='auto',
                             plate_center=None, plate_radius=0):
    """
    OpenCFU-inspired colony detection with density-channel routing.

    1. Compute foreground signal (background subtraction)
    2. Build score map (threshold ladder)
    3. Density pre-scan → classify as LOW / MEDIUM / HIGH
    4. Route to density-specific channel for extraction, splitting, filtering
    """
    params = _adjust_params_for_sensitivity(sensitivity, colony_size)

    # Step 1: Background subtraction
    foreground = compute_foreground_signal(rgb_array, plate_mask)

    fg_max = float(foreground.max())
    fg_positive = foreground[plate_mask & (foreground > 0)]
    fg_mean = float(np.mean(fg_positive)) if len(fg_positive) > 0 else 0.0

    # Step 2: Build score map
    score_map = build_score_map(
        foreground, plate_mask,
        n_thresholds=params['n_thresholds'],
        min_area=params['ladder_min_area'],
        max_area=params['ladder_max_area'],
        min_circularity=params['ladder_min_circularity']
    )

    # Step 3: Density pre-scan
    density_channel, high_score_frac = _classify_density(score_map, plate_mask)
    logger.info("[COLONY-DEBUG] high_score_fraction=%.6f", high_score_frac)
    logger.info("[COLONY-DEBUG] channel_selected=%s", density_channel)

    # Step 4: Channel-specific parameters
    if density_channel == 'LOW':
        # Hardcoded v1.8.0 value — sensitivity must NOT lower this for sparse plates
        score_frac = 0.19
        oversized_threshold = 250  # original v1.8.0 value
        min_circularity = 0.40  # stricter — noise specks are irregular
    elif density_channel == 'MEDIUM':
        # v1.9.0 settings — threshold reduction + aggressive splitting
        score_frac = params['score_threshold_fraction'] * 0.85
        oversized_threshold = params['oversized_segment_threshold']  # 80px
        min_circularity = params['min_circularity']
    else:  # HIGH
        score_frac = min(0.35, params['score_threshold_fraction'] + 0.05)
        oversized_threshold = params['oversized_segment_threshold']
        min_circularity = params['min_circularity']

    logger.info("[COLONY-DEBUG] score_threshold_fraction=%.4f", score_frac)
    logger.info("[COLONY-DEBUG] watershed_split_trigger=%d", oversized_threshold)
    logger.info("[COLONY-DEBUG] min_circularity=%.3f", min_circularity)

    # Step 5: Extract colony mask
    binary_mask = extract_colonies_from_score_map(
        score_map, plate_mask,
        score_threshold_fraction=score_frac
    )

    # Step 6: Split touching colonies
    labeled, n_segments = split_touching_colonies(
        binary_mask, foreground,
        min_area=params['min_colony_area'],
        oversized_threshold=oversized_threshold
    )

    # Step 7: Filter by size and circularity
    labeled, colony_count, colony_sizes, colony_circs = filter_colonies(
        labeled,
        min_area=params['min_colony_area'],
        max_area=params['max_colony_area'],
        min_circularity=min_circularity
    )

    logger.info("[COLONY-DEBUG] raw_count_before_noise_filter=%d", colony_count)

    # Step 8: Low-density post-filter (edge, center, runt, contrast noise)
    noise_rejected = 0
    rejection_details = {'edge': 0, 'center': 0, 'undersized': 0, 'contrast': 0}
    if density_channel == 'LOW' and colony_count > 0 and plate_center is not None:
        pre_filter_count = colony_count
        labeled, colony_count, colony_sizes, colony_circs, rejection_details = (
            _filter_edge_and_center_noise(
                labeled, plate_mask, plate_center, plate_radius, colony_sizes,
                luminance=luminance
            )
        )
        noise_rejected = pre_filter_count - colony_count

    logger.info("[COLONY-DEBUG] noise_rejected_edge=%d", rejection_details['edge'])
    logger.info("[COLONY-DEBUG] noise_rejected_center=%d", rejection_details['center'])
    logger.info("[COLONY-DEBUG] noise_rejected_undersized=%d", rejection_details['undersized'])
    logger.info("[COLONY-DEBUG] noise_rejected_contrast=%d", rejection_details['contrast'])
    logger.info("[COLONY-DEBUG] final_colony_count=%d", colony_count)

    avg_size = float(np.mean(colony_sizes)) if colony_sizes else 0.0
    avg_circ = float(np.mean(colony_circs)) if colony_circs else 0.0

    debug_info = {
        'algorithm': 'opencfu_ladder_v1.9.2',
        'density_channel': density_channel,
        'high_score_fraction': round(high_score_frac, 4),
        'sensitivity': round(sensitivity, 2),
        'colony_size': colony_size,
        'n_thresholds': params['n_thresholds'],
        'score_threshold_used': round(score_frac, 3),
        'oversized_threshold_used': oversized_threshold,
        'min_circularity_used': round(min_circularity, 3),
        'score_map_max': int(score_map.max()),
        'foreground_max': round(fg_max, 1),
        'foreground_mean': round(fg_mean, 1),
        'binary_mask_pixels': int(np.sum(binary_mask)),
        'segments_before_filter': n_segments,
        'noise_rejected': noise_rejected,
        'colonies_after_filter': colony_count,
    }

    return colony_count, avg_size, avg_circ, labeled, debug_info


# === HEMOLYSIS DETECTION (unchanged) ===

def detect_hemolysis_hsv(rgb_array, plate_mask):
    """Hemolysis detection using HSV color space"""
    hsv = rgb_to_hsv(rgb_array)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    plate_h = h[plate_mask]
    plate_s = s[plate_mask]
    plate_v = v[plate_mask]

    bg_h = np.median(plate_h)
    bg_s = np.median(plate_s)
    bg_v = np.median(plate_v)

    is_bright = v > (bg_v + 8)
    is_desaturated = s < (bg_s * 0.80)
    beta_zones = (is_bright | is_desaturated) & plate_mask

    very_bright = v > (bg_v + 25)
    colony_cores = very_bright & plate_mask

    if np.any(colony_cores):
        halo_region = binary_dilation(colony_cores, iterations=8) & ~colony_cores
        beta_in_halo = beta_zones & halo_region
        beta_ratio = np.sum(beta_in_halo) / np.sum(plate_mask) if np.sum(plate_mask) > 0 else 0
    else:
        beta_ratio = np.sum(beta_zones) / np.sum(plate_mask) if np.sum(plate_mask) > 0 else 0

    overall_desat = np.mean(plate_s) < 120

    is_greenish = (h > 40) & (h < 160)
    is_darker = v < (bg_v - 3)
    alpha_zones = is_greenish & is_darker & plate_mask
    alpha_ratio = np.sum(alpha_zones) / np.sum(plate_mask) if np.sum(plate_mask) > 0 else 0

    if beta_ratio > 0.02 or overall_desat:
        hemo_type = 'beta'
        confidence = min(0.95, 0.60 + beta_ratio * 4)
    elif alpha_ratio > 0.04:
        hemo_type = 'alpha'
        confidence = min(0.90, 0.50 + alpha_ratio * 3)
    else:
        hemo_type = 'gamma'
        confidence = 0.65

    return {
        'type': hemo_type,
        'confidence': round(confidence, 4),
        'details': {
            'beta_zone_ratio': round(beta_ratio, 4),
            'alpha_zone_ratio': round(alpha_ratio, 4),
            'background_hue': round(bg_h, 1),
            'background_saturation': round(bg_s, 1)
        }
    }


# === CLINICAL DECISION (unchanged) ===

def generate_decision(colony_count, hemolysis, media_type):
    """Generate clinical decision label"""
    if colony_count <= 3:
        growth = "No growth"
        confidence = "HIGH"
        review = False
    elif colony_count <= 20:
        growth = f"Low count ({colony_count} CFU)"
        confidence = "MEDIUM"
        review = True
    elif colony_count <= 130:
        growth = f"Moderate growth ({colony_count} CFU)"
        confidence = "HIGH"
        review = False
    else:
        growth = f"Significant growth ({colony_count} CFU)"
        confidence = "HIGH"
        review = False

    label_parts = [growth]

    if media_type == 'blood_agar':
        if hemolysis['type'] == 'beta':
            label_parts.append("beta-hemolytic \u2013 follow Strep protocol")
        elif hemolysis['type'] == 'alpha':
            label_parts.append("alpha-hemolytic \u2013 rule out S. pneumoniae")
        else:
            label_parts.append("non-hemolytic")

    return {
        'label': " \u2013 ".join(label_parts),
        'confidence': confidence,
        'requires_review': review
    }


# === MAIN ANALYSIS ===

def _safe_plate_ratio(plate_mask, total_pixels):
    """Safe division for plate coverage ratio."""
    if total_pixels == 0:
        return 0.0
    return float(np.sum(plate_mask)) / total_pixels


def _run_analysis(img, media_type, detection_mode, colony_size):
    """Core analysis logic, separated for memory-fallback retry."""
    sensitivity = MODE_SENSITIVITY.get(detection_mode, 0.5)
    params = _adjust_params_for_sensitivity(sensitivity, colony_size)
    plate_mask, center, radius = create_plate_mask(
        img.shape, rgb_array=img, radius_factor=params['plate_radius_factor']
    )

    total_pixels = img.shape[0] * img.shape[1]
    plate_coverage = _safe_plate_ratio(plate_mask, total_pixels)

    # Safety net: if plate coverage is too low, expand radius by 10%
    if plate_coverage < 0.50:
        expanded_factor = min(params['plate_radius_factor'] * 1.10, 0.98)
        plate_mask, center, radius = create_plate_mask(
            img.shape, rgb_array=img, radius_factor=expanded_factor
        )
        plate_coverage = _safe_plate_ratio(plate_mask, total_pixels)

    luminance = get_luminance(img)

    colony_count, avg_size, avg_circ, labeled, debug_info = colony_detection_opencfu(
        img, luminance, plate_mask, sensitivity=sensitivity, colony_size=colony_size,
        plate_center=center, plate_radius=radius
    )

    hemolysis = detect_hemolysis_hsv(img, plate_mask)
    decision = generate_decision(colony_count, hemolysis, media_type)

    logger.info("Analysis complete: %d colonies detected (mode=%s, size=%s, image=%dx%d)",
                colony_count, detection_mode, colony_size, img.shape[1], img.shape[0])

    return {
        'status': 'success',
        'colony_count': colony_count,
        'detection_mode': detection_mode,
        'colony_size': colony_size,
        'decision_label': decision['label'],
        'decision_confidence': decision['confidence'],
        'requires_manual_review': decision['requires_review'],
        'hemolysis': hemolysis,
        'colony_statistics': {
            'average_size_px': round(avg_size, 1),
            'average_circularity': round(avg_circ, 3)
        },
        'plate_info': {
            'center': list(center),
            'radius_px': radius,
            'analyzed_size': list(img.shape[:2]),
            'plate_coverage_pct': round(100 * plate_coverage, 1)
        },
        'debug': debug_info,
        'version': '1.9.2'
    }


def analyze_plate(image_bytes, media_type='blood_agar', detection_mode='sensitive',
                  colony_size='auto'):
    """Main analysis function - v1.9.1 OpenCFU pipeline with density channels."""
    # Try at full resolution (1600px) first
    try:
        img = load_and_resize_image(image_bytes, max_size=MAX_IMAGE_SIZE)
        return _run_analysis(img, media_type, detection_mode, colony_size)
    except MemoryError:
        logger.warning("MemoryError at %dpx, retrying at %dpx fallback",
                       MAX_IMAGE_SIZE, FALLBACK_IMAGE_SIZE)
        img = load_and_resize_image(image_bytes, max_size=FALLBACK_IMAGE_SIZE)
        result = _run_analysis(img, media_type, detection_mode, colony_size)
        result['debug']['fallback_resize'] = FALLBACK_IMAGE_SIZE
        return result


# === FLASK ROUTES ===

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.9.2',
        'max_image_size': MAX_IMAGE_SIZE,
        'supported_media_types': SUPPORTED_MEDIA_TYPES,
        'detection_modes': {
            'sensitive': 'Higher sensitivity (catches weak colonies)',
            'strict': 'Higher specificity (filters aggressively)',
            'auto': 'Balanced auto-adaptive detection'
        },
        'colony_sizes': {
            'small': 'Tiny colonies (~45px avg) — lower thresholds, wider plate mask',
            'medium': 'Standard colonies — default parameters',
            'large': 'Large colonies — stricter filtering, higher split threshold',
            'auto': 'Auto-adaptive (default)'
        },
        'algorithms': [
            'Local background subtraction (large Gaussian)',
            'Threshold ladder with score-map accumulation (OpenCFU)',
            'Auto-adaptive score threshold',
            'Voronoi colony splitting (distance-transform markers)',
            'Robust circularity filtering (moments + Cauchy-Crofton)',
            'HSV hemolysis classification'
        ]
    })


@app.route('/warmup', methods=['GET'])
def warmup():
    """Warm-up endpoint"""
    return jsonify({
        'status': 'warm',
        'version': '1.9.2',
        'message': 'Service is ready for analysis'
    })


def _persist_upload(image_bytes, result, user_email, image_metadata, content_type=None):
    """Best-effort persistence to Supabase. Never raises."""
    if supabase is None:
        return
    try:
        store_upload(
            image_bytes=image_bytes,
            count=result.get('colony_count'),
            density_channel=result.get('debug', {}).get('density_channel'),
            user_email=user_email,
            image_metadata=image_metadata,
            content_type=content_type,
        )
    except Exception as e:
        logger.error("Supabase storage failed (analysis still returned): %s", e)


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze plate image"""
    try:
        media_type = request.form.get('media_type', 'blood_agar')
        if media_type not in SUPPORTED_MEDIA_TYPES:
            media_type = 'blood_agar'

        detection_mode = request.form.get('detection_mode', 'sensitive')
        if detection_mode not in DETECTION_MODES:
            detection_mode = 'sensitive'

        colony_size = request.form.get('colony_size', 'auto')
        if colony_size not in COLONY_SIZES:
            colony_size = 'auto'

        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                image_bytes = file.read()
                result = analyze_plate(image_bytes, media_type, detection_mode, colony_size)
                _persist_upload(
                    image_bytes,
                    result,
                    user_email=request.form.get('user_email'),
                    image_metadata={
                        'original_filename': file.filename,
                        'original_content_type': file.mimetype,
                        'media_type': media_type,
                        'detection_mode': detection_mode,
                        'colony_size': colony_size,
                        'source': 'multipart',
                    },
                    content_type=file.mimetype,
                )
                return jsonify(result)

        if request.is_json:
            data = request.get_json()
            if data and 'image_base64' in data:
                image_bytes, detected_ct = _parse_image_base64(data['image_base64'])
                # Allow an explicit content_type field in the body to override
                content_type = data.get('content_type') or detected_ct
                dm = data.get('detection_mode', 'sensitive')
                if dm not in DETECTION_MODES:
                    dm = 'sensitive'
                cs = data.get('colony_size', 'auto')
                if cs not in COLONY_SIZES:
                    cs = 'auto'
                result = analyze_plate(image_bytes, media_type, dm, cs)
                _persist_upload(
                    image_bytes,
                    result,
                    user_email=data.get('user_email'),
                    image_metadata={
                        'original_content_type': content_type,
                        'media_type': media_type,
                        'detection_mode': dm,
                        'colony_size': cs,
                        'source': 'json_base64',
                    },
                    content_type=content_type,
                )
                return jsonify(result)

        return jsonify({
            'status': 'error',
            'error': 'No image provided. Send as "image" file or "image_base64" in JSON.'
        }), 400

    except Exception as e:
        logger.error("Analysis failed: %s\n%s", str(e), traceback.format_exc())
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
