"""
Colony Counting and Hemolysis Detection API
Version: 1.8.0 - OpenCFU-inspired pipeline

Algorithm based on OpenCFU (Geissmann 2013, PLOS ONE):
1. Local background subtraction (large Gaussian) — handles uneven illumination
2. Threshold ladder with score-map accumulation — robust multi-level detection
3. Colony extraction from score map with auto-adaptive threshold
4. Proper watershed splitting (scipy.ndimage.watershed_ift on distance transform)
5. Robust circularity filtering (moments-based for small colonies)

Detection mode ('sensitive', 'strict', 'auto') maps to a continuous
sensitivity parameter [0, 1] that makes small adjustments. The core
algorithm is the same regardless of mode.
"""

import os
import traceback
import base64
from io import BytesIO
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image

app = Flask(__name__)

# Configuration
MAX_IMAGE_SIZE = 800
SUPPORTED_MEDIA_TYPES = ['blood_agar', 'nutrient_agar', 'macconkey_agar']
DETECTION_MODES = ['sensitive', 'strict', 'auto']

# Unified pipeline parameters
PIPELINE_PARAMS = {
    'bg_sigma_fraction': 0.12,
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


def _adjust_params_for_sensitivity(sensitivity):
    """Adjust pipeline parameters based on sensitivity value in [0, 1]."""
    p = dict(PIPELINE_PARAMS)
    p['score_threshold_fraction'] = 0.29 - 0.20 * sensitivity
    p['min_colony_area'] = int(14 - 8 * sensitivity)
    p['min_circularity'] = 0.35 - 0.13 * sensitivity
    return p


# === IMAGE LOADING ===

def load_and_resize_image(image_bytes):
    """Load image and resize to manageable size"""
    img = Image.open(BytesIO(image_bytes))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    w, h = img.size
    if max(w, h) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return np.array(img)


# === COLOR SPACE ===

def apply_noise_reduction(gray_image, kernel_size=3):
    """Apply median filter for noise reduction"""
    from scipy.ndimage import median_filter
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
    from scipy.ndimage import binary_closing, binary_opening, binary_fill_holes

    hsv = rgb_to_hsv(rgb_array)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    is_red_expanded = ((h <= 45) | (h >= 315)) & (s > 40) & (v > 30) & (v < 250)

    agar_mask = binary_closing(is_red_expanded, iterations=8)
    agar_mask = binary_opening(agar_mask, iterations=4)
    agar_mask = binary_fill_holes(agar_mask)

    return agar_mask


def create_plate_mask(shape, rgb_array=None):
    """Create plate mask combining circular and color detection"""
    h, w = shape[:2]
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 2 - 5

    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    circular_mask = dist <= radius * 0.88

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
    from scipy.ndimage import gaussian_filter

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
    from scipy.ndimage import median_filter

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
    from scipy.ndimage import label, find_objects

    fg_values = foreground[plate_mask & (foreground > 0)]
    if len(fg_values) == 0:
        return np.zeros_like(foreground, dtype=np.int32)

    t_high = np.percentile(fg_values, 95)
    t_low = np.percentile(fg_values, 5)

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

def _auto_adjust_score_threshold(score_map, plate_mask, base_fraction):
    """
    Auto-adjust score threshold based on plate density.
    Dense plates get higher threshold (to separate), sparse plates lower (to not miss).
    """
    max_score = score_map.max()
    if max_score == 0:
        return base_fraction

    high_score_fraction = np.sum(
        (score_map > 0.5 * max_score) & plate_mask
    ) / max(1, np.sum(plate_mask))

    if high_score_fraction > 0.15:
        return min(0.35, base_fraction + 0.05)
    elif high_score_fraction < 0.02:
        return max(0.10, base_fraction - 0.05)

    return base_fraction


def extract_colonies_from_score_map(score_map, plate_mask,
                                    score_threshold_fraction=0.3):
    """Threshold the score map to get a binary colony mask."""
    from scipy.ndimage import binary_fill_holes, binary_opening

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
    from scipy.ndimage import distance_transform_edt, label, find_objects

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


def split_touching_colonies(binary_mask, foreground, min_area=6):
    """
    Two-pass colony splitting:
    Pass 1: EDT peaks with Voronoi partition (standard splitting)
    Pass 2: For oversized segments (>300px), apply conservative foreground
            intensity peaks (footprint 9, 70th percentile) to split only
            the most obvious merged clusters.
    """
    from scipy.ndimage import (distance_transform_edt, label,
                               maximum_filter, find_objects,
                               gaussian_filter)

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

        if area > 250:
            # Try foreground peak splitting on this oversized segment
            local_fg = fg_smooth[slc] * seg_mask
            fg_local_max = maximum_filter(local_fg, size=9)
            fg_peaks = (local_fg == fg_local_max) & (local_fg > fg_threshold) & seg_mask
            sub_markers, n_sub_markers = label(fg_peaks)

            if n_sub_markers >= 2:
                sub_split, n_sub = _voronoi_split(seg_mask, sub_markers, n_sub_markers, min_area)
                for k in range(1, n_sub + 1):
                    next_id += 1
                    output[slc][sub_split == k] = next_id
                continue

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
    from scipy.ndimage import binary_erosion

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
    from scipy.ndimage import find_objects

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


# === MAIN DETECTION PIPELINE ===

def colony_detection_opencfu(rgb_array, luminance, plate_mask, sensitivity=0.5):
    """
    OpenCFU-inspired colony detection pipeline.

    1. Compute foreground signal (background subtraction)
    2. Build score map (threshold ladder)
    3. Extract binary colony mask from score map
    4. Split touching colonies (watershed_ift)
    5. Filter by size and circularity
    """
    params = _adjust_params_for_sensitivity(sensitivity)

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

    # Step 3: Extract colony mask
    score_frac = params['score_threshold_fraction']
    score_frac = _auto_adjust_score_threshold(score_map, plate_mask, score_frac)

    binary_mask = extract_colonies_from_score_map(
        score_map, plate_mask,
        score_threshold_fraction=score_frac
    )

    # Step 4: Split touching colonies
    labeled, n_segments = split_touching_colonies(
        binary_mask, foreground,
        min_area=params['min_colony_area']
    )

    # Step 5: Filter by size and circularity
    labeled, colony_count, colony_sizes, colony_circs = filter_colonies(
        labeled,
        min_area=params['min_colony_area'],
        max_area=params['max_colony_area'],
        min_circularity=params['min_circularity']
    )

    avg_size = float(np.mean(colony_sizes)) if colony_sizes else 0.0
    avg_circ = float(np.mean(colony_circs)) if colony_circs else 0.0

    debug_info = {
        'algorithm': 'opencfu_ladder_v1.8.0',
        'sensitivity': round(sensitivity, 2),
        'n_thresholds': params['n_thresholds'],
        'score_threshold_used': round(score_frac, 3),
        'score_map_max': int(score_map.max()),
        'foreground_max': round(fg_max, 1),
        'foreground_mean': round(fg_mean, 1),
        'binary_mask_pixels': int(np.sum(binary_mask)),
        'segments_before_filter': n_segments,
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

    from scipy.ndimage import binary_dilation

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
    elif colony_count <= 50:
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

def analyze_plate(image_bytes, media_type='blood_agar', detection_mode='sensitive'):
    """Main analysis function - v1.8.0 OpenCFU pipeline"""
    img = load_and_resize_image(image_bytes)
    plate_mask, center, radius = create_plate_mask(img.shape, rgb_array=img)
    luminance = get_luminance(img)

    sensitivity = MODE_SENSITIVITY.get(detection_mode, 0.5)

    colony_count, avg_size, avg_circ, labeled, debug_info = colony_detection_opencfu(
        img, luminance, plate_mask, sensitivity=sensitivity
    )

    hemolysis = detect_hemolysis_hsv(img, plate_mask)
    decision = generate_decision(colony_count, hemolysis, media_type)

    return {
        'status': 'success',
        'colony_count': colony_count,
        'detection_mode': detection_mode,
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
            'plate_coverage_pct': round(100 * np.sum(plate_mask) / (img.shape[0] * img.shape[1]), 1)
        },
        'debug': debug_info,
        'version': '1.8.0-opencfu'
    }


# === FLASK ROUTES ===

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.8.0-opencfu',
        'max_image_size': MAX_IMAGE_SIZE,
        'supported_media_types': SUPPORTED_MEDIA_TYPES,
        'detection_modes': {
            'sensitive': 'Higher sensitivity (catches weak colonies)',
            'strict': 'Higher specificity (filters aggressively)',
            'auto': 'Balanced auto-adaptive detection'
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
        'version': '1.8.0-opencfu',
        'message': 'Service is ready for analysis'
    })


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

        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                image_bytes = file.read()
                result = analyze_plate(image_bytes, media_type, detection_mode)
                return jsonify(result)

        if request.is_json:
            data = request.get_json()
            if data and 'image_base64' in data:
                image_bytes = base64.b64decode(data['image_base64'])
                dm = data.get('detection_mode', 'sensitive')
                if dm not in DETECTION_MODES:
                    dm = 'sensitive'
                result = analyze_plate(image_bytes, media_type, dm)
                return jsonify(result)

        return jsonify({
            'status': 'error',
            'error': 'No image provided. Send as "image" file or "image_base64" in JSON.'
        }), 400

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
