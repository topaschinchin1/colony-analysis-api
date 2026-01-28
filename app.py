"""
Colony Counting and Hemolysis Detection API
Version: 1.7.0 - Dual-mode detection

TWO DETECTION MODES:
1. "sensitive" - Best for sparse/low-density plates (like GAS-03)
   - Lower thresholds, catches more colonies
   - Based on v1.5.3 parameters (114 CFU for GAS-03, target 108)

2. "strict" - Best for dense/high-density plates (like GAS-01)  
   - Higher thresholds, filters more aggressively
   - Based on v1.6.2 parameters but tuned

User selects mode via 'detection_mode' parameter in the form.
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

# Mode-specific parameters
MODE_PARAMS = {
    'sensitive': {
        # Based on v1.5.3 - good for GAS-03 (sparse plates)
        'min_colony_size': 8,
        'max_colony_size': 5000,
        'min_circularity': 0.25,
        'adaptive_c_values': [5, 8, 12],
        'voting_threshold': 0.35,
        'hsv_brightness_offset': 8,
        'hsv_saturation_ratio': 0.88,
        'intensity_std_multiplier': 0.5,
        'watershed_footprint': 9,
        'watershed_percentile': 30,
    },
    'strict': {
        # Tuned for GAS-01 (dense plates)
        'min_colony_size': 12,
        'max_colony_size': 5000,
        'min_circularity': 0.28,
        'adaptive_c_values': [12, 16, 20],
        'voting_threshold': 0.55,
        'hsv_brightness_offset': 15,
        'hsv_saturation_ratio': 0.78,
        'intensity_std_multiplier': 1.0,
        'watershed_footprint': 13,
        'watershed_percentile': 60,
    }
}


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


def apply_clahe(gray_image, clip_limit=2.0, tile_size=8):
    """Apply CLAHE for contrast enhancement"""
    if gray_image.dtype != np.uint8:
        img_normalized = ((gray_image - gray_image.min()) / 
                         (gray_image.max() - gray_image.min() + 1e-8) * 255).astype(np.uint8)
    else:
        img_normalized = gray_image
    
    h, w = img_normalized.shape
    tile_h = max(1, h // tile_size)
    tile_w = max(1, w // tile_size)
    
    result = np.zeros_like(img_normalized, dtype=float)
    
    for i in range(tile_size):
        for j in range(tile_size):
            y1 = i * tile_h
            y2 = min((i + 1) * tile_h, h)
            x1 = j * tile_w
            x2 = min((j + 1) * tile_w, w)
            
            if y2 <= y1 or x2 <= x1:
                continue
                
            tile = img_normalized[y1:y2, x1:x2]
            hist, _ = np.histogram(tile.flatten(), bins=256, range=(0, 256))
            clip_threshold = clip_limit * tile.size / 256
            excess = np.sum(np.maximum(hist - clip_threshold, 0))
            hist = np.minimum(hist, clip_threshold)
            hist = hist + excess / 256
            cdf = hist.cumsum()
            cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-8) * 255
            result[y1:y2, x1:x2] = cdf_normalized[tile.astype(int)]
    
    return result.astype(np.uint8)


def apply_noise_reduction(gray_image, kernel_size=3):
    """Apply median filter for noise reduction"""
    from scipy import ndimage
    return ndimage.median_filter(gray_image, size=kernel_size)


def adaptive_threshold(gray_image, block_size=51, C=10):
    """Adaptive thresholding"""
    from scipy import ndimage
    local_mean = ndimage.gaussian_filter(gray_image.astype(float), sigma=block_size/6)
    binary = gray_image > (local_mean + C)
    return binary


def rgb_to_hsv(rgb_array):
    """Convert RGB to HSV color space"""
    rgb = rgb_array.astype(float) / 255.0
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    
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
    return 0.299 * rgb_array[:,:,0] + 0.587 * rgb_array[:,:,1] + 0.114 * rgb_array[:,:,2]


def detect_plate_region(rgb_array):
    """Detect blood agar plate using HSV color space"""
    hsv = rgb_to_hsv(rgb_array)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    is_red_expanded = ((h <= 45) | (h >= 315)) & (s > 40) & (v > 30) & (v < 250)
    
    from scipy import ndimage
    agar_mask = ndimage.binary_closing(is_red_expanded, iterations=8)
    agar_mask = ndimage.binary_opening(agar_mask, iterations=4)
    agar_mask = ndimage.binary_fill_holes(agar_mask)
    
    return agar_mask


def create_plate_mask(shape, rgb_array=None):
    """Create plate mask combining circular and color detection"""
    h, w = shape[:2]
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 2 - 5
    
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    circular_mask = dist <= radius * 0.88
    
    if rgb_array is not None:
        color_mask = detect_plate_region(rgb_array)
        combined = circular_mask & color_mask
        if np.sum(combined) > np.sum(circular_mask) * 0.25:
            return combined, (center_x, center_y), radius
    
    return circular_mask, (center_x, center_y), radius


def calculate_circularity(component_mask):
    """Calculate circularity of a component"""
    from scipy import ndimage
    
    area = np.sum(component_mask)
    if area == 0:
        return 0
    
    dilated = ndimage.binary_dilation(component_mask)
    perimeter = np.sum(dilated & ~component_mask)
    
    if perimeter == 0:
        return 0
    
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return min(circularity, 1.0)


def watershed_segmentation(binary_mask, intensity_image, params):
    """Watershed segmentation with mode-specific parameters"""
    from scipy import ndimage
    from scipy.ndimage import distance_transform_edt, label, maximum_filter
    
    if np.sum(binary_mask) == 0:
        return np.zeros_like(binary_mask, dtype=int), 0
    
    distance = distance_transform_edt(binary_mask)
    footprint_size = params['watershed_footprint']
    local_max = maximum_filter(distance, size=footprint_size)
    
    distance_threshold = max(3, np.percentile(distance[binary_mask], params['watershed_percentile']))
    peaks = (distance == local_max) & (distance >= distance_threshold) & binary_mask
    
    markers, num_markers = label(peaks)
    
    if num_markers == 0:
        labeled, num_features = label(binary_mask)
        return labeled, num_features
    
    # Region growing from markers
    segmented = np.zeros_like(binary_mask, dtype=int)
    segmented[markers > 0] = markers[markers > 0]
    
    marker_coords = {}
    for i in range(1, num_markers + 1):
        coords = np.where(markers == i)
        if len(coords[0]) > 0:
            marker_coords[i] = (np.mean(coords[0]), np.mean(coords[1]))
    
    fg_coords = np.where(binary_mask & (segmented == 0))
    for y, x in zip(fg_coords[0], fg_coords[1]):
        min_dist = float('inf')
        nearest_marker = 0
        for marker_id, (my, mx) in marker_coords.items():
            d = (y - my)**2 + (x - mx)**2
            if d < min_dist:
                min_dist = d
                nearest_marker = marker_id
        if nearest_marker > 0 and min_dist < 2500:
            segmented[y, x] = nearest_marker
    
    remaining = binary_mask & (segmented == 0)
    if np.any(remaining):
        remaining_labeled, _ = label(remaining)
        max_label = segmented.max()
        segmented = np.where(remaining, remaining_labeled + max_label, segmented)
    
    return segmented, int(segmented.max())


def colony_detection_dual_mode(rgb_array, luminance, plate_mask, mode='sensitive'):
    """
    Dual-mode colony detection
    
    mode='sensitive': Better for sparse plates (GAS-03 style)
    mode='strict': Better for dense plates (GAS-01 style)
    """
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter, maximum_filter, label
    
    params = MODE_PARAMS[mode]
    
    # === PREPROCESSING ===
    denoised = apply_noise_reduction(luminance, kernel_size=3)
    enhanced = apply_clahe(denoised.astype(np.uint8), clip_limit=2.0, tile_size=8)
    enhanced = enhanced.astype(float)
    
    plate_pixels = enhanced[plate_mask]
    bg_mean = np.median(plate_pixels)
    bg_std = np.std(plate_pixels)
    
    # === METHOD 1: Adaptive thresholding ===
    adaptive_detections = np.zeros_like(luminance, dtype=float)
    c_values = params['adaptive_c_values']
    
    for block_size in [41, 61, 81]:
        for C_val in c_values:
            adaptive_mask = adaptive_threshold(enhanced, block_size=block_size, C=C_val)
            adaptive_mask = adaptive_mask & plate_mask
            adaptive_detections += adaptive_mask.astype(float) / 9
    
    method1_mask = (adaptive_detections >= 0.35) & plate_mask
    method1_weight = 1.2
    
    # === METHOD 2: HSV-based detection ===
    hsv = rgb_to_hsv(rgb_array)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    plate_v = v[plate_mask]
    plate_s = s[plate_mask]
    bg_v = np.median(plate_v)
    bg_s = np.median(plate_s)
    
    brightness_offset = params['hsv_brightness_offset']
    saturation_ratio = params['hsv_saturation_ratio']
    
    is_bright_hsv = v > (bg_v + brightness_offset)
    is_less_saturated = s < (bg_s * saturation_ratio)
    method2_mask = is_bright_hsv & is_less_saturated & plate_mask
    method2_weight = 1.0
    
    # === METHOD 3: Local maxima detection ===
    blurred = gaussian_filter(enhanced, sigma=1.2)
    local_max = maximum_filter(blurred, size=5)
    peaks = (blurred == local_max) & (enhanced > bg_mean + 0.4 * bg_std) & plate_mask
    method3_mask = ndimage.binary_dilation(peaks, iterations=2)
    method3_weight = 0.8
    
    # === METHOD 4: Direct intensity threshold ===
    intensity_multiplier = params['intensity_std_multiplier']
    intensity_thresh = bg_mean + intensity_multiplier * bg_std
    method4_mask = (enhanced > intensity_thresh) & plate_mask
    method4_weight = 0.7
    
    # === WEIGHTED VOTING ===
    weighted_score = (
        method1_mask.astype(float) * method1_weight +
        method2_mask.astype(float) * method2_weight +
        method3_mask.astype(float) * method3_weight +
        method4_mask.astype(float) * method4_weight
    )
    
    total_weight = method1_weight + method2_weight + method3_weight + method4_weight
    voting_threshold = params['voting_threshold']
    
    combined_mask = (weighted_score >= total_weight * voting_threshold) & plate_mask
    
    # === MORPHOLOGICAL CLEANUP ===
    combined_mask = ndimage.binary_opening(combined_mask, iterations=1)
    combined_mask = ndimage.binary_closing(combined_mask, iterations=1)
    combined_mask = ndimage.binary_fill_holes(combined_mask)
    
    # === WATERSHED SEGMENTATION ===
    segmented, num_segments = watershed_segmentation(combined_mask, enhanced, params)
    
    # === FILTER BY SIZE AND CIRCULARITY ===
    if isinstance(segmented, np.ndarray) and segmented.max() > 0:
        labeled = segmented
        num_features = int(segmented.max())
    else:
        labeled, num_features = label(combined_mask)
    
    valid_colonies = 0
    colony_sizes = []
    colony_circularities = []
    
    min_size = params['min_colony_size']
    max_size = params['max_colony_size']
    min_circ = params['min_circularity']
    
    for i in range(1, num_features + 1):
        component = labeled == i
        size = np.sum(component)
        
        if min_size <= size <= max_size:
            circ = calculate_circularity(component)
            
            if circ >= min_circ:
                valid_colonies += 1
                colony_sizes.append(size)
                colony_circularities.append(circ)
    
    avg_size = np.mean(colony_sizes) if colony_sizes else 0
    avg_circ = np.mean(colony_circularities) if colony_circularities else 0
    
    debug_info = {
        'mode': mode,
        'voting_threshold': voting_threshold,
        'method1_detections': int(np.sum(method1_mask)),
        'method2_detections': int(np.sum(method2_mask)),
        'method3_detections': int(np.sum(method3_mask)),
        'method4_detections': int(np.sum(method4_mask)),
        'combined_pixels': int(np.sum(combined_mask)),
        'segments_before_filter': num_features,
        'bg_mean': round(float(bg_mean), 1),
        'bg_std': round(float(bg_std), 1)
    }
    
    return valid_colonies, avg_size, avg_circ, labeled, debug_info


def detect_hemolysis_hsv(rgb_array, plate_mask):
    """Hemolysis detection using HSV color space"""
    hsv = rgb_to_hsv(rgb_array)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
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
            label_parts.append("beta-hemolytic – follow Strep protocol")
        elif hemolysis['type'] == 'alpha':
            label_parts.append("alpha-hemolytic – rule out S. pneumoniae")
        else:
            label_parts.append("non-hemolytic")
    
    return {
        'label': " – ".join(label_parts),
        'confidence': confidence,
        'requires_review': review
    }


def analyze_plate(image_bytes, media_type='blood_agar', detection_mode='sensitive'):
    """Main analysis function - v1.7.0 with dual mode"""
    img = load_and_resize_image(image_bytes)
    plate_mask, center, radius = create_plate_mask(img.shape, rgb_array=img)
    luminance = get_luminance(img)
    
    # Validate detection mode
    if detection_mode not in ['sensitive', 'strict']:
        detection_mode = 'sensitive'
    
    colony_count, avg_size, avg_circ, labeled, debug_info = colony_detection_dual_mode(
        img, luminance, plate_mask, mode=detection_mode
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
        'version': '1.7.0-dual-mode'
    }


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.7.0-dual-mode',
        'max_image_size': MAX_IMAGE_SIZE,
        'supported_media_types': SUPPORTED_MEDIA_TYPES,
        'detection_modes': {
            'sensitive': 'Best for sparse/low-density plates - catches more colonies',
            'strict': 'Best for dense/high-density plates - filters more aggressively'
        },
        'algorithms': [
            'CLAHE preprocessing',
            'Adaptive thresholding (mode-specific C values)',
            'HSV color space analysis',
            'Local maxima detection',
            'Weighted voting (mode-specific threshold)',
            'Watershed segmentation',
            'Circularity filtering'
        ]
    })


@app.route('/warmup', methods=['GET'])
def warmup():
    """Warm-up endpoint"""
    return jsonify({
        'status': 'warm',
        'version': '1.7.0-dual-mode',
        'message': 'Service is ready for analysis'
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze plate image"""
    try:
        media_type = request.form.get('media_type', 'blood_agar')
        if media_type not in SUPPORTED_MEDIA_TYPES:
            media_type = 'blood_agar'
        
        # Get detection mode - NEW PARAMETER
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
