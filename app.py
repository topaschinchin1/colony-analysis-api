"""
Colony Counting and Hemolysis Detection API
Version: 1.5.3 - Research-based improvements

Based on:
- OpenCFU: Recursive thresholding + watershed segmentation
- CFUCounter: Iterative adaptive thresholding + local minima watershed
- Savardi et al.: HSV color space for hemolysis detection
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

# Colony detection parameters (calibrated)
MIN_COLONY_SIZE = 8       # pixels - filter tiny noise
MAX_COLONY_SIZE = 5000    # pixels - filter large artifacts
MIN_CIRCULARITY = 0.25    # 0-1 scale, circles = 1.0
BRIGHTNESS_THRESHOLD = 0.8  # std deviations above background


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


def rgb_to_hsv(rgb_array):
    """Convert RGB to HSV color space - better for color analysis"""
    rgb = rgb_array.astype(float) / 255.0
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    diff = max_c - min_c
    
    # Value
    v = max_c
    
    # Saturation
    s = np.where(max_c != 0, diff / max_c, 0)
    
    # Hue
    h = np.zeros_like(max_c)
    mask = diff != 0
    
    # Red is max
    idx = (max_c == r) & mask
    h[idx] = (60 * ((g[idx] - b[idx]) / diff[idx]) + 360) % 360
    
    # Green is max
    idx = (max_c == g) & mask
    h[idx] = (60 * ((b[idx] - r[idx]) / diff[idx]) + 120) % 360
    
    # Blue is max
    idx = (max_c == b) & mask
    h[idx] = (60 * ((r[idx] - g[idx]) / diff[idx]) + 240) % 360
    
    return np.stack([h, s * 255, v * 255], axis=-1)


def get_luminance(rgb_array):
    """Calculate luminance from RGB"""
    return 0.299 * rgb_array[:,:,0] + 0.587 * rgb_array[:,:,1] + 0.114 * rgb_array[:,:,2]


def detect_plate_region(rgb_array):
    """
    Detect blood agar plate using HSV color space
    Blood agar: Hue ~0-30 (red), high saturation
    RESTORED from v1.5.3 (which had 94% accuracy on GAS-03)
    """
    hsv = rgb_to_hsv(rgb_array)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    # Blood agar is RED (hue 0-30 or 330-360)
    is_red = ((h <= 30) | (h >= 330)) & (s > 50) & (v > 40) & (v < 240)
    
    # Also accept orange-red range (0-45) - v1.5.3 original
    is_red_expanded = ((h <= 45) | (h >= 315)) & (s > 40) & (v > 30) & (v < 250)
    
    from scipy import ndimage
    
    # Use expanded detection, clean up
    agar_mask = ndimage.binary_closing(is_red_expanded, iterations=8)
    agar_mask = ndimage.binary_opening(agar_mask, iterations=4)
    agar_mask = ndimage.binary_fill_holes(agar_mask)
    
    return agar_mask


def create_plate_mask(shape, rgb_array=None):
    """Create plate mask combining circular and color detection - v1.5.3 original"""
    h, w = shape[:2]
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 2 - 5
    
    # Basic circular mask (fallback) - v1.5.3 used 0.88
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    circular_mask = dist <= radius * 0.88  # Conservative
    
    if rgb_array is not None:
        color_mask = detect_plate_region(rgb_array)
        
        # Combine: use color mask but only within reasonable bounds
        combined = circular_mask & color_mask
        
        # If color detection got enough area, use it (v1.5.3 used 0.25)
        if np.sum(combined) > np.sum(circular_mask) * 0.25:
            return combined, (center_x, center_y), radius
    
    return circular_mask, (center_x, center_y), radius


def watershed_colony_segmentation(binary_mask, luminance):
    """
    Watershed segmentation to separate touching colonies
    Based on CFUCounter approach: local minima detection + watershed
    """
    from scipy import ndimage
    from scipy.ndimage import distance_transform_edt, label
    
    if np.sum(binary_mask) == 0:
        return binary_mask, 0
    
    # Distance transform - peaks are colony centers
    distance = distance_transform_edt(binary_mask)
    
    # Find local maxima in distance transform (colony centers)
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(distance, size=7)
    peaks = (distance == local_max) & (distance > 2)
    
    # Label the peaks as markers
    markers, num_markers = label(peaks)
    
    if num_markers == 0:
        # No clear peaks, just use connected components
        labeled, num_features = label(binary_mask)
        return labeled, num_features
    
    # Simple watershed using distance-based region growing
    # Assign each foreground pixel to nearest marker
    try:
        from scipy.ndimage import distance_transform_edt
        
        # For each marker, compute distance from marker
        # Assign pixels to closest marker
        segmented = np.zeros_like(binary_mask, dtype=int)
        
        for i in range(1, num_markers + 1):
            marker_mask = markers == i
            if np.any(marker_mask):
                # Dilate marker to fill region
                dist_from_marker = distance_transform_edt(~marker_mask)
                # Will be assigned in priority order (closest wins)
                segmented = np.where(
                    (binary_mask) & (segmented == 0) & (dist_from_marker < 50),
                    i,
                    segmented
                )
        
        # Fill remaining foreground pixels with nearest marker
        remaining = binary_mask & (segmented == 0)
        if np.any(remaining):
            labeled_remaining, _ = label(remaining)
            max_label = segmented.max()
            segmented = np.where(remaining, labeled_remaining + max_label, segmented)
            
    except Exception:
        # Fallback to simple labeling
        segmented, _ = label(binary_mask)
    
    return segmented, segmented.max()


def calculate_circularity(component_mask):
    """
    Calculate circularity of a component
    Circularity = 4π × Area / Perimeter²
    Perfect circle = 1.0
    """
    from scipy import ndimage
    
    area = np.sum(component_mask)
    if area == 0:
        return 0
    
    # Perimeter = pixels on boundary
    dilated = ndimage.binary_dilation(component_mask)
    perimeter = np.sum(dilated & ~component_mask)
    
    if perimeter == 0:
        return 0
    
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return min(circularity, 1.0)  # Cap at 1.0


def iterative_threshold_detection(luminance, plate_mask, rgb_array):
    """
    Iterative adaptive thresholding - inspired by OpenCFU
    Try multiple thresholds and combine results
    """
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter
    
    plate_pixels = luminance[plate_mask]
    background = np.median(plate_pixels)
    std_dev = np.std(plate_pixels)
    
    # Smooth to reduce noise
    smoothed = gaussian_filter(luminance.astype(float), sigma=1.0)
    
    # Multiple threshold levels
    thresholds = [
        background + 0.6 * std_dev,  # Sensitive
        background + 0.9 * std_dev,  # Medium
        background + 1.2 * std_dev,  # Conservative
    ]
    
    # Score map - accumulate detections across thresholds
    score_map = np.zeros_like(luminance, dtype=float)
    
    for thresh in thresholds:
        detected = (smoothed > thresh) & plate_mask
        score_map += detected.astype(float)
    
    # Require detection at 2+ thresholds
    combined = (score_map >= 2) & plate_mask
    
    return combined


def colony_detection_v159(rgb_array, luminance, plate_mask):
    """
    Version 1.5.9 colony detection - EXACT v1.5.3 parameters restored
    v1.5.3 had 94% accuracy on GAS-03, 50% on GAS-01
    """
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter, maximum_filter, label
    
    # === METHOD 1: Iterative thresholding ===
    iterative_mask = iterative_threshold_detection(luminance, plate_mask, rgb_array)
    
    # === METHOD 2: HSV-based detection - EXACT v1.5.3 ===
    hsv = rgb_to_hsv(rgb_array)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    plate_v = v[plate_mask]
    plate_s = s[plate_mask]
    bg_v = np.median(plate_v)
    bg_s = np.median(plate_s)
    
    # v1.5.3 exact: brightness +10, saturation 0.85
    is_bright_hsv = v > (bg_v + 10)
    is_less_saturated = s < (bg_s * 0.85)
    hsv_colonies = is_bright_hsv & is_less_saturated & plate_mask
    
    # === METHOD 3: Local maxima detection - EXACT v1.5.3 ===
    blurred = gaussian_filter(luminance.astype(float), sigma=1.2)  # v1.5.3 exact
    local_max = maximum_filter(blurred, size=5)
    
    plate_lum = luminance[plate_mask]
    bg_lum = np.median(plate_lum)
    std_lum = np.std(plate_lum)
    
    # v1.5.3 exact: 0.5 std
    peaks = (blurred == local_max) & (luminance > bg_lum + 0.5 * std_lum) & plate_mask
    peaks_dilated = ndimage.binary_dilation(peaks, iterations=2)
    
    # === COMBINE METHODS - v1.5.3 exact ===
    method_agreement = (
        iterative_mask.astype(int) + 
        hsv_colonies.astype(int) + 
        peaks_dilated.astype(int)
    )
    
    # v1.5.3 exact: require 2+ methods
    combined_mask = (method_agreement >= 2) & plate_mask
    
    # Light cleanup - v1.5.3 exact
    combined_mask = ndimage.binary_opening(combined_mask, iterations=1)
    combined_mask = ndimage.binary_closing(combined_mask, iterations=1)
    
    # === WATERSHED SEGMENTATION ===
    segmented, num_segments = watershed_colony_segmentation(combined_mask, luminance)
    
    # === FILTER BY SIZE AND CIRCULARITY - v1.5.3 exact ===
    if isinstance(segmented, np.ndarray) and segmented.max() > 0:
        labeled = segmented
        num_features = int(segmented.max())
    else:
        labeled, num_features = label(combined_mask)
    
    valid_colonies = 0
    colony_sizes = []
    colony_circularities = []
    
    # v1.5.3 exact filters
    min_size = 8       # MIN_COLONY_SIZE
    max_size = 5000    # MAX_COLONY_SIZE
    min_circ = 0.25    # MIN_CIRCULARITY
    
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
    
    return valid_colonies, avg_size, avg_circ, labeled


def detect_hemolysis_hsv(rgb_array, plate_mask):
    """
    Hemolysis detection using HSV color space - v1.5.4 FIXED
    
    Beta hemolysis: Clear zones around colonies (high V, low S)
    Alpha hemolysis: Green zones (H shifts toward green)
    Gamma: No change from background
    """
    hsv = rgb_to_hsv(rgb_array)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    plate_h = h[plate_mask]
    plate_s = s[plate_mask]
    plate_v = v[plate_mask]
    
    bg_h = np.median(plate_h)
    bg_s = np.median(plate_s)
    bg_v = np.median(plate_v)
    
    # === BETA HEMOLYSIS (FIXED - more sensitive) ===
    # Clear zones: brighter AND/OR less saturated than background
    is_bright = v > (bg_v + 8)  # Was +15
    is_desaturated = s < (bg_s * 0.80)  # Was 0.7
    
    # Beta can be detected by EITHER brightness OR desaturation
    beta_zones = (is_bright | is_desaturated) & plate_mask
    
    # But exclude the very center (colonies themselves are also bright)
    # Focus on the halo region
    from scipy.ndimage import binary_erosion, binary_dilation
    
    # Find colony-like bright spots
    very_bright = v > (bg_v + 25)
    colony_cores = very_bright & plate_mask
    
    # Dilate to get halo region
    if np.any(colony_cores):
        halo_region = binary_dilation(colony_cores, iterations=8) & ~colony_cores
        beta_in_halo = beta_zones & halo_region
        beta_ratio = np.sum(beta_in_halo) / np.sum(plate_mask) if np.sum(plate_mask) > 0 else 0
    else:
        beta_ratio = np.sum(beta_zones) / np.sum(plate_mask) if np.sum(plate_mask) > 0 else 0
    
    # Also check overall desaturation (clear plates have less red)
    overall_desat = np.mean(plate_s) < 120  # Blood agar normally has high saturation
    
    # === ALPHA HEMOLYSIS ===
    is_greenish = (h > 40) & (h < 160)
    is_darker = v < (bg_v - 3)  # Was -5
    alpha_zones = is_greenish & is_darker & plate_mask
    alpha_ratio = np.sum(alpha_zones) / np.sum(plate_mask) if np.sum(plate_mask) > 0 else 0
    
    # === CLASSIFICATION (more sensitive to beta) ===
    if beta_ratio > 0.02 or overall_desat:  # Was 0.03
        hemo_type = 'beta'
        confidence = min(0.95, 0.60 + beta_ratio * 4)
    elif alpha_ratio > 0.04:  # Was 0.05
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


def analyze_plate(image_bytes, media_type='blood_agar'):
    """Main analysis function - v1.5.9 (exact v1.5.3 restoration)"""
    img = load_and_resize_image(image_bytes)
    
    # Create plate mask - v1.5.3 exact
    plate_mask, center, radius = create_plate_mask(img.shape, rgb_array=img)
    
    luminance = get_luminance(img)
    
    # Colony detection - v1.5.3 exact parameters
    colony_count, avg_size, avg_circ, labeled = colony_detection_v159(
        img, luminance, plate_mask
    )
    
    # Hemolysis detection with HSV
    hemolysis = detect_hemolysis_hsv(img, plate_mask)
    
    # Generate decision
    decision = generate_decision(colony_count, hemolysis, media_type)
    
    return {
        'status': 'success',
        'colony_count': colony_count,
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
        'version': '1.5.9-v153-restored'
    }
    }


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'version': '1.5.9-v153-restored',
        'max_image_size': MAX_IMAGE_SIZE,
        'supported_media_types': SUPPORTED_MEDIA_TYPES,
        'algorithms': [
            'HSV color space analysis',
            'Iterative adaptive thresholding', 
            'Watershed segmentation',
            'Circularity filtering'
        ],
        'note': 'Exact v1.5.3 parameters restored (94% on GAS-03)'
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze plate image"""
    try:
        media_type = request.form.get('media_type', 'blood_agar')
        if media_type not in SUPPORTED_MEDIA_TYPES:
            media_type = 'blood_agar'
        
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                image_bytes = file.read()
                result = analyze_plate(image_bytes, media_type)
                return jsonify(result)
        
        if request.is_json:
            data = request.get_json()
            if data and 'image_base64' in data:
                image_bytes = base64.b64decode(data['image_base64'])
                result = analyze_plate(image_bytes, media_type)
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
