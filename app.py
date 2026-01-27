"""
Colony Counting and Hemolysis Detection API
Version: 1.7.0 - OpenCFU-inspired multi-thresholding

MAJOR CHANGES:
1. OpenCFU-style global multi-thresholding (10 levels)
2. Per-colony intensity validation
3. Improved watershed with distance transform
4. Colony merging across threshold levels
5. Better noise filtering

Based on:
- OpenCFU (PLOS ONE 2013): Multi-thresholding + particle filtering
- CFUCounter: Local minima watershed markers
- krransby/colony-counter: Hough + Watershed pipeline
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

# Colony detection parameters
MIN_COLONY_SIZE = 8       # pixels
MAX_COLONY_SIZE = 5000    # pixels
MIN_CIRCULARITY = 0.22    # slightly relaxed for watershed fragments


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


def opencfu_multi_threshold(luminance, plate_mask, num_thresholds=10):
    """
    OpenCFU-style multi-thresholding
    
    Apply multiple global thresholds and track detections across levels.
    This is more robust than single-threshold approaches.
    """
    from scipy.ndimage import label
    
    plate_pixels = luminance[plate_mask]
    min_val = np.percentile(plate_pixels, 5)
    max_val = np.percentile(plate_pixels, 95)
    
    # Generate threshold levels
    thresholds = np.linspace(min_val + 0.3 * (max_val - min_val), 
                             min_val + 0.8 * (max_val - min_val), 
                             num_thresholds)
    
    # Track colony candidates across thresholds
    # Each candidate: (centroid_y, centroid_x, area, threshold_level)
    all_candidates = []
    
    for thresh_idx, thresh in enumerate(thresholds):
        binary = (luminance > thresh) & plate_mask
        
        # Label connected components
        labeled, num_features = label(binary)
        
        for i in range(1, num_features + 1):
            component = labeled == i
            area = np.sum(component)
            
            if MIN_COLONY_SIZE <= area <= MAX_COLONY_SIZE:
                # Get centroid
                coords = np.where(component)
                cy, cx = np.mean(coords[0]), np.mean(coords[1])
                
                # Get mean intensity
                mean_intensity = np.mean(luminance[component])
                
                all_candidates.append({
                    'cy': cy,
                    'cx': cx,
                    'area': area,
                    'intensity': mean_intensity,
                    'thresh_idx': thresh_idx,
                    'mask': component
                })
    
    return all_candidates, thresholds


def merge_colony_candidates(candidates, merge_distance=15):
    """
    Merge colony candidates detected at different thresholds
    
    If two candidates are within merge_distance pixels, they're likely
    the same colony detected at different threshold levels.
    """
    if not candidates:
        return []
    
    # Sort by area (larger first)
    candidates = sorted(candidates, key=lambda x: -x['area'])
    
    merged = []
    used = set()
    
    for i, cand in enumerate(candidates):
        if i in used:
            continue
        
        # Find all candidates near this one
        group = [cand]
        used.add(i)
        
        for j, other in enumerate(candidates):
            if j in used:
                continue
            
            dist = np.sqrt((cand['cy'] - other['cy'])**2 + 
                          (cand['cx'] - other['cx'])**2)
            
            if dist < merge_distance:
                group.append(other)
                used.add(j)
        
        # Merge: take average position, max area, count detections
        merged_cand = {
            'cy': np.mean([c['cy'] for c in group]),
            'cx': np.mean([c['cx'] for c in group]),
            'area': max(c['area'] for c in group),
            'intensity': np.mean([c['intensity'] for c in group]),
            'detection_count': len(group),  # How many thresholds detected this
            'masks': [c['mask'] for c in group]
        }
        merged.append(merged_cand)
    
    return merged


def watershed_separation(binary_mask, luminance):
    """
    Watershed segmentation for overlapping colonies
    Uses distance transform + intensity for markers
    """
    from scipy import ndimage
    from scipy.ndimage import distance_transform_edt, label, maximum_filter
    
    if np.sum(binary_mask) == 0:
        return np.zeros_like(binary_mask, dtype=int), 0
    
    # Distance transform
    distance = distance_transform_edt(binary_mask)
    
    # Find local maxima as markers
    footprint_size = 9
    local_max = maximum_filter(distance, size=footprint_size)
    distance_threshold = max(2, np.percentile(distance[binary_mask], 25))
    peaks = (distance == local_max) & (distance >= distance_threshold) & binary_mask
    
    # Also consider intensity peaks
    intensity_masked = np.where(binary_mask, luminance, 0)
    intensity_local_max = maximum_filter(intensity_masked, size=7)
    intensity_peaks = (intensity_masked == intensity_local_max) & binary_mask & (distance > 1)
    
    # Combine markers
    combined_peaks = peaks | intensity_peaks
    markers, num_markers = label(combined_peaks)
    
    if num_markers == 0:
        return label(binary_mask)
    
    # Simple region growing from markers
    segmented = np.zeros_like(binary_mask, dtype=int)
    segmented[markers > 0] = markers[markers > 0]
    
    # Get marker centroids
    marker_coords = {}
    for i in range(1, num_markers + 1):
        coords = np.where(markers == i)
        if len(coords[0]) > 0:
            marker_coords[i] = (np.mean(coords[0]), np.mean(coords[1]))
    
    # Assign each foreground pixel to nearest marker
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
    
    # Handle remaining pixels
    remaining = binary_mask & (segmented == 0)
    if np.any(remaining):
        remaining_labeled, _ = label(remaining)
        max_label = segmented.max()
        segmented = np.where(remaining, remaining_labeled + max_label, segmented)
    
    return segmented, int(segmented.max())


def colony_detection_v170(rgb_array, luminance, plate_mask):
    """
    Version 1.7.0 - OpenCFU-inspired colony detection
    
    Pipeline:
    1. CLAHE preprocessing
    2. Multi-threshold detection (OpenCFU style)
    3. Candidate merging across thresholds
    4. Filter by detection count (must be detected at 2+ thresholds)
    5. Watershed for overlapping colonies
    6. Per-colony validation (size, circularity, intensity)
    """
    from scipy import ndimage
    from scipy.ndimage import label, gaussian_filter
    
    # === PREPROCESSING ===
    denoised = ndimage.median_filter(luminance, size=3)
    enhanced = apply_clahe(denoised.astype(np.uint8), clip_limit=2.0, tile_size=8)
    enhanced = enhanced.astype(float)
    
    # Background statistics
    plate_pixels = enhanced[plate_mask]
    bg_mean = np.median(plate_pixels)
    bg_std = np.std(plate_pixels)
    
    # === OPENCFU-STYLE MULTI-THRESHOLDING ===
    candidates, thresholds = opencfu_multi_threshold(enhanced, plate_mask, num_thresholds=10)
    
    # === MERGE CANDIDATES ACROSS THRESHOLDS ===
    merged_candidates = merge_colony_candidates(candidates, merge_distance=12)
    
    # === FILTER BY DETECTION COUNT ===
    # Colonies detected at multiple thresholds are more reliable
    # Require detection at 2+ threshold levels
    reliable_candidates = [c for c in merged_candidates if c['detection_count'] >= 2]
    
    # === ADDITIONAL HSV-BASED DETECTION ===
    # Catch colonies that multi-threshold might miss
    hsv = rgb_to_hsv(rgb_array)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    plate_v = v[plate_mask]
    plate_s = s[plate_mask]
    bg_v = np.median(plate_v)
    bg_s = np.median(plate_s)
    
    # HSV detection mask
    hsv_mask = (v > bg_v + 10) & (s < bg_s * 0.85) & plate_mask
    hsv_mask = ndimage.binary_opening(hsv_mask, iterations=1)
    
    # === CREATE COMBINED BINARY MASK ===
    combined_mask = np.zeros_like(plate_mask, dtype=bool)
    
    # Add reliable multi-threshold candidates
    for cand in reliable_candidates:
        # Use the largest mask from this candidate
        if cand['masks']:
            largest_mask = max(cand['masks'], key=lambda m: np.sum(m))
            combined_mask |= largest_mask
    
    # Add HSV detections that overlap with at least some multi-threshold detection
    # This helps catch colonies that were detected at only 1 threshold
    for cand in merged_candidates:
        if cand['detection_count'] == 1:
            # Check if HSV also detected this region
            if cand['masks']:
                cand_mask = cand['masks'][0]
                overlap = np.sum(cand_mask & hsv_mask)
                if overlap > np.sum(cand_mask) * 0.3:  # 30% overlap with HSV
                    combined_mask |= cand_mask
    
    # === WATERSHED SEGMENTATION ===
    segmented, num_segments = watershed_separation(combined_mask, enhanced)
    
    # === VALIDATE EACH COLONY ===
    valid_colonies = 0
    colony_sizes = []
    colony_circularities = []
    colony_intensities = []
    
    for i in range(1, num_segments + 1):
        component = segmented == i
        area = np.sum(component)
        
        if MIN_COLONY_SIZE <= area <= MAX_COLONY_SIZE:
            circ = calculate_circularity(component)
            mean_intensity = np.mean(enhanced[component])
            
            # Intensity validation: colony should be brighter than background
            intensity_ok = mean_intensity > bg_mean + 0.3 * bg_std
            
            # Circularity check
            circ_ok = circ >= MIN_CIRCULARITY
            
            if intensity_ok and circ_ok:
                valid_colonies += 1
                colony_sizes.append(area)
                colony_circularities.append(circ)
                colony_intensities.append(mean_intensity)
    
    avg_size = np.mean(colony_sizes) if colony_sizes else 0
    avg_circ = np.mean(colony_circularities) if colony_circularities else 0
    
    debug_info = {
        'total_candidates': len(candidates),
        'merged_candidates': len(merged_candidates),
        'reliable_candidates': len(reliable_candidates),
        'num_thresholds': len(thresholds),
        'threshold_range': f"{thresholds[0]:.1f}-{thresholds[-1]:.1f}",
        'segments_before_filter': num_segments,
        'bg_mean': round(float(bg_mean), 1),
        'bg_std': round(float(bg_std), 1),
        'avg_colony_intensity': round(float(np.mean(colony_intensities)), 1) if colony_intensities else 0
    }
    
    return valid_colonies, avg_size, avg_circ, segmented, debug_info


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


def analyze_plate(image_bytes, media_type='blood_agar'):
    """Main analysis function - v1.7.0"""
    img = load_and_resize_image(image_bytes)
    plate_mask, center, radius = create_plate_mask(img.shape, rgb_array=img)
    luminance = get_luminance(img)
    
    colony_count, avg_size, avg_circ, labeled, debug_info = colony_detection_v170(
        img, luminance, plate_mask
    )
    
    hemolysis = detect_hemolysis_hsv(img, plate_mask)
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
        'debug': debug_info,
        'version': '1.7.0-opencfu-style'
    }


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.7.0-opencfu-style',
        'max_image_size': MAX_IMAGE_SIZE,
        'supported_media_types': SUPPORTED_MEDIA_TYPES,
        'algorithms': [
            'CLAHE preprocessing',
            'OpenCFU-style multi-thresholding (10 levels)',
            'Cross-threshold candidate merging',
            'Detection count filtering (2+ thresholds)',
            'HSV color validation',
            'Watershed segmentation',
            'Per-colony intensity validation',
            'Circularity filtering'
        ],
        'based_on': 'OpenCFU (PLOS ONE 2013), CFUCounter'
    })


@app.route('/warmup', methods=['GET'])
def warmup():
    """Warm-up endpoint"""
    return jsonify({
        'status': 'warm',
        'version': '1.7.0-opencfu-style',
        'message': 'Service is ready for analysis'
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
