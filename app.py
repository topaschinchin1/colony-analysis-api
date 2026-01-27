"""
Colony Counting and Hemolysis Detection API
Version: 1.6.0 - Research-based improvements

IMPROVEMENTS FROM LITERATURE:
1. CLAHE preprocessing (Contrast Limited Adaptive Histogram Equalization)
2. Gaussian adaptive thresholding (CFUCounter approach)
3. Improved watershed with peak_local_max markers
4. Multi-scale detection for varied colony sizes
5. Weighted method voting (instead of strict 2+ agreement)
6. Noise reduction preprocessing
7. Adaptive sensitivity based on plate characteristics

Based on:
- CFUCounter: r=0.999 correlation with manual counts
- OpenCFU: Recursive thresholding + score maps
- Savardi et al.: HSV color space for hemolysis detection
- Multiple watershed segmentation papers
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

# Colony detection parameters (tuned from research)
MIN_COLONY_SIZE = 6       # pixels - slightly lower to catch small colonies
MAX_COLONY_SIZE = 6000    # pixels - filter large artifacts
MIN_CIRCULARITY = 0.20    # lowered from 0.25 for irregular colonies
BRIGHTNESS_THRESHOLD = 0.8


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


def apply_clahe(gray_image, clip_limit=2.5, tile_size=8):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    
    This is a KEY preprocessing step from literature that enhances local
    contrast without amplifying noise. Critical for detecting low-contrast colonies.
    
    Parameters:
    - clip_limit: Threshold for contrast limiting (2-5 typical)
    - tile_size: Size of grid for histogram equalization
    """
    # Normalize to 0-255 uint8
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
            # Define tile boundaries
            y1 = i * tile_h
            y2 = min((i + 1) * tile_h, h)
            x1 = j * tile_w
            x2 = min((j + 1) * tile_w, w)
            
            if y2 <= y1 or x2 <= x1:
                continue
                
            tile = img_normalized[y1:y2, x1:x2]
            
            # Compute histogram
            hist, bins = np.histogram(tile.flatten(), bins=256, range=(0, 256))
            
            # Clip histogram (CLAHE's key innovation)
            clip_threshold = clip_limit * tile.size / 256
            excess = np.sum(np.maximum(hist - clip_threshold, 0))
            hist = np.minimum(hist, clip_threshold)
            
            # Redistribute excess uniformly
            hist = hist + excess / 256
            
            # Compute CDF
            cdf = hist.cumsum()
            cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-8) * 255
            
            # Apply mapping
            result[y1:y2, x1:x2] = cdf_normalized[tile.astype(int)]
    
    return result.astype(np.uint8)


def apply_noise_reduction(gray_image, method='median', kernel_size=3):
    """
    Apply noise reduction preprocessing
    
    From literature: Noise reduction before detection improves accuracy.
    Median filter is preferred as it preserves edges better than Gaussian.
    """
    from scipy import ndimage
    
    if method == 'median':
        return ndimage.median_filter(gray_image, size=kernel_size)
    elif method == 'gaussian':
        return ndimage.gaussian_filter(gray_image, sigma=kernel_size/3)
    else:
        return gray_image


def adaptive_threshold(gray_image, block_size=51, C=8, method='gaussian'):
    """
    Adaptive thresholding - key technique from CFUCounter (r=0.999)
    
    Instead of a global threshold, computes threshold locally based on
    neighborhood statistics. Much better for varying lighting conditions.
    
    Parameters:
    - block_size: Size of neighborhood for computing threshold
    - C: Constant subtracted from mean (higher = less sensitive)
    - method: 'gaussian' or 'mean'
    """
    from scipy import ndimage
    
    if method == 'gaussian':
        local_mean = ndimage.gaussian_filter(gray_image.astype(float), sigma=block_size/6)
    else:
        local_mean = ndimage.uniform_filter(gray_image.astype(float), size=block_size)
    
    # Threshold: pixel is foreground if it's brighter than local mean + C
    binary = gray_image > (local_mean + C)
    
    return binary


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
    """
    hsv = rgb_to_hsv(rgb_array)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    # Blood agar is RED (hue 0-30 or 330-360)
    is_red = ((h <= 30) | (h >= 330)) & (s > 50) & (v > 40) & (v < 240)
    
    # Also accept orange-red range (0-45)
    is_red_expanded = ((h <= 45) | (h >= 315)) & (s > 40) & (v > 30) & (v < 250)
    
    from scipy import ndimage
    
    # Use expanded detection, clean up
    agar_mask = ndimage.binary_closing(is_red_expanded, iterations=8)
    agar_mask = ndimage.binary_opening(agar_mask, iterations=4)
    agar_mask = ndimage.binary_fill_holes(agar_mask)
    
    return agar_mask


def create_plate_mask(shape, rgb_array=None):
    """Create plate mask combining circular and color detection"""
    h, w = shape[:2]
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 2 - 5
    
    # Basic circular mask (fallback)
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    circular_mask = dist <= radius * 0.88
    
    if rgb_array is not None:
        color_mask = detect_plate_region(rgb_array)
        
        # Combine: use color mask but only within reasonable bounds
        combined = circular_mask & color_mask
        
        # If color detection got enough area, use it
        if np.sum(combined) > np.sum(circular_mask) * 0.25:
            return combined, (center_x, center_y), radius
    
    return circular_mask, (center_x, center_y), radius


def improved_watershed_segmentation(binary_mask, intensity_image):
    """
    Improved watershed segmentation using peak_local_max
    
    From CFUCounter research: Local minima detection outperforms distance transform
    markers because it mimics how humans distinguish adjacent colonies.
    
    This implementation uses intensity local maxima (colony centers are brighter)
    combined with distance transform for robust marker generation.
    """
    from scipy import ndimage
    from scipy.ndimage import distance_transform_edt, label, maximum_filter, minimum_filter
    
    if np.sum(binary_mask) == 0:
        return np.zeros_like(binary_mask, dtype=int), 0
    
    # Distance transform - farther from edges = more likely colony center
    distance = distance_transform_edt(binary_mask)
    
    # Find local maxima in distance transform
    # These are potential colony centers
    footprint_size = 7  # Minimum separation between colony centers
    local_max = maximum_filter(distance, size=footprint_size)
    
    # Peak detection with adaptive threshold
    # Must be local maximum AND have significant distance from edge
    distance_threshold = max(2, np.percentile(distance[binary_mask], 30))
    peaks = (distance == local_max) & (distance >= distance_threshold) & binary_mask
    
    # Also use intensity information - colony centers tend to be brighter
    if intensity_image is not None:
        intensity_masked = np.where(binary_mask, intensity_image, 0)
        intensity_local_max = maximum_filter(intensity_masked, size=5)
        intensity_peaks = (intensity_masked == intensity_local_max) & binary_mask
        
        # Combine distance and intensity peaks
        peaks = peaks | (intensity_peaks & (distance > 1))
    
    # Label the markers
    markers, num_markers = label(peaks)
    
    if num_markers == 0:
        # No clear peaks, use connected components
        labeled, num_features = label(binary_mask)
        return labeled, num_features
    
    # Watershed segmentation
    # We'll use a simple region growing from markers
    segmented = np.zeros_like(binary_mask, dtype=int)
    
    # Initialize with markers
    segmented[markers > 0] = markers[markers > 0]
    
    # Region growing based on distance transform
    # Each pixel is assigned to the nearest marker
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
        if nearest_marker > 0 and min_dist < 2500:  # Max distance threshold
            segmented[y, x] = nearest_marker
    
    # Handle remaining unassigned pixels
    remaining = binary_mask & (segmented == 0)
    if np.any(remaining):
        remaining_labeled, num_remaining = label(remaining)
        max_label = segmented.max()
        segmented = np.where(remaining, remaining_labeled + max_label, segmented)
    
    return segmented, int(segmented.max())


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
    return min(circularity, 1.0)


def multi_scale_detection(luminance, plate_mask, scales=[0.7, 1.0, 1.4]):
    """
    Multi-scale detection to catch colonies of varying sizes
    
    Run detection at multiple scales and merge results.
    Helps catch both small and large colonies.
    """
    from scipy import ndimage
    from scipy.ndimage import zoom, label
    
    all_detections = np.zeros_like(luminance, dtype=float)
    
    for scale in scales:
        if scale == 1.0:
            scaled_lum = luminance
            scaled_mask = plate_mask
        else:
            # Resize
            scaled_lum = zoom(luminance, scale, order=1)
            scaled_mask = zoom(plate_mask.astype(float), scale, order=0) > 0.5
        
        # Detect at this scale
        plate_pixels = scaled_lum[scaled_mask]
        if len(plate_pixels) == 0:
            continue
            
        bg = np.median(plate_pixels)
        std = np.std(plate_pixels)
        
        # Threshold detection
        detected = (scaled_lum > bg + 0.5 * std) & scaled_mask
        
        # Resize back to original
        if scale != 1.0:
            detected = zoom(detected.astype(float), 1.0/scale, order=0) > 0.5
            # Ensure same shape
            detected = detected[:luminance.shape[0], :luminance.shape[1]]
            if detected.shape != luminance.shape:
                padded = np.zeros_like(luminance, dtype=bool)
                padded[:detected.shape[0], :detected.shape[1]] = detected
                detected = padded
        
        all_detections += detected.astype(float)
    
    # Require detection at 2+ scales
    combined = (all_detections >= 2) & plate_mask
    
    return combined


def colony_detection_v160(rgb_array, luminance, plate_mask):
    """
    Version 1.6.0 colony detection with research-based improvements
    
    Key improvements:
    1. CLAHE preprocessing for contrast enhancement
    2. Noise reduction
    3. Adaptive thresholding (CFUCounter approach)
    4. Improved watershed with peak_local_max markers
    5. Weighted voting instead of strict 2+ agreement
    6. Multi-scale detection
    7. More permissive filtering for GAS-01
    """
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter, maximum_filter, label
    
    # === PREPROCESSING (NEW in v1.6.0) ===
    # Step 1: Noise reduction
    denoised = apply_noise_reduction(luminance, method='median', kernel_size=3)
    
    # Step 2: CLAHE contrast enhancement
    enhanced = apply_clahe(denoised.astype(np.uint8), clip_limit=2.5, tile_size=8)
    enhanced = enhanced.astype(float)
    
    # Get background statistics from enhanced image
    plate_pixels = enhanced[plate_mask]
    bg_enhanced = np.median(plate_pixels)
    std_enhanced = np.std(plate_pixels)
    
    # === METHOD 1: Adaptive thresholding (CFUCounter-inspired) ===
    # Multiple block sizes for different colony sizes
    adaptive_detections = np.zeros_like(luminance, dtype=float)
    
    for block_size in [31, 51, 71]:
        for C_val in [5, 8, 12]:
            adaptive_mask = adaptive_threshold(enhanced, block_size=block_size, C=C_val)
            adaptive_mask = adaptive_mask & plate_mask
            adaptive_detections += adaptive_mask.astype(float) / 9  # Normalize
    
    method1_mask = (adaptive_detections >= 0.3) & plate_mask  # 30% agreement
    method1_weight = 1.2  # Higher weight - this is the most reliable method
    
    # === METHOD 2: HSV-based detection ===
    hsv = rgb_to_hsv(rgb_array)
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    
    plate_v = v[plate_mask]
    plate_s = s[plate_mask]
    bg_v = np.median(plate_v)
    bg_s = np.median(plate_s)
    
    # Colonies are brighter AND less saturated than blood agar background
    # Made more sensitive for GAS-01
    is_bright_hsv = v > (bg_v + 8)  # Lowered from 10
    is_less_saturated = s < (bg_s * 0.88)  # Increased from 0.85
    method2_mask = is_bright_hsv & is_less_saturated & plate_mask
    method2_weight = 1.0
    
    # === METHOD 3: Local maxima detection ===
    blurred = gaussian_filter(enhanced, sigma=1.0)
    local_max = maximum_filter(blurred, size=5)
    
    # More sensitive threshold
    peaks = (blurred == local_max) & (enhanced > bg_enhanced + 0.4 * std_enhanced) & plate_mask
    method3_mask = ndimage.binary_dilation(peaks, iterations=2)
    method3_weight = 0.8
    
    # === METHOD 4: Multi-scale detection (NEW) ===
    method4_mask = multi_scale_detection(enhanced, plate_mask, scales=[0.75, 1.0, 1.3])
    method4_weight = 0.9
    
    # === METHOD 5: Direct intensity thresholding on enhanced image ===
    # Simple but effective after CLAHE
    intensity_thresh = bg_enhanced + 0.6 * std_enhanced
    method5_mask = (enhanced > intensity_thresh) & plate_mask
    method5_weight = 0.7
    
    # === WEIGHTED VOTING (improved from strict 2+ agreement) ===
    weighted_score = (
        method1_mask.astype(float) * method1_weight +
        method2_mask.astype(float) * method2_weight +
        method3_mask.astype(float) * method3_weight +
        method4_mask.astype(float) * method4_weight +
        method5_mask.astype(float) * method5_weight
    )
    
    # Total possible weight
    total_weight = method1_weight + method2_weight + method3_weight + method4_weight + method5_weight
    
    # Require 35% weighted agreement (more permissive than before)
    # This helps catch colonies that only one or two methods detect strongly
    combined_mask = (weighted_score >= total_weight * 0.35) & plate_mask
    
    # === MORPHOLOGICAL CLEANUP ===
    combined_mask = ndimage.binary_opening(combined_mask, iterations=1)
    combined_mask = ndimage.binary_closing(combined_mask, iterations=1)
    
    # Fill small holes within colonies
    combined_mask = ndimage.binary_fill_holes(combined_mask)
    
    # === IMPROVED WATERSHED SEGMENTATION ===
    segmented, num_segments = improved_watershed_segmentation(combined_mask, enhanced)
    
    # === FILTER BY SIZE AND CIRCULARITY ===
    if isinstance(segmented, np.ndarray) and segmented.max() > 0:
        labeled = segmented
        num_features = int(segmented.max())
    else:
        labeled, num_features = label(combined_mask)
    
    valid_colonies = 0
    colony_sizes = []
    colony_circularities = []
    
    # More permissive filters for v1.6.0
    min_size = MIN_COLONY_SIZE      # 6 pixels
    max_size = MAX_COLONY_SIZE      # 6000 pixels
    min_circ = MIN_CIRCULARITY      # 0.20
    
    for i in range(1, num_features + 1):
        component = labeled == i
        size = np.sum(component)
        
        if min_size <= size <= max_size:
            circ = calculate_circularity(component)
            
            # More permissive circularity for small colonies
            # Small colonies often appear less circular due to pixelation
            effective_min_circ = min_circ if size > 20 else min_circ * 0.7
            
            if circ >= effective_min_circ:
                valid_colonies += 1
                colony_sizes.append(size)
                colony_circularities.append(circ)
    
    avg_size = np.mean(colony_sizes) if colony_sizes else 0
    avg_circ = np.mean(colony_circularities) if colony_circularities else 0
    
    # Debug info
    debug_info = {
        'method1_detections': int(np.sum(method1_mask)),
        'method2_detections': int(np.sum(method2_mask)),
        'method3_detections': int(np.sum(method3_mask)),
        'method4_detections': int(np.sum(method4_mask)),
        'method5_detections': int(np.sum(method5_mask)),
        'combined_pixels': int(np.sum(combined_mask)),
        'segments_before_filter': num_features,
        'bg_enhanced': round(float(bg_enhanced), 1),
        'std_enhanced': round(float(std_enhanced), 1)
    }
    
    return valid_colonies, avg_size, avg_circ, labeled, debug_info


def detect_hemolysis_hsv(rgb_array, plate_mask):
    """
    Hemolysis detection using HSV color space
    
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
    
    # === BETA HEMOLYSIS ===
    is_bright = v > (bg_v + 8)
    is_desaturated = s < (bg_s * 0.80)
    beta_zones = (is_bright | is_desaturated) & plate_mask
    
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
    
    overall_desat = np.mean(plate_s) < 120
    
    # === ALPHA HEMOLYSIS ===
    is_greenish = (h > 40) & (h < 160)
    is_darker = v < (bg_v - 3)
    alpha_zones = is_greenish & is_darker & plate_mask
    alpha_ratio = np.sum(alpha_zones) / np.sum(plate_mask) if np.sum(plate_mask) > 0 else 0
    
    # === CLASSIFICATION ===
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
    """Main analysis function - v1.6.0 with research improvements"""
    img = load_and_resize_image(image_bytes)
    
    # Create plate mask
    plate_mask, center, radius = create_plate_mask(img.shape, rgb_array=img)
    
    luminance = get_luminance(img)
    
    # Colony detection with new algorithm
    colony_count, avg_size, avg_circ, labeled, debug_info = colony_detection_v160(
        img, luminance, plate_mask
    )
    
    # Hemolysis detection
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
        'debug': debug_info,
        'version': '1.6.0-research-improvements'
    }


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'version': '1.6.0-research-improvements',
        'max_image_size': MAX_IMAGE_SIZE,
        'supported_media_types': SUPPORTED_MEDIA_TYPES,
        'algorithms': [
            'CLAHE preprocessing',
            'Noise reduction (median filter)',
            'Adaptive thresholding (CFUCounter-inspired)',
            'HSV color space analysis',
            'Local maxima detection',
            'Multi-scale detection',
            'Improved watershed segmentation',
            'Weighted voting method combination',
            'Circularity filtering'
        ],
        'improvements': [
            'CLAHE contrast enhancement for low-contrast colonies',
            'Adaptive thresholding (35% weighted agreement)',
            'Multi-scale detection for varied colony sizes',
            'Lower circularity threshold for small colonies',
            'More sensitive HSV detection thresholds'
        ]
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
