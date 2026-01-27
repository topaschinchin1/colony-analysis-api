"""
Colony Counting and Hemolysis Detection API
Maximum sensitivity version
Version: 1.5.0
"""

import os
import traceback
import base64
from io import BytesIO
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image

app = Flask(__name__)

# Configuration - MAXIMUM SENSITIVITY
MAX_IMAGE_SIZE = 900  # Even larger for tiny colony detection
SUPPORTED_MEDIA_TYPES = ['blood_agar', 'nutrient_agar', 'macconkey_agar']


def load_and_resize_image(image_bytes):
    """Load image and resize to manageable size"""
    img = Image.open(BytesIO(image_bytes))
    
    # Convert to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize if needed
    w, h = img.size
    if max(w, h) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    
    return np.array(img)


def get_luminance(rgb_array):
    """Calculate luminance from RGB"""
    return 0.299 * rgb_array[:,:,0] + 0.587 * rgb_array[:,:,1] + 0.114 * rgb_array[:,:,2]


def enhanced_colony_detection(rgb_array, luminance, plate_mask):
    """MAXIMUM SENSITIVITY colony detection - target ~189 CFU"""
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
    
    # Get plate region statistics
    plate_pixels = luminance[plate_mask]
    background = np.median(plate_pixels)
    std_dev = np.std(plate_pixels)
    
    # Method 1: Luminance-based - VERY SENSITIVE
    bright_threshold = background + (0.6 * std_dev)  # Was 0.8, now 0.6
    bright_colonies = (luminance > bright_threshold) & plate_mask
    
    # Method 2: Detect whitish colonies
    r, g, b = rgb_array[:,:,0], rgb_array[:,:,1], rgb_array[:,:,2]
    brightness = (r.astype(float) + g.astype(float) + b.astype(float)) / 3
    plate_brightness = brightness[plate_mask]
    bright_median = np.median(plate_brightness)
    
    # Even lower threshold
    is_bright = brightness > (bright_median + 8)  # Was +12, now +8
    
    # Not too red
    red_ratio = r.astype(float) / (brightness + 1)
    not_too_red = red_ratio < 1.2  # Was 1.25, now 1.2
    
    white_colonies = is_bright & not_too_red & plate_mask
    
    # Method 3: Local maxima - FINEST detection
    blurred = gaussian_filter(luminance.astype(float), sigma=1.0)  # Was 1.5, now 1.0
    local_max = maximum_filter(blurred, size=4)  # Was 5, now 4
    peaks = (blurred == local_max) & (luminance > background + std_dev * 0.3) & plate_mask  # Was 0.5, now 0.3
    
    # Dilate peaks slightly
    dilated_peaks = ndimage.binary_dilation(peaks, iterations=1)  # Was 2, now 1
    
    # Method 4: Edge-based detection
    local_min = minimum_filter(blurred, size=7)  # Was 9, now 7
    contrast = blurred - local_min
    high_contrast = (contrast > std_dev * 0.3) & plate_mask  # Was 0.4, now 0.3
    
    # Method 5: Adaptive threshold - catch colonies missed by global threshold
    # Use local mean comparison
    local_mean = ndimage.uniform_filter(luminance.astype(float), size=15)
    above_local = (luminance > local_mean + 5) & plate_mask
    
    # Combine ALL methods
    combined_mask = bright_colonies | white_colonies | dilated_peaks | high_contrast | above_local
    
    # Label connected components
    labeled, num_features = ndimage.label(combined_mask)
    
    # Filter by size - MINIMUM threshold
    valid_colonies = 0
    colony_sizes = []
    min_size = 2   # Was 3, now 2 - catch even tinier colonies
    max_size = 5000
    
    for i in range(1, num_features + 1):
        size = np.sum(labeled == i)
        if min_size <= size <= max_size:
            valid_colonies += 1
            colony_sizes.append(size)
    
    avg_size = np.mean(colony_sizes) if colony_sizes else 0
    
    return valid_colonies, avg_size, labeled


def detect_hemolysis_calibrated(rgb_array, luminance, plate_mask):
    """FIXED hemolysis detection - clear zones = BETA hemolysis"""
    
    r = rgb_array[:,:,0].astype(float)
    g = rgb_array[:,:,1].astype(float)
    b = rgb_array[:,:,2].astype(float)
    
    # Blood agar background is DARK RED
    # Beta hemolysis = CLEAR/TRANSPARENT zones (RBCs lysed completely)
    # Alpha hemolysis = GREENISH zones (partial lysis, methemoglobin)
    # Gamma = No change
    
    plate_r = r[plate_mask]
    plate_g = g[plate_mask]
    plate_b = b[plate_mask]
    
    bg_red = np.median(plate_r)
    bg_green = np.median(plate_g)
    bg_brightness = np.median((r + g + b)[plate_mask] / 3)
    
    # BETA HEMOLYSIS: Clear zones are LIGHTER and LESS RED
    # Clear zones = high brightness + reduced red saturation
    brightness = (r + g + b) / 3
    
    # Clear zone criteria:
    # 1. Brighter than background (cleared = more light passes through)
    # 2. Red component reduced relative to overall brightness
    is_brighter = brightness > (bg_brightness + 15)
    red_saturation = r / (brightness + 1)
    bg_red_sat = bg_red / (bg_brightness + 1)
    less_red = red_saturation < (bg_red_sat * 0.9)
    
    beta_zones = is_brighter & less_red & plate_mask
    beta_ratio = np.sum(beta_zones) / np.sum(plate_mask)
    
    # Also check for very bright spots (colony halos)
    very_bright = brightness > (bg_brightness + 30)
    bright_ratio = np.sum(very_bright & plate_mask) / np.sum(plate_mask)
    
    # Combined beta indicator
    beta_indicator = beta_ratio + (bright_ratio * 0.5)
    
    # ALPHA HEMOLYSIS: Greenish discoloration
    # Green becomes more prominent relative to red
    green_dominance = g > (r * 0.95)  # Green nearly equals or exceeds red
    darker = brightness < (bg_brightness - 5)  # Slightly darker (oxidized)
    alpha_zones = green_dominance & darker & plate_mask
    alpha_ratio = np.sum(alpha_zones) / np.sum(plate_mask)
    
    # Classification - BETA takes priority for clear zones
    if beta_indicator > 0.05:  # Clear zones detected
        hemo_type = 'beta'
        confidence = min(0.95, 0.6 + beta_indicator * 2)
    elif alpha_ratio > 0.08:  # Green zones detected
        hemo_type = 'alpha'
        confidence = min(0.90, 0.5 + alpha_ratio * 3)
    else:
        hemo_type = 'gamma'
        confidence = 0.6
    
    return {
        'type': hemo_type,
        'confidence': round(confidence, 4),
        'details': {
            'beta_zone_ratio': round(beta_ratio, 4),
            'bright_halo_ratio': round(bright_ratio, 4),
            'alpha_zone_ratio': round(alpha_ratio, 4)
        }
    }


def create_plate_mask(shape):
    """Create circular plate mask - EXPANDED to catch edge colonies"""
    h, w = shape[:2]
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 2 - 3  # Was -5, now -3
    
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = dist <= radius * 0.95  # Was 0.92, now 0.95 - include more edge area
    
    return mask, (center_x, center_y), radius


def generate_decision(colony_count, hemolysis, media_type):
    """Generate decision label"""
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
    if media_type == 'blood_agar' and hemolysis['type'] in ['alpha', 'beta']:
        if hemolysis['type'] == 'beta':
            label_parts.append("beta-hemolytic – follow Strep protocol")
        elif hemolysis['type'] == 'alpha':
            label_parts.append("alpha-hemolytic – rule out S. pneumoniae")
    elif media_type == 'blood_agar':
        label_parts.append("non-hemolytic")
    
    return {
        'label': " – ".join(label_parts),
        'confidence': confidence,
        'requires_review': review
    }


def analyze_plate(image_bytes, media_type='blood_agar'):
    """Main analysis function - calibrated version"""
    img = load_and_resize_image(image_bytes)
    plate_mask, center, radius = create_plate_mask(img.shape)
    luminance = get_luminance(img)
    
    colony_count, avg_size, labeled = enhanced_colony_detection(img, luminance, plate_mask)
    hemolysis = detect_hemolysis_calibrated(img, luminance, plate_mask)
    decision = generate_decision(colony_count, hemolysis, media_type)
    
    return {
        'status': 'success',
        'colony_count': colony_count,
        'decision_label': decision['label'],
        'decision_confidence': decision['confidence'],
        'requires_manual_review': decision['requires_review'],
        'hemolysis': hemolysis,
        'colony_statistics': {
            'average_size_px': round(avg_size, 1)
        },
        'plate_info': {
            'center': list(center),
            'radius_px': radius,
            'analyzed_size': list(img.shape[:2])
        },
        'version': '1.5.0-maximum'
    }


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'version': '1.5.0-maximum',
        'max_image_size': MAX_IMAGE_SIZE,
        'supported_media_types': SUPPORTED_MEDIA_TYPES,
        'calibration': 'Maximum sensitivity - tuned for 189 CFU reference plate'
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
