"""
Colony Counting and Hemolysis Detection API
Calibrated version - improved sensitivity
Version: 1.3.0
"""

import os
import traceback
import base64
from io import BytesIO
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image

app = Flask(__name__)

# Configuration - CALIBRATED
MAX_IMAGE_SIZE = 700  # Increased for better small colony detection
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
    """Enhanced colony detection with multiple methods"""
    from scipy import ndimage
    from scipy.ndimage import gaussian_filter, maximum_filter
    
    # Get plate region statistics
    plate_pixels = luminance[plate_mask]
    background = np.median(plate_pixels)
    std_dev = np.std(plate_pixels)
    
    # Method 1: Luminance-based (bright colonies on dark background)
    # CALIBRATED: Lower threshold for better sensitivity
    bright_threshold = background + (1.2 * std_dev)
    bright_colonies = (luminance > bright_threshold) & plate_mask
    
    # Method 2: Detect whitish colonies specifically
    r, g, b = rgb_array[:,:,0], rgb_array[:,:,1], rgb_array[:,:,2]
    
    # White/light colonies: high overall brightness
    brightness = (r.astype(float) + g.astype(float) + b.astype(float)) / 3
    is_bright = brightness > (np.median(brightness[plate_mask]) + 20)
    
    # Not too red (to distinguish from blood agar background)
    red_ratio = r.astype(float) / (brightness + 1)
    not_too_red = red_ratio < 1.3
    
    white_colonies = is_bright & not_too_red & plate_mask
    
    # Method 3: Local contrast detection - find local maxima
    blurred = gaussian_filter(luminance.astype(float), sigma=2)
    local_max = maximum_filter(blurred, size=7)
    peaks = (blurred == local_max) & (luminance > background + std_dev * 0.8) & plate_mask
    
    # Dilate peaks to get colony regions
    dilated_peaks = ndimage.binary_dilation(peaks, iterations=2)
    
    # Combine all methods
    combined_mask = bright_colonies | white_colonies | dilated_peaks
    
    # Clean up with morphological operations
    combined_mask = ndimage.binary_opening(combined_mask, iterations=1)
    
    # Label connected components
    labeled, num_features = ndimage.label(combined_mask)
    
    # Filter by size - CALIBRATED for small colonies
    valid_colonies = 0
    colony_sizes = []
    min_size = 4   # Catch smaller colonies
    max_size = 3000
    
    for i in range(1, num_features + 1):
        size = np.sum(labeled == i)
        if min_size <= size <= max_size:
            valid_colonies += 1
            colony_sizes.append(size)
    
    avg_size = np.mean(colony_sizes) if colony_sizes else 0
    
    return valid_colonies, avg_size, labeled


def detect_hemolysis_calibrated(rgb_array, luminance, plate_mask):
    """Calibrated hemolysis detection - looks for clear zones"""
    
    r = rgb_array[:,:,0].astype(float)
    g = rgb_array[:,:,1].astype(float)
    b = rgb_array[:,:,2].astype(float)
    
    # Blood agar background is RED
    # Beta hemolysis = CLEAR zones (less red, more transparent/yellow)
    plate_r = r[plate_mask]
    bg_red = np.median(plate_r)
    bg_green = np.median(g[plate_mask])
    
    # BETA HEMOLYSIS: Clear/yellow zones - reduced red but still bright
    red_reduction = r < (bg_red * 0.85)
    brightness = (r + g + b) / 3
    still_bright = brightness > (np.median(brightness[plate_mask]) * 0.7)
    beta_zones = red_reduction & still_bright & plate_mask
    
    beta_ratio = np.sum(beta_zones) / np.sum(plate_mask)
    
    # ALPHA HEMOLYSIS: Greenish zones
    green_shift = (g > r * 0.9) & (g > bg_green * 1.1)
    alpha_zones = green_shift & plate_mask
    
    alpha_ratio = np.sum(alpha_zones) / np.sum(plate_mask)
    
    # Classification - CALIBRATED thresholds
    if beta_ratio > 0.08:
        hemo_type = 'beta'
        confidence = min(0.95, 0.5 + beta_ratio * 3)
    elif alpha_ratio > 0.05:
        hemo_type = 'alpha'
        confidence = min(0.90, 0.5 + alpha_ratio * 4)
    else:
        hemo_type = 'gamma'
        confidence = 0.6
    
    return {
        'type': hemo_type,
        'confidence': round(confidence, 4),
        'details': {
            'beta_zone_ratio': round(beta_ratio, 4),
            'alpha_zone_ratio': round(alpha_ratio, 4)
        }
    }


def create_plate_mask(shape):
    """Create circular plate mask"""
    h, w = shape[:2]
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 2 - 5
    
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = dist <= radius * 0.92
    
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
        'version': '1.3.0-calibrated'
    }


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'version': '1.3.0-calibrated',
        'max_image_size': MAX_IMAGE_SIZE,
        'supported_media_types': SUPPORTED_MEDIA_TYPES,
        'calibration': 'Tuned for blood agar plates with white colonies'
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
