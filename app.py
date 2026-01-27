"""
Colony Counting and Hemolysis Detection API - MEMORY OPTIMIZED
Optimized for Render free tier (512MB RAM)
Version: 1.1.0
"""

import os
import tempfile
import traceback
from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io as python_io

app = Flask(__name__)

# Configuration
MAX_IMAGE_SIZE = 600  # Reduced from 800 for memory safety
SUPPORTED_MEDIA_TYPES = ['blood_agar', 'nutrient_agar', 'macconkey_agar']

def resize_image_pil(image_bytes, max_size=MAX_IMAGE_SIZE):
    """Resize image using PIL (more memory efficient than skimage)"""
    img = Image.open(python_io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize if needed
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    
    return np.array(img)

def rgb_to_lab_simple(rgb_array):
    """Simple RGB to LAB conversion without heavy dependencies"""
    # Normalize RGB to 0-1
    rgb = rgb_array.astype(np.float32) / 255.0
    
    # Simplified LAB approximation
    # L = lightness (0-100)
    L = 0.2126 * rgb[:,:,0] + 0.7152 * rgb[:,:,1] + 0.0722 * rgb[:,:,2]
    L = L * 100
    
    # a = green-red (-128 to 127)
    a = (rgb[:,:,0] - rgb[:,:,1]) * 128
    
    # b = blue-yellow (-128 to 127)  
    b = (rgb[:,:,1] - rgb[:,:,2]) * 128
    
    return L, a, b

def detect_plate_region(L_channel):
    """Detect the circular plate region"""
    h, w = L_channel.shape
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 2 - 10
    
    # Create circular mask
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    plate_mask = dist_from_center <= radius * 0.95
    
    return plate_mask, (center_x, center_y), radius

def simple_colony_detection(L_channel, plate_mask):
    """Simple threshold-based colony detection (memory efficient)"""
    from scipy import ndimage
    
    # Work only within plate region
    masked_L = L_channel.copy()
    masked_L[~plate_mask] = 0
    
    # Calculate background (median of plate region)
    plate_values = L_channel[plate_mask]
    background = np.median(plate_values)
    
    # Colonies are typically darker OR lighter than background
    # For blood agar, colonies are often lighter (cleared zones)
    threshold_high = background + 15
    threshold_low = background - 20
    
    # Binary mask of potential colonies
    colony_mask = ((masked_L > threshold_high) | (masked_L < threshold_low)) & plate_mask
    
    # Label connected components
    labeled_array, num_features = ndimage.label(colony_mask)
    
    # Filter by size
    colonies = []
    min_area = 20
    max_area = 5000
    
    for i in range(1, num_features + 1):
        component_mask = labeled_array == i
        area = np.sum(component_mask)
        
        if min_area <= area <= max_area:
            # Get centroid
            coords = np.where(component_mask)
            cy, cx = np.mean(coords[0]), np.mean(coords[1])
            
            # Calculate circularity
            perimeter = np.sum(ndimage.binary_dilation(component_mask) & ~component_mask)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1)
            
            if circularity > 0.2:  # Reasonably circular
                colonies.append({
                    'center': (int(cx), int(cy)),
                    'area': int(area),
                    'circularity': float(circularity),
                    'diameter_px': float(np.sqrt(area / np.pi) * 2)
                })
    
    return colonies, labeled_array

def analyze_hemolysis(L_channel, a_channel, colonies, plate_mask, background_L, background_a):
    """Analyze hemolysis zones around colonies"""
    if not colonies:
        return {'type': 'none', 'confidence': 0, 'details': {}}
    
    beta_count = 0
    alpha_count = 0
    gamma_count = 0
    
    h, w = L_channel.shape
    
    for colony in colonies:
        cx, cy = colony['center']
        radius = int(colony['diameter_px'] / 2) + 5
        
        # Define annular region around colony
        inner_r = radius
        outer_r = radius + 15
        
        y_grid, x_grid = np.ogrid[max(0,cy-outer_r):min(h,cy+outer_r), 
                                   max(0,cx-outer_r):min(w,cx+outer_r)]
        
        # Adjust coordinates for local patch
        local_cy = cy - max(0, cy-outer_r)
        local_cx = cx - max(0, cx-outer_r)
        
        dist = np.sqrt((x_grid - local_cx)**2 + (y_grid - local_cy)**2)
        ring_mask = (dist >= inner_r) & (dist <= outer_r)
        
        if np.sum(ring_mask) < 10:
            gamma_count += 1
            continue
        
        # Get L* and a* values in ring
        L_patch = L_channel[max(0,cy-outer_r):min(h,cy+outer_r), 
                           max(0,cx-outer_r):min(w,cx+outer_r)]
        a_patch = a_channel[max(0,cy-outer_r):min(h,cy+outer_r),
                           max(0,cx-outer_r):min(w,cx+outer_r)]
        
        if L_patch.shape != ring_mask.shape:
            gamma_count += 1
            continue
            
        ring_L = np.mean(L_patch[ring_mask])
        ring_a = np.mean(a_patch[ring_mask])
        
        # Classification
        L_diff = ring_L - background_L
        a_diff = ring_a - background_a
        
        if L_diff > 8:  # Significantly lighter = beta
            beta_count += 1
        elif a_diff < -3:  # Green shift = alpha
            alpha_count += 1
        else:
            gamma_count += 1
    
    total = beta_count + alpha_count + gamma_count
    if total == 0:
        return {'type': 'none', 'confidence': 0, 'details': {}}
    
    # Determine dominant type
    if beta_count >= total * 0.5:
        hemo_type = 'beta'
        confidence = beta_count / total
    elif alpha_count >= total * 0.5:
        hemo_type = 'alpha'
        confidence = alpha_count / total
    elif gamma_count >= total * 0.6:
        hemo_type = 'gamma'
        confidence = gamma_count / total
    else:
        hemo_type = 'mixed'
        confidence = max(beta_count, alpha_count, gamma_count) / total
    
    return {
        'type': hemo_type,
        'confidence': round(confidence, 4),
        'details': {
            'beta_count': beta_count,
            'alpha_count': alpha_count,
            'gamma_count': gamma_count,
            'beta_percent': round(100 * beta_count / total, 2) if total > 0 else 0
        }
    }

def generate_decision_label(colony_count, hemolysis, artifacts, media_type):
    """Generate clinical decision support label"""
    
    # Growth classification
    if colony_count <= 3:
        growth = "No growth"
        confidence = "HIGH"
        requires_review = False
    elif colony_count <= 20:
        growth = f"Low colony count ({colony_count} CFU)"
        confidence = "MEDIUM"
        requires_review = True
    elif colony_count <= 50:
        growth = f"Moderate growth ({colony_count} CFU)"
        confidence = "HIGH"
        requires_review = False
    else:
        growth = f"Significant growth ({colony_count} CFU)"
        confidence = "HIGH"
        requires_review = False
    
    # Build label
    label_parts = [growth]
    
    # Add hemolysis info for blood agar
    if media_type == 'blood_agar' and hemolysis['type'] != 'none':
        hemo_type = hemolysis['type']
        if hemo_type == 'beta':
            label_parts.append("beta-hemolytic")
            label_parts.append("follow Strep workup protocol")
        elif hemo_type == 'alpha':
            label_parts.append("alpha-hemolytic")
            label_parts.append("rule out S. pneumoniae")
        elif hemo_type == 'gamma':
            label_parts.append("non-hemolytic")
        elif hemo_type == 'mixed':
            label_parts.append("mixed hemolysis")
            requires_review = True
    
    # Artifact warning
    if artifacts:
        requires_review = True
    
    decision_label = " – ".join(label_parts)
    
    return {
        'label': decision_label,
        'confidence': confidence,
        'requires_review': requires_review
    }

def analyze_plate(image_bytes, media_type='blood_agar'):
    """Main analysis function - memory optimized"""
    
    # Step 1: Load and resize image (PIL is more memory efficient)
    img_array = resize_image_pil(image_bytes)
    
    # Step 2: Convert to LAB color space (simplified)
    L_channel, a_channel, b_channel = rgb_to_lab_simple(img_array)
    
    # Step 3: Detect plate region
    plate_mask, center, radius = detect_plate_region(L_channel)
    
    # Step 4: Calculate background values
    background_L = np.median(L_channel[plate_mask])
    background_a = np.median(a_channel[plate_mask])
    
    # Step 5: Detect colonies
    from scipy import ndimage  # Import here to delay loading
    colonies, labeled = simple_colony_detection(L_channel, plate_mask)
    
    # Step 6: Analyze hemolysis (blood agar only)
    if media_type == 'blood_agar':
        hemolysis = analyze_hemolysis(L_channel, a_channel, colonies, 
                                       plate_mask, background_L, background_a)
    else:
        hemolysis = {'type': 'none', 'confidence': 0, 'details': {}}
    
    # Step 7: Detect artifacts (simplified)
    artifacts = []
    high_L_ratio = np.sum(L_channel[plate_mask] > 80) / np.sum(plate_mask)
    if high_L_ratio > 0.3:
        artifacts.append('possible_glare_or_reflection')
    
    # Step 8: Generate decision label
    colony_count = len(colonies)
    decision = generate_decision_label(colony_count, hemolysis, artifacts, media_type)
    
    # Step 9: Calculate statistics
    if colonies:
        areas = [c['area'] for c in colonies]
        circularities = [c['circularity'] for c in colonies]
        diameters = [c['diameter_px'] for c in colonies]
        stats = {
            'mean_area': round(np.mean(areas), 2),
            'mean_circularity': round(np.mean(circularities), 3),
            'mean_diameter_px': round(np.mean(diameters), 1)
        }
    else:
        stats = {'mean_area': 0, 'mean_circularity': 0, 'mean_diameter_px': 0}
    
    return {
        'status': 'success',
        'colony_count': colony_count,
        'decision_label': decision['label'],
        'decision_confidence': decision['confidence'],
        'requires_manual_review': decision['requires_review'],
        'hemolysis': hemolysis,
        'colony_statistics': stats,
        'artifacts_detected': artifacts,
        'plate_info': {
            'center': list(center),
            'radius_px': radius,
            'image_size': list(img_array.shape[:2])
        }
    }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.1.0-optimized',
        'description': 'Colony Counting and Hemolysis Detection API (Memory Optimized)',
        'endpoints': {
            '/health': 'GET - Health check',
            '/analyze': 'POST - Analyze plate image'
        },
        'supported_media_types': SUPPORTED_MEDIA_TYPES,
        'max_image_size': MAX_IMAGE_SIZE,
        'decision_thresholds': {
            'no_growth': '≤ 3 CFU',
            'low_count': '≤ 20 CFU',
            'significant': '> 50 CFU'
        }
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze plate image endpoint"""
    try:
        # Get media type
        media_type = request.form.get('media_type', 'blood_agar')
        if media_type not in SUPPORTED_MEDIA_TYPES:
            media_type = 'blood_agar'
        
        # Get image
        image_bytes = None
        
        # Check for file upload
        if 'image' in request.files:
            file = request.files['image']
            if file.filename:
                image_bytes = file.read()
        
        # Check for base64
        if image_bytes is None and request.is_json:
            data = request.get_json()
            if data and 'image_base64' in data:
                import base64
                image_bytes = base64.b64decode(data['image_base64'])
        
        if image_bytes is None:
            return jsonify({
                'status': 'error',
                'error': 'No image provided. Send as "image" file or "image_base64" in JSON.'
            }), 400
        
        # Analyze
        result = analyze_plate(image_bytes, media_type)
        return jsonify(result)
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
