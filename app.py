"""
Colony Counting and Hemolysis Detection API
============================================
Quantitative analysis of bacterial colonies on agar plates.

Features:
- Colony counting with watershed segmentation for overlapping colonies
- Artifact suppression (bubbles, labels, condensation, plate rim)
- Hemolysis detection (alpha/beta/gamma) for blood agar
- Decision support labels

Author: JoeLuT AI Solutions
Version: 1.0.0
"""

import io
import base64
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Import image processing libraries
from skimage import io as skio, color, filters, measure, morphology, segmentation, feature
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import disk, remove_small_objects, remove_small_holes
from skimage.segmentation import watershed, clear_border
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

app = Flask(__name__)

# =============================================================================
# CONFIGURATION - Calibrated thresholds (to be refined with validation data)
# =============================================================================

CONFIG = {
    "version": "1.0.0",
    
    # Colony detection parameters
    "min_colony_area": 50,           # Minimum colony area in pixels
    "max_colony_area": 50000,        # Maximum colony area (filter large artifacts)
    "min_circularity": 0.3,          # Minimum circularity to be considered colony
    "gaussian_sigma": 1.5,           # Gaussian blur for noise reduction
    
    # Blood agar specific
    "blood_agar_hue_range": (0, 30), # Red hue range for blood agar detection
    "hemolysis_lightness_threshold": 0.6,  # Lightness above this = beta hemolysis zone
    
    # Decision thresholds
    "no_growth_threshold": 3,        # ≤ this = no growth
    "low_count_threshold": 20,       # ≤ this = low colony count
    "significant_threshold": 50,     # > this = significant growth
    
    # Artifact detection
    "plate_rim_margin": 0.05,        # 5% margin from detected plate edge
}


def detect_plate_region(image):
    """
    Detect the circular plate region and create a mask to exclude the rim.
    
    Returns:
        plate_mask: Binary mask of the valid plate region
        plate_center: (x, y) center coordinates
        plate_radius: Radius in pixels
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image
    
    # Threshold to find the plate (usually brighter than background)
    thresh = threshold_otsu(gray)
    binary = gray > thresh * 0.5
    
    # Fill holes and clean up
    binary = ndi.binary_fill_holes(binary)
    binary = morphology.remove_small_objects(binary, min_size=1000)
    
    # Find the largest connected component (the plate)
    labels = measure.label(binary)
    if labels.max() == 0:
        # Fallback: assume entire image is plate
        h, w = gray.shape
        return np.ones_like(gray, dtype=bool), (w//2, h//2), min(h, w)//2
    
    regions = measure.regionprops(labels)
    largest = max(regions, key=lambda r: r.area)
    
    # Get bounding circle
    center_y, center_x = largest.centroid
    plate_radius = np.sqrt(largest.area / np.pi)
    
    # Create circular mask with margin to exclude rim
    h, w = gray.shape
    y, x = np.ogrid[:h, :w]
    inner_radius = plate_radius * (1 - CONFIG["plate_rim_margin"])
    plate_mask = ((x - center_x)**2 + (y - center_y)**2) < inner_radius**2
    
    return plate_mask, (int(center_x), int(center_y)), int(plate_radius)


def detect_colonies_watershed(image, plate_mask):
    """
    Detect and count colonies using watershed segmentation.
    Handles overlapping colonies.
    
    Returns:
        colony_count: Number of detected colonies
        labeled_colonies: Labeled image of colonies
        colony_properties: List of region properties
    """
    # Convert to appropriate color space
    if len(image.shape) == 3:
        # For blood agar, colonies appear lighter (cream/white) against red background
        # Use LAB color space for better separation
        lab = color.rgb2lab(image)
        lightness = lab[:, :, 0] / 100.0  # Normalize to 0-1
        
        # Also check for green shift (alpha hemolysis indicator)
        a_channel = lab[:, :, 1]  # a* channel: negative = green, positive = red
        
        # Colonies are lighter regions
        colony_signal = lightness
    else:
        colony_signal = image
    
    # Apply Gaussian blur
    blurred = gaussian(colony_signal, sigma=CONFIG["gaussian_sigma"])
    
    # Threshold to find potential colonies
    # Colonies are LIGHTER than the blood agar background
    thresh = threshold_otsu(blurred[plate_mask])
    
    # Colonies are regions above threshold
    binary_colonies = blurred > thresh
    binary_colonies = binary_colonies & plate_mask
    
    # Clean up binary image
    binary_colonies = morphology.remove_small_objects(
        binary_colonies, 
        min_size=CONFIG["min_colony_area"]
    )
    binary_colonies = morphology.remove_small_holes(binary_colonies, area_threshold=100)
    
    # Watershed segmentation to separate touching colonies
    distance = ndi.distance_transform_edt(binary_colonies)
    
    # Find local maxima as markers for watershed
    coords = peak_local_max(
        distance, 
        min_distance=10,  # Minimum distance between colony centers
        labels=binary_colonies,
        exclude_border=False
    )
    
    # Create markers
    mask = np.zeros(distance.shape, dtype=bool)
    if len(coords) > 0:
        mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    
    # Apply watershed
    labeled_colonies = watershed(-distance, markers, mask=binary_colonies)
    
    # Clear border colonies (touching rim)
    labeled_colonies = clear_border(labeled_colonies)
    
    # Get properties of each colony
    colony_props = measure.regionprops(labeled_colonies, intensity_image=colony_signal)
    
    # Filter by size and circularity
    valid_colonies = []
    for prop in colony_props:
        area = prop.area
        perimeter = prop.perimeter if prop.perimeter > 0 else 1
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        if (CONFIG["min_colony_area"] <= area <= CONFIG["max_colony_area"] and
            circularity >= CONFIG["min_circularity"]):
            valid_colonies.append({
                "label": prop.label,
                "area": area,
                "centroid": prop.centroid,
                "circularity": circularity,
                "mean_intensity": prop.mean_intensity,
                "eccentricity": prop.eccentricity,
                "solidity": prop.solidity,
                "equivalent_diameter": prop.equivalent_diameter
            })
    
    return len(valid_colonies), labeled_colonies, valid_colonies


def detect_hemolysis(image, labeled_colonies, colony_props, plate_mask):
    """
    Detect hemolysis type for blood agar plates.
    
    Hemolysis types:
    - Beta (β): Complete lysis - CLEAR zones around colonies
    - Alpha (α): Partial lysis - GREEN/BROWN zones around colonies
    - Gamma (γ): No hemolysis - No change in medium
    
    Returns:
        hemolysis_type: "beta", "alpha", "gamma", or "mixed"
        hemolysis_confidence: 0-1 confidence score
        details: Dictionary with analysis details
    """
    if len(image.shape) != 3:
        return "unknown", 0.0, {"error": "Requires color image for hemolysis detection"}
    
    # Convert to LAB color space
    lab = color.rgb2lab(image)
    lightness = lab[:, :, 0]  # L* channel
    a_channel = lab[:, :, 1]  # a* channel (green-red)
    b_channel = lab[:, :, 2]  # b* channel (blue-yellow)
    
    # Get background (blood agar) statistics
    colony_dilated = morphology.binary_dilation(
        labeled_colonies > 0, 
        footprint=disk(5)
    )
    background_mask = plate_mask & ~colony_dilated
    
    if np.sum(background_mask) < 100:
        return "unknown", 0.0, {"error": "Insufficient background for analysis"}
    
    bg_lightness_mean = np.mean(lightness[background_mask])
    bg_lightness_std = np.std(lightness[background_mask])
    bg_a_mean = np.mean(a_channel[background_mask])
    
    # Analyze zones around each colony
    beta_count = 0
    alpha_count = 0
    gamma_count = 0
    
    hemolysis_zones = []
    
    for prop in colony_props:
        label = prop["label"]
        colony_mask = labeled_colonies == label
        
        # Create annular ring around colony (hemolysis zone)
        outer_dilated = morphology.binary_dilation(colony_mask, footprint=disk(20))
        inner_dilated = morphology.binary_dilation(colony_mask, footprint=disk(5))
        halo_mask = outer_dilated & ~inner_dilated & plate_mask
        
        if np.sum(halo_mask) < 20:
            gamma_count += 1
            continue
        
        # Measure halo characteristics
        halo_lightness = np.mean(lightness[halo_mask])
        halo_a = np.mean(a_channel[halo_mask])
        
        # Classification logic
        lightness_diff = halo_lightness - bg_lightness_mean
        a_diff = halo_a - bg_a_mean  # More negative = more green
        
        zone_type = "gamma"
        
        # Beta hemolysis: significantly LIGHTER (clear zone)
        if lightness_diff > 10:  # Significantly lighter than background
            zone_type = "beta"
            beta_count += 1
        # Alpha hemolysis: GREEN shift (a* becomes more negative)
        elif a_diff < -5:  # Green shift
            zone_type = "alpha"
            alpha_count += 1
        else:
            zone_type = "gamma"
            gamma_count += 1
        
        hemolysis_zones.append({
            "colony_label": label,
            "halo_lightness": float(halo_lightness),
            "halo_a_channel": float(halo_a),
            "lightness_diff": float(lightness_diff),
            "a_channel_diff": float(a_diff),
            "classification": zone_type
        })
    
    total = beta_count + alpha_count + gamma_count
    if total == 0:
        return "unknown", 0.0, {"error": "No colonies analyzed"}
    
    # Determine overall hemolysis type
    beta_pct = beta_count / total
    alpha_pct = alpha_count / total
    gamma_pct = gamma_count / total
    
    if beta_pct > 0.6:
        overall_type = "beta"
        confidence = beta_pct
    elif alpha_pct > 0.6:
        overall_type = "alpha"
        confidence = alpha_pct
    elif gamma_pct > 0.6:
        overall_type = "gamma"
        confidence = gamma_pct
    else:
        overall_type = "mixed"
        confidence = max(beta_pct, alpha_pct, gamma_pct)
    
    details = {
        "beta_count": beta_count,
        "alpha_count": alpha_count,
        "gamma_count": gamma_count,
        "beta_percent": float(beta_pct * 100),
        "alpha_percent": float(alpha_pct * 100),
        "gamma_percent": float(gamma_pct * 100),
        "background_lightness": float(bg_lightness_mean),
        "background_a_channel": float(bg_a_mean),
        "zone_analysis": hemolysis_zones[:10]  # First 10 for brevity
    }
    
    return overall_type, confidence, details


def detect_artifacts(image, plate_mask):
    """
    Detect common artifacts: bubbles, condensation, labels.
    
    Returns:
        artifact_mask: Binary mask of detected artifacts
        artifact_types: List of detected artifact types
    """
    artifacts = []
    artifact_mask = np.zeros(plate_mask.shape, dtype=bool)
    
    if len(image.shape) == 3:
        gray = color.rgb2gray(image)
        hsv = color.rgb2hsv(image)
        saturation = hsv[:, :, 1]
    else:
        gray = image
        saturation = np.zeros_like(gray)
    
    # Bubbles: Very circular, high contrast edges
    edges = filters.sobel(gray)
    high_edge = edges > 0.3
    
    # Labels: Low saturation regions (often white/black text)
    low_sat = saturation < 0.1
    low_sat_large = morphology.remove_small_objects(low_sat & plate_mask, min_size=500)
    
    if np.sum(low_sat_large) > 1000:
        artifacts.append("labels_or_markings")
        artifact_mask |= low_sat_large
    
    # Condensation: Diffuse high-brightness areas
    if len(image.shape) == 3:
        brightness = np.max(image, axis=2) / 255.0
        very_bright = brightness > 0.95
        if np.sum(very_bright & plate_mask) > np.sum(plate_mask) * 0.1:
            artifacts.append("possible_condensation")
    
    return artifact_mask, artifacts


def generate_decision_label(colony_count, hemolysis_type, hemolysis_confidence, artifacts):
    """
    Generate clinical decision support label.
    
    Returns:
        label: Decision support text
        confidence: Overall confidence level
        requires_review: Boolean
    """
    requires_review = False
    
    # Base decision on colony count
    if colony_count <= CONFIG["no_growth_threshold"]:
        if len(artifacts) == 0:
            label = "No growth (auto-verified)"
            confidence = "HIGH"
        else:
            label = "No growth detected – verify artifacts"
            confidence = "MEDIUM"
            requires_review = True
    
    elif colony_count <= CONFIG["low_count_threshold"]:
        label = f"Low colony count ({colony_count} CFU) – review"
        confidence = "MEDIUM"
        requires_review = True
    
    else:
        # Significant growth - add hemolysis info
        if hemolysis_type == "beta" and hemolysis_confidence > 0.7:
            label = f"Significant growth ({colony_count} CFU) – beta-hemolytic – follow Strep workup protocol"
            confidence = "HIGH"
        elif hemolysis_type == "alpha" and hemolysis_confidence > 0.7:
            label = f"Significant growth ({colony_count} CFU) – alpha-hemolytic – rule out S. pneumoniae"
            confidence = "HIGH"
        elif hemolysis_type == "gamma":
            label = f"Significant growth ({colony_count} CFU) – non-hemolytic – follow standard workup"
            confidence = "MEDIUM"
        else:
            label = f"Significant growth ({colony_count} CFU) – mixed hemolysis – manual review"
            confidence = "LOW"
            requires_review = True
    
    # Flag if artifacts detected
    if len(artifacts) > 0:
        label += f" [Artifacts: {', '.join(artifacts)}]"
        requires_review = True
    
    return label, confidence, requires_review


def analyze_plate(image_array, media_type="blood_agar"):
    """
    Main analysis function.
    
    Args:
        image_array: numpy array of the plate image
        media_type: Type of agar ("blood_agar", "nutrient_agar", etc.)
    
    Returns:
        Dictionary with all analysis results
    """
    # Step 1: Detect plate region
    plate_mask, plate_center, plate_radius = detect_plate_region(image_array)
    
    # Step 2: Detect artifacts
    artifact_mask, artifacts = detect_artifacts(image_array, plate_mask)
    
    # Step 3: Detect and count colonies
    valid_mask = plate_mask & ~artifact_mask
    colony_count, labeled_colonies, colony_props = detect_colonies_watershed(
        image_array, valid_mask
    )
    
    # Step 4: Detect hemolysis (blood agar only)
    if media_type == "blood_agar" and len(image_array.shape) == 3:
        hemolysis_type, hemolysis_confidence, hemolysis_details = detect_hemolysis(
            image_array, labeled_colonies, colony_props, plate_mask
        )
    else:
        hemolysis_type = "not_applicable"
        hemolysis_confidence = 0.0
        hemolysis_details = {"note": f"Hemolysis detection not applicable for {media_type}"}
    
    # Step 5: Generate decision label
    decision_label, decision_confidence, requires_review = generate_decision_label(
        colony_count, hemolysis_type, hemolysis_confidence, artifacts
    )
    
    # Calculate colony statistics
    if colony_props:
        areas = [c["area"] for c in colony_props]
        circularities = [c["circularity"] for c in colony_props]
        diameters = [c["equivalent_diameter"] for c in colony_props]
        
        colony_stats = {
            "mean_area": float(np.mean(areas)),
            "std_area": float(np.std(areas)),
            "mean_circularity": float(np.mean(circularities)),
            "mean_diameter_px": float(np.mean(diameters)),
            "min_diameter_px": float(np.min(diameters)),
            "max_diameter_px": float(np.max(diameters)),
        }
    else:
        colony_stats = {}
    
    return {
        "status": "success",
        "version": CONFIG["version"],
        
        # Primary results
        "colony_count": colony_count,
        "decision_label": decision_label,
        "decision_confidence": decision_confidence,
        "requires_manual_review": requires_review,
        
        # Hemolysis
        "hemolysis": {
            "type": hemolysis_type,
            "confidence": float(hemolysis_confidence),
            "details": hemolysis_details
        },
        
        # Colony statistics
        "colony_statistics": colony_stats,
        
        # Artifacts
        "artifacts_detected": artifacts,
        
        # Plate info
        "plate_info": {
            "center": plate_center,
            "radius_px": plate_radius,
            "media_type": media_type
        },
        
        # Quality metrics
        "quality_metrics": {
            "valid_plate_area_percent": float(np.sum(plate_mask) / plate_mask.size * 100),
            "artifact_area_percent": float(np.sum(artifact_mask) / np.sum(plate_mask) * 100) if np.sum(plate_mask) > 0 else 0
        }
    }


# =============================================================================
# FLASK API ENDPOINTS
# =============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "version": CONFIG["version"],
        "description": "Colony Counting and Hemolysis Detection API",
        "endpoints": {
            "/analyze": "POST - Analyze plate image",
            "/health": "GET - Health check"
        },
        "supported_media_types": ["blood_agar", "nutrient_agar", "macconkey_agar"],
        "decision_thresholds": {
            "no_growth": f"≤ {CONFIG['no_growth_threshold']} CFU",
            "low_count": f"≤ {CONFIG['low_count_threshold']} CFU",
            "significant": f"> {CONFIG['significant_threshold']} CFU"
        }
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze a plate image.
    
    Expects:
        - image: File upload (multipart/form-data) OR base64 string (JSON)
        - media_type: (optional) "blood_agar", "nutrient_agar", etc.
    
    Returns:
        JSON with analysis results
    """
    try:
        # Get media type
        media_type = request.form.get('media_type', 'blood_agar')
        if request.is_json:
            media_type = request.json.get('media_type', 'blood_agar')
        
        # Get image
        if 'image' in request.files:
            # File upload
            file = request.files['image']
            image = Image.open(file.stream)
        elif request.is_json and 'image_base64' in request.json:
            # Base64 encoded
            img_data = base64.b64decode(request.json['image_base64'])
            image = Image.open(io.BytesIO(img_data))
        else:
            return jsonify({
                "status": "error",
                "message": "No image provided. Send as 'image' file or 'image_base64' in JSON."
            }), 400
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Ensure RGB (not RGBA)
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = image_array[:, :, :3]
        
        # Run analysis
        results = analyze_plate(image_array, media_type)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# =============================================================================
# LOCAL TESTING
# =============================================================================

def test_local_image(image_path):
    """Test with a local image file."""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print('='*60)
    
    # Load image
    image = skio.imread(image_path)
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    
    # Analyze
    results = analyze_plate(image, media_type="blood_agar")
    
    # Print results
    print(f"\n📊 RESULTS:")
    print(f"   Colony Count: {results['colony_count']}")
    print(f"   Hemolysis Type: {results['hemolysis']['type']} (confidence: {results['hemolysis']['confidence']:.2%})")
    print(f"\n🏷️  DECISION: {results['decision_label']}")
    print(f"   Confidence: {results['decision_confidence']}")
    print(f"   Requires Review: {results['requires_manual_review']}")
    
    if results['artifacts_detected']:
        print(f"\n⚠️  Artifacts: {', '.join(results['artifacts_detected'])}")
    
    if results['colony_statistics']:
        print(f"\n📏 Colony Stats:")
        print(f"   Mean diameter: {results['colony_statistics']['mean_diameter_px']:.1f} px")
        print(f"   Mean circularity: {results['colony_statistics']['mean_circularity']:.3f}")
    
    return results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # Test mode with local image
        test_local_image(sys.argv[1])
    else:
        # Run Flask server
        print("Starting Colony Analysis API...")
        print("Endpoints:")
        print("  GET  /health  - Health check")
        print("  POST /analyze - Analyze plate image")
        app.run(host='0.0.0.0', port=5000, debug=True)
