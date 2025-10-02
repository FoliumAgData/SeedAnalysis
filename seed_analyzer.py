import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology, filters
from skimage.segmentation import watershed

from scipy import ndimage
import argparse
import os

class SeedAnalyzer:
    def __init__(self, image_path, min_area_pixels=200, max_area_pixels=5000, 
                 min_circularity=0.3, max_circularity=0.95, 
                 min_aspect_ratio=1.0, max_aspect_ratio=4.0,
                 min_brightness=30, max_brightness=220):
        """
        Initialize the seed analyzer with an image path and filtering parameters.
        
        Args:
            image_path (str): Path to the image file
            min_area_pixels (int): Minimum area in pixels for seed detection
            max_area_pixels (int): Maximum area in pixels for seed detection
            min_circularity (float): Minimum circularity (0-1)
            max_circularity (float): Maximum circularity (0-1)
            min_aspect_ratio (float): Minimum aspect ratio
            max_aspect_ratio (float): Maximum aspect ratio
            min_brightness (int): Minimum brightness (0-255)
            max_brightness (int): Maximum brightness (0-255)
        """
        self.image_path = image_path
        self.image = None
        self.scale_factor = None  # pixels per cm
        self.seeds_data = []
        
        # Filtering parameters
        self.min_area_pixels = min_area_pixels
        self.max_area_pixels = max_area_pixels
        self.min_circularity = min_circularity
        self.max_circularity = max_circularity
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        
    def load_image(self):
        """Load and preprocess the image."""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")
            
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {self.image_path}")
            
        # Convert BGR to RGB for matplotlib
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        print(f"Image loaded: {self.image.shape[1]}x{self.image.shape[0]} pixels")
        
    def detect_yellow_square(self):
        """
        Detect the yellow 1x1 cm square for scale calibration.
        Returns the scale factor (pixels per cm).
        """
        # Convert to HSV color space for better yellow detection
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Define yellow color range in HSV
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        
        # Create mask for yellow regions
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Find contours in the yellow mask
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No yellow square detected. Please ensure the 1x1 cm yellow square is visible.")
        
        # Find the largest yellow contour (should be the square)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Calculate scale factor (pixels per cm)
        # Since the square is 1x1 cm, the area should be 1 cm²
        self.scale_factor = np.sqrt(area)  # pixels per cm
        
        print(f"Yellow square detected. Scale factor: {self.scale_factor:.2f} pixels/cm")
        
        # Draw the detected square for verification
        self.image_with_square = self.image_rgb.copy()
        cv2.drawContours(self.image_with_square, [largest_contour], -1, (255, 0, 0), 2)
        
        return self.scale_factor
    
    def segment_seeds(self):
        """
        Simple seed segmentation - detect all objects and let filtering happen later.
        Returns a binary mask of the seeds.
        """
        print("Segmenting seeds...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Simple thresholding - much more reliable than adaptive
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove the yellow square from the binary image
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Remove yellow regions from seed mask
        binary = cv2.bitwise_and(binary, cv2.bitwise_not(yellow_mask))
        
        # Minimal morphological cleaning - just remove tiny noise
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Only remove extremely small objects (likely dust)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"Found {len(contours)} potential objects")
        
        clean_mask = np.zeros_like(binary)
        
        # Very low threshold - only remove obvious dust
        min_area = 10  # Extremely low - catch everything
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                cv2.drawContours(clean_mask, [contour], -1, 255, -1)
        
        self.seed_mask = clean_mask
        print(f"Seed segmentation complete - {np.sum(clean_mask > 0)} pixels in mask")
        return clean_mask
    
    def separate_touching_seeds(self):
        """
        Since seeds are already separated, just label them directly.
        Returns labeled image with individual seeds.
        """
        print("Seeds are already separated - labeling directly...")
        
        # Just label the connected components directly - no watershed needed
        labels = measure.label(self.seed_mask)
        print(f"Found {labels.max()} individual seeds")
        
        self.seed_labels = labels
        return labels
    
    def analyze_seeds(self):
        """
        Analyze individual seed properties - seeds are already separated.
        Returns list of seed characteristics.
        """
        if not hasattr(self, 'seed_labels'):
            raise ValueError("Must run separate_touching_seeds() first")
        
        print("Analyzing individual seeds...")
        
        # Get properties of each seed region
        props = measure.regionprops(self.seed_labels)
        print(f"Analyzing {len(props)} regions...")
        
        self.seeds_data = []
        seed_id = 1
        
        for i, prop in enumerate(props):
            if i % 20 == 0:  # Progress indicator
                print(f"Analyzing region {i+1}/{len(props)}")
                
            # Filter out very small objects based on actual area in cm²
            area_pixels = prop.area
            area_cm2 = area_pixels / (self.scale_factor ** 2)
            if area_cm2 < 0.01:  # Discard objects smaller than 0.01 cm²
                continue
            
            # Convert pixel measurements to cm using scale factor
            perimeter_cm = prop.perimeter / self.scale_factor
            
            # Calculate shape characteristics
            circularity = (4 * np.pi * area_pixels) / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
            aspect_ratio = prop.major_axis_length / prop.minor_axis_length if prop.minor_axis_length > 0 else 1
            
            # Get color information
            mask = self.seed_labels == prop.label
            mean_color = cv2.mean(self.image_rgb, mask=mask.astype(np.uint8))
            
            seed_data = {
                'id': seed_id,
                'area_cm2': area_cm2,
                'perimeter_cm': perimeter_cm,
                'circularity': circularity,
                'aspect_ratio': aspect_ratio,
                'major_axis_cm': prop.major_axis_length / self.scale_factor,
                'minor_axis_cm': prop.minor_axis_length / self.scale_factor,
                'centroid': prop.centroid,
                'mean_color': mean_color[:3],  # RGB values
                'bbox': prop.bbox
            }
            
            self.seeds_data.append(seed_data)
            seed_id += 1
        
        print(f"Seed analysis complete - {len(self.seeds_data)} unique seeds")
        return self.seeds_data
    
    def generate_report(self):
        """Generate a comprehensive analysis report."""
        if not self.seeds_data:
            raise ValueError("No seed data available. Run analyze_seeds() first.")
        
        print("\n" + "="*50)
        print("ZUCCHINI SEED ANALYSIS REPORT")
        print("="*50)
        
        print(f"\nTotal seeds detected: {len(self.seeds_data)}")
        
        # Calculate statistics
        areas = [seed['area_cm2'] for seed in self.seeds_data]
        perimeters = [seed['perimeter_cm'] for seed in self.seeds_data]
        circularities = [seed['circularity'] for seed in self.seeds_data]
        aspect_ratios = [seed['aspect_ratio'] for seed in self.seeds_data]
        
        print(f"\nAREA STATISTICS (cm²):")
        print(f"  Mean: {np.mean(areas):.4f}")
        print(f"  Std:  {np.std(areas):.4f}")
        print(f"  Min:  {np.min(areas):.4f}")
        print(f"  Max:  {np.max(areas):.4f}")
        
        print(f"\nPERIMETER STATISTICS (cm):")
        print(f"  Mean: {np.mean(perimeters):.4f}")
        print(f"  Std:  {np.std(perimeters):.4f}")
        print(f"  Min:  {np.min(perimeters):.4f}")
        print(f"  Max:  {np.max(perimeters):.4f}")
        
        print(f"\nSHAPE CHARACTERISTICS:")
        print(f"  Mean circularity: {np.mean(circularities):.4f}")
        print(f"  Mean aspect ratio: {np.mean(aspect_ratios):.4f}")
        
        # Color analysis
        colors = np.array([seed['mean_color'] for seed in self.seeds_data])
        print(f"\nCOLOR ANALYSIS (RGB):")
        print(f"  Mean R: {np.mean(colors[:, 0]):.1f}")
        print(f"  Mean G: {np.mean(colors[:, 1]):.1f}")
        print(f"  Mean B: {np.mean(colors[:, 2]):.1f}")
        
        return self.seeds_data
    
    def visualize_results(self):
        """Create visualizations of the analysis results."""
        if not hasattr(self, 'seeds_data'):
            raise ValueError("No seed data available. Run analyze_seeds() first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image with detected square
        axes[0, 0].imshow(self.image_with_square)
        axes[0, 0].set_title('Original Image with Detected Yellow Square')
        axes[0, 0].axis('off')
        
        # Seed mask
        axes[0, 1].imshow(self.seed_mask, cmap='gray')
        axes[0, 1].set_title('Seed Segmentation Mask')
        axes[0, 1].axis('off')
        
        # Labeled seeds
        axes[1, 0].imshow(self.seed_labels, cmap='tab20')
        axes[1, 0].set_title(f'Individual Seeds (Total: {len(self.seeds_data)})')
        axes[1, 0].axis('off')
        
        # Area distribution histogram
        areas = [seed['area_cm2'] for seed in self.seeds_data]
        axes[1, 1].hist(areas, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_xlabel('Area (cm²)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Seed Area Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('seed_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nVisualization saved as 'seed_analysis_results.png'")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline."""
        print("Starting zucchini seed analysis...")
        
        # Step 1: Load image
        self.load_image()
        
        # Step 2: Detect yellow square for scale
        self.detect_yellow_square()
        
        # Step 3: Segment seeds
        self.segment_seeds()
        
        # Step 4: Separate touching seeds
        self.separate_touching_seeds()
        
        # Step 5: Analyze individual seeds
        self.analyze_seeds()
        
        # Step 6: Generate report
        self.generate_report()
        
        # Note: Basic visualization removed - enhanced version is created in run_simple.py
        
        return self.seeds_data

def main():
    parser = argparse.ArgumentParser(description='Analyze zucchini seeds from an image')
    parser.add_argument('image_path', help='Path to the image file')
    args = parser.parse_args()
    
    try:
        analyzer = SeedAnalyzer(args.image_path)
        seeds_data = analyzer.run_full_analysis()
        
        # Save detailed results to CSV
        import pandas as pd
        df = pd.DataFrame(seeds_data)
        df.to_csv('seed_analysis_results.csv', index=False)
        print(f"\nDetailed results saved to 'seed_analysis_results.csv'")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
