#!/usr/bin/env python3
"""
Enhanced script to run zucchini seed analysis with brightness calculation,
seed numbering, and comprehensive visualizations.
Now runs analysis separately for every image in the folder.
"""

from seed_analyzer import SeedAnalyzer
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # Get all image files in the current folder
    image_files = [f for f in os.listdir(".") if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print("No images found in this folder!")
        return

    for image_path in image_files:
        print(f"\nProcessing {image_path}...")

        try:
            analyzer = SeedAnalyzer(
                image_path,
                min_area_pixels=1,
                max_area_pixels=999999,
                min_circularity=0.0,
                max_circularity=1.0,
                min_aspect_ratio=0.1,
                max_aspect_ratio=10.0,
                min_brightness=0,
                max_brightness=255
            )

            # Run the complete analysis
            seeds_data = analyzer.run_full_analysis()

            # Add brightness/whiteness calculation
            for seed in seeds_data:
                seed['brightness'] = np.mean(seed['mean_color'])
                seed['whiteness'] = 255 - np.std(seed['mean_color'])

            print(f"Analysis complete for {image_path}")
            print(f"Total seeds found: {len(seeds_data)}")

            if seeds_data:
                # Summary stats
                areas = [seed['area_cm2'] for seed in seeds_data]
                brightnesses = [seed['brightness'] for seed in seeds_data]
                whitenesses = [seed['whiteness'] for seed in seeds_data]

                print(f"Average seed area: {sum(areas)/len(areas):.4f} cm²")
                print(f"Largest seed: {max(areas):.4f} cm²")
                print(f"Smallest seed: {min(areas):.4f} cm²")
                print(f"Average brightness: {sum(brightnesses)/len(brightnesses):.1f}")
                print(f"Average whiteness: {sum(whitenesses)/len(whitenesses):.1f}")

                # Save CSV per image
                df = pd.DataFrame(seeds_data)
                csv_filename = f"{os.path.splitext(image_path)[0]}_results.csv"
                df.to_csv(csv_filename, index=False)
                print(f"CSV file created: {csv_filename}")

                # Save visualization per image
                create_enhanced_visualizations(analyzer, seeds_data, image_path)

        except Exception as e:
            print(f"Error analyzing {image_path}: {e}")
            print("Make sure your image contains:")
            print("- Zucchini seeds on a black background")
            print("- A yellow 1x1 cm square for scale reference")
            print("- Good lighting and contrast")

def create_enhanced_visualizations(analyzer, seeds_data, image_path):
    """Create enhanced visualizations with seed numbering and parameter distributions."""

    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.3)

    # Original image with detected square
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(analyzer.image_with_square)
    ax1.set_title('Original Image with Detected Yellow Square', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Seed mask
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(analyzer.seed_mask, cmap='gray')
    ax2.set_title('Seed Segmentation Mask', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # Labeled seeds with numbers
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(analyzer.seed_labels, cmap='tab20')
    ax3.set_title(f'Individual Seeds (Total: {len(seeds_data)})', fontsize=12, fontweight='bold')
    ax3.axis('off')
    for seed in seeds_data:
        centroid = seed['centroid']
        ax3.text(centroid[1], centroid[0], str(seed['id']),
                 color='red', fontsize=8, fontweight='bold',
                 ha='center', va='center')

    # Area distribution
    ax4 = fig.add_subplot(gs[1, 0])
    areas = [seed['area_cm2'] for seed in seeds_data]
    ax4.hist(areas, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax4.set_xlabel('Area (cm²)', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Seed Area Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Perimeter distribution
    ax5 = fig.add_subplot(gs[1, 1])
    perimeters = [seed['perimeter_cm'] for seed in seeds_data]
    ax5.hist(perimeters, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax5.set_xlabel('Perimeter (cm)', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title('Seed Perimeter Distribution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Circularity distribution
    ax6 = fig.add_subplot(gs[1, 2])
    circularities = [seed['circularity'] for seed in seeds_data]
    ax6.hist(circularities, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax6.set_xlabel('Circularity', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.set_title('Seed Circularity Distribution', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Aspect ratio distribution
    ax7 = fig.add_subplot(gs[2, 0])
    aspect_ratios = [seed['aspect_ratio'] for seed in seeds_data]
    ax7.hist(aspect_ratios, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax7.set_xlabel('Aspect Ratio', fontsize=10)
    ax7.set_ylabel('Frequency', fontsize=10)
    ax7.set_title('Seed Aspect Ratio Distribution', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)

    # Brightness distribution
    ax8 = fig.add_subplot(gs[2, 1])
    brightnesses = [seed['brightness'] for seed in seeds_data]
    ax8.hist(brightnesses, bins=20, alpha=0.7, color='yellow', edgecolor='black')
    ax8.set_xlabel('Brightness', fontsize=10)
    ax8.set_ylabel('Frequency', fontsize=10)
    ax8.set_title('Seed Brightness Distribution', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)

    # Whiteness distribution
    ax9 = fig.add_subplot(gs[2, 2])
    whitenesses = [seed['whiteness'] for seed in seeds_data]
    ax9.hist(whitenesses, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
    ax9.set_xlabel('Whiteness', fontsize=10)
    ax9.set_ylabel('Frequency', fontsize=10)
    ax9.set_title('Seed Whiteness Distribution', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3)

    # Save figure per image
    out_name = f"{os.path.splitext(image_path)[0]}_enhanced.png"
    plt.savefig(out_name, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close(fig)

    print(f"Enhanced visualization saved as {out_name}")

if __name__ == "__main__":
    main()
