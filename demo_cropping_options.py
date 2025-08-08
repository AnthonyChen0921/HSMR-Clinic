#!/usr/bin/env python3
"""
GAVD Cropping Options Demo
=========================

This script demonstrates the different cropping modes available in the GAVD processor:

1. Standard Mode: Uses bbox top-left as crop origin
2. Center Crop Mode: Centers crop around bbox center  
3. Top-Left as Center Mode: Treats bbox top-left as center point

Usage:
    python demo_cropping_options.py
"""

from GAVD_process_bbox import GAVDBBoxProcessor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_cropping_modes():
    """Visualize the different cropping modes with diagrams."""
    
    # Example bounding box and crop dimensions
    bbox = {'left': 200, 'top': 150, 'width': 100, 'height': 200}
    crop_width, crop_height = 150, 250  # Max dimensions
    
    # Calculate crop positions for each mode
    modes = {
        'Standard Mode\n(center_crop=False)': {
            'crop_left': bbox['left'],
            'crop_top': bbox['top'],
            'color': 'red',
            'description': 'Crop starts at bbox top-left'
        },
        'Center Crop Mode\n(center_crop=True)': {
            'crop_left': bbox['left'] + bbox['width']/2 - crop_width/2,
            'crop_top': bbox['top'] + bbox['height']/2 - crop_height/2,
            'color': 'blue',
            'description': 'Crop centered on bbox center'
        },
        'Top-Left as Center\n(topleft_as_center=True)': {
            'crop_left': bbox['left'] - crop_width/2,
            'crop_top': bbox['top'] - crop_height/2,
            'color': 'green',
            'description': 'Crop centered on bbox top-left'
        }
    }
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (mode_name, mode_info) in enumerate(modes.items()):
        ax = axes[i]
        
        # Draw original frame boundary
        frame_rect = patches.Rectangle((0, 0), 500, 400, 
                                     linewidth=2, edgecolor='black', 
                                     facecolor='lightgray', alpha=0.3)
        ax.add_patch(frame_rect)
        
        # Draw bounding box
        bbox_rect = patches.Rectangle((bbox['left'], bbox['top']), 
                                    bbox['width'], bbox['height'],
                                    linewidth=2, edgecolor='orange', 
                                    facecolor='yellow', alpha=0.5)
        ax.add_patch(bbox_rect)
        
        # Draw crop region
        crop_rect = patches.Rectangle((mode_info['crop_left'], mode_info['crop_top']), 
                                    crop_width, crop_height,
                                    linewidth=3, edgecolor=mode_info['color'], 
                                    facecolor='none', linestyle='--')
        ax.add_patch(crop_rect)
        
        # Mark key points
        # Bbox top-left
        ax.plot(bbox['left'], bbox['top'], 'o', color='orange', markersize=8, label='Bbox top-left')
        
        # Bbox center
        bbox_center_x = bbox['left'] + bbox['width']/2
        bbox_center_y = bbox['top'] + bbox['height']/2
        ax.plot(bbox_center_x, bbox_center_y, 's', color='orange', markersize=8, label='Bbox center')
        
        # Crop center
        crop_center_x = mode_info['crop_left'] + crop_width/2
        crop_center_y = mode_info['crop_top'] + crop_height/2
        ax.plot(crop_center_x, crop_center_y, '^', color=mode_info['color'], markersize=10, label='Crop center')
        
        # Set up the plot
        ax.set_xlim(-50, 450)
        ax.set_ylim(-50, 450)
        ax.set_aspect('equal')
        ax.invert_yaxis()  # Match image coordinate system
        ax.set_title(f"{mode_name}\n{mode_info['description']}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        # Add coordinate annotations
        ax.annotate(f'Crop: ({mode_info["crop_left"]:.0f}, {mode_info["crop_top"]:.0f})', 
                   xy=(mode_info['crop_left'], mode_info['crop_top']), 
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=mode_info['color'], alpha=0.3),
                   fontsize=8)
    
    plt.tight_layout()
    plt.suptitle('GAVD Cropping Mode Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.show()
    
    # Print numerical comparison
    print("📊 CROPPING MODE COMPARISON:")
    print("=" * 60)
    print(f"Original bbox: ({bbox['left']}, {bbox['top']}) - {bbox['width']}×{bbox['height']}")
    print(f"Crop dimensions: {crop_width}×{crop_height}")
    print()
    
    for mode_name, mode_info in modes.items():
        print(f"{mode_name.replace(chr(10), ' ')}:")
        print(f"  Crop position: ({mode_info['crop_left']:.0f}, {mode_info['crop_top']:.0f})")
        print(f"  Description: {mode_info['description']}")
        print()

def demo_processing_with_options():
    """Demo processing a sequence with different cropping options."""
    
    print("🚀 GAVD Cropping Options Demo")
    print("=" * 50)
    
    # Initialize processor
    processor = GAVDBBoxProcessor()
    
    # Find an available sequence
    sequence_ids = processor.df_all['seq'].dropna().unique()
    available_sequence = None
    
    for seq_id in sequence_ids[:10]:
        video_path = processor.sequences_dir / f"{seq_id}.mp4"
        if video_path.exists():
            available_sequence = seq_id
            break
    
    if not available_sequence:
        print("❌ No sequences with video files found")
        return
    
    print(f"🎯 Using sequence: {available_sequence}")
    
    # Get sequence info
    seq_info = processor.get_sequence_info(available_sequence)
    if not seq_info:
        print("❌ Could not get sequence info")
        return
    
    print(f"📊 Sequence info:")
    print(f"  Gait pattern: {seq_info['metadata']['gait_pat']}")
    print(f"  Max dimensions: {seq_info['max_width']:.0f}×{seq_info['max_height']:.0f}")
    print(f"  Frame range: {seq_info['frame_range'][0]} to {seq_info['frame_range'][1]}")
    
    # Demo different cropping modes
    cropping_modes = [
        {
            'name': 'Standard Mode',
            'params': {'use_max_dimensions': True, 'center_crop': False, 'topleft_as_center': False},
            'suffix': '_standard'
        },
        {
            'name': 'Center Crop Mode', 
            'params': {'use_max_dimensions': True, 'center_crop': True, 'topleft_as_center': False},
            'suffix': '_centered'
        },
        {
            'name': 'Top-Left as Center Mode',
            'params': {'use_max_dimensions': True, 'center_crop': False, 'topleft_as_center': True},
            'suffix': '_topleft_center'
        }
    ]
    
    print(f"\n🎬 Processing sequence with different cropping modes:")
    
    for mode in cropping_modes:
        print(f"\n📐 {mode['name']}:")
        
        # Temporarily modify the output path for this mode
        original_output_dir = processor.output_dir
        mode_output_dir = processor.output_dir / "cropping_modes"
        mode_output_dir.mkdir(exist_ok=True)
        processor.output_dir = mode_output_dir
        
        # Create custom output filename
        output_path = mode_output_dir / f"{available_sequence}{mode['suffix']}_cropped.mp4"
        
        # Check if already exists
        if output_path.exists():
            print(f"  ⏭️ Already exists: {output_path.name}")
        else:
            # Process with this mode's parameters
            success = processor.crop_sequence_video(available_sequence, **mode['params'])
            
            if success:
                # Rename to include mode suffix
                default_output = mode_output_dir / f"{available_sequence}_cropped.mp4"
                if default_output.exists():
                    default_output.rename(output_path)
                print(f"  ✅ Created: {output_path.name}")
            else:
                print(f"  ❌ Failed to process")
        
        # Restore original output directory
        processor.output_dir = original_output_dir
    
    print(f"\n📁 Check the cropping_modes folder for outputs with different cropping styles!")

def main():
    """Main demo function."""
    print("🎨 GAVD Cropping Options Visualization")
    print("-" * 40)
    
    # Show visualization
    try:
        visualize_cropping_modes()
    except Exception as e:
        print(f"⚠️ Could not create visualization: {e}")
        print("Install matplotlib to see the visual comparison")
    
    print("\n" + "="*60)
    
    # Demo processing
    try:
        demo_processing_with_options()
    except Exception as e:
        print(f"❌ Demo processing failed: {e}")

if __name__ == "__main__":
    main() 