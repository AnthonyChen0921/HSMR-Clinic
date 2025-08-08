#!/usr/bin/env python3
"""
Create GAVD Bounding Box Overlay Videos
======================================

This script creates 5 videos with green bounding box overlays on the original sequences.
The overlays show how the bounding boxes change over time and include frame information.

Usage:
    python create_bbox_overlays.py
"""

from GAVD_process_bbox import GAVDBBoxProcessor

def main():
    print("🎬 GAVD Bounding Box Overlay Creator")
    print("=" * 50)
    
    try:
        # Initialize processor
        processor = GAVDBBoxProcessor()
        
        # Get dataset statistics
        stats = processor.get_processing_stats()
        print(f"\n📊 DATASET INFO:")
        print(f"Available videos: {stats['available_videos']:,}")
        print(f"Gait patterns: {list(stats['gait_patterns'].keys())}")
        
        if stats['available_videos'] == 0:
            print("\n❌ No video files found! Please check your GAVD-sequences directory.")
            return 1
        
        # Create 5 bbox overlay videos
        print(f"\n🎯 Creating 5 bounding box overlay videos...")
        print("Features:")
        print("  ✅ Green bounding box outlines")
        print("  ✅ Corner markers for better visibility")
        print("  ✅ Frame information overlay")
        print("  ✅ Bbox dimensions and position")
        print("  ✅ Gait pattern information")
        print("  ✅ Max dimensions reference")
        
        # Start creation
        processor.create_multiple_bbox_overlays(num_videos=5, skip_existing=True)
        
        print(f"\n🎉 Bbox overlay creation complete!")
        print(f"📁 Check the 'GAVD-cropped-sequences/bbox_overlays/' directory for output videos")
        
        # List created files
        overlay_dir = processor.output_dir / "bbox_overlays"
        if overlay_dir.exists():
            overlay_files = list(overlay_dir.glob("*_bbox_overlay.mp4"))
            if overlay_files:
                print(f"\n📄 Created overlay videos:")
                for i, file_path in enumerate(overlay_files[:5], 1):
                    size_mb = file_path.stat().st_size / (1024*1024)
                    print(f"  {i}. {file_path.name} ({size_mb:.1f} MB)")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Creation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 