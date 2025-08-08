#!/usr/bin/env python3
"""
GAVD Sequence Processing Script
==============================

This script processes GAVD video sequences by cropping them using dynamic bounding box information.
It finds the maximum width and height across all frames and creates consistently sized crops.

Usage:
    python process_gavd_sequences.py
    
Or import and use programmatically:
    from GAVD_process_bbox import GAVDBBoxProcessor
    processor = GAVDBBoxProcessor()
    processor.process_all_sequences()
"""

from GAVD_process_bbox import GAVDBBoxProcessor
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Process GAVD sequences with dynamic bounding box cropping')
    parser.add_argument('--max_sequences', type=int, default=None, 
                       help='Maximum number of sequences to process (default: all)')
    parser.add_argument('--data_dir', type=str, default='GAVD/data/',
                       help='Directory containing GAVD annotation CSV files')
    parser.add_argument('--sequences_dir', type=str, default='GAVD-sequences/',
                       help='Directory containing sequence video files') 
    parser.add_argument('--output_dir', type=str, default='GAVD-cropped-sequences/',
                       help='Directory to save cropped videos')
    parser.add_argument('--no_center_crop', action='store_true',
                       help='Disable center cropping (use top-left alignment)')
    parser.add_argument('--topleft_as_center', action='store_true',
                       help='Treat bbox top-left as center point for cropping')
    parser.add_argument('--no_max_dimensions', action='store_true',
                       help='Use variable crop size instead of max dimensions')
    parser.add_argument('--force_reprocess', action='store_true',
                       help='Reprocess existing videos')
    parser.add_argument('--create_bbox_overlays', action='store_true',
                       help='Create videos with green bounding box overlays')
    parser.add_argument('--bbox_only', action='store_true',
                       help='Only create bbox overlays, skip cropping')
    
    args = parser.parse_args()
    
    print("🚀 GAVD Bounding Box Video Processor")
    print("=" * 50)
    
    try:
        # Initialize processor
        processor = GAVDBBoxProcessor(
            data_dir=args.data_dir,
            sequences_dir=args.sequences_dir,
            output_dir=args.output_dir
        )
        
        # Get and display statistics
        stats = processor.get_processing_stats()
        print(f"\n📊 DATASET STATISTICS:")
        print(f"Total annotations: {stats['total_annotations']:,}")
        print(f"Unique sequences: {stats['unique_sequences']:,}")
        print(f"Available videos: {stats['available_videos']:,}")
        print(f"Already processed: {stats['processed_videos']:,}")
        
        print(f"\n🏃 Gait patterns:")
        for pattern, count in stats['gait_patterns'].items():
            print(f"  {pattern}: {count:,}")
        
        print(f"\n📹 Camera views:")
        for view, count in stats['camera_views'].items():
            print(f"  {view}: {count:,}")
        
        # Processing configuration
        print(f"\n⚙️ PROCESSING CONFIGURATION:")
        print(f"Max sequences: {args.max_sequences or 'All'}")
        print(f"Use max dimensions: {not args.no_max_dimensions}")
        print(f"Center crop: {not args.no_center_crop}")
        print(f"Top-left as center: {args.topleft_as_center}")
        print(f"Skip existing: {not args.force_reprocess}")
        print(f"Create bbox overlays: {args.create_bbox_overlays or args.bbox_only}")
        print(f"Bbox overlays only: {args.bbox_only}")
        
        if stats['available_videos'] == 0:
            print("\n❌ No video files found! Please check your sequences directory.")
            return 1
        
        # Confirm processing
        if args.max_sequences is None and stats['available_videos'] > 10:
            response = input(f"\n⚠️  This will process {stats['available_videos']} videos. Continue? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Processing cancelled.")
                return 0
        
        # Start processing
        if not args.bbox_only:
            print(f"\n🎬 Starting video cropping...")
            processor.process_all_sequences(
                max_sequences=args.max_sequences,
                skip_existing=not args.force_reprocess,
                use_max_dimensions=not args.no_max_dimensions,
                center_crop=not args.no_center_crop,
                topleft_as_center=args.topleft_as_center
            )
        
        # Create bbox overlays if requested
        if args.create_bbox_overlays or args.bbox_only:
            print(f"\n📹 Creating bounding box overlay videos...")
            processor.create_multiple_bbox_overlays(
                num_videos=args.max_sequences or 5,
                skip_existing=not args.force_reprocess
            )
        
        print(f"\n🎉 Processing complete!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

def demo_single_sequence():
    """Demo function showing how to process a single sequence."""
    print("🔬 DEMO: Processing single sequence")
    print("-" * 40)
    
    # Initialize processor
    processor = GAVDBBoxProcessor()
    
    # Get first available sequence
    sequence_ids = processor.df_all['seq'].dropna().unique()
    
    for seq_id in sequence_ids[:5]:  # Try first 5 sequences
        video_path = processor.sequences_dir / f"{seq_id}.mp4"
        if video_path.exists():
            print(f"\n🎯 Processing sequence: {seq_id}")
            
            # Get sequence info first
            seq_info = processor.get_sequence_info(seq_id)
            if seq_info:
                print(f"📊 Gait pattern: {seq_info['metadata']['gait_pat']}")
                print(f"📐 Max dimensions: {seq_info['max_width']:.0f}x{seq_info['max_height']:.0f}")
                print(f"🎯 Frame range: {seq_info['frame_range'][0]} to {seq_info['frame_range'][1]}")
                print(f"📹 Total frames: {seq_info['total_frames']}")
                
                # Process the sequence (demo with top-left as center)
                success = processor.crop_sequence_video(
                    seq_id, 
                    use_max_dimensions=True,
                    center_crop=False,
                    topleft_as_center=True
                )
                if success:
                    print(f"✅ Successfully processed {seq_id} (using top-left as center)")
                    break
                else:
                    print(f"❌ Failed to process {seq_id}")
            else:
                print(f"❌ No sequence info for {seq_id}")
    
    print(f"\n📁 Check output directory: {processor.output_dir}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_single_sequence()
    else:
        sys.exit(main()) 