import pandas as pd
import numpy as np
import cv2
import os
import ast
import json
from pathlib import Path
from collections import defaultdict
import subprocess
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class GAVDBBoxProcessor:
    def __init__(self, data_dir='GAVD/data/', sequences_dir='GAVD-sequences/', 
                 output_dir='GAVD-cropped-sequences/'):
        """
        Initialize the GAVD bounding box processor.
        
        Args:
            data_dir: Directory containing GAVD annotation CSV files
            sequences_dir: Directory containing sequence video files
            output_dir: Directory to save cropped videos
        """
        self.data_dir = Path(data_dir)
        self.sequences_dir = Path(sequences_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Load all annotation data
        self.df_all = self._load_all_annotations()
        
        print(f"✅ Loaded {len(self.df_all)} total annotations")
        print(f"📁 Sequences directory: {self.sequences_dir}")
        print(f"💾 Output directory: {self.output_dir}")
    
    def _load_all_annotations(self):
        """Load and combine all GAVD annotation CSV files."""
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        dfs = []
        for filename in sorted(csv_files):
            filepath = self.data_dir / filename
            print(f"📄 Loading {filename}...")
            df = pd.read_csv(filepath)
            dfs.append(df)
        
        # Combine all dataframes
        df_combined = pd.concat(dfs, ignore_index=True)
        
        # Parse bbox column from string to dict
        df_combined['bbox_parsed'] = df_combined['bbox'].apply(self._parse_bbox)
        
        return df_combined
    
    def _parse_bbox(self, bbox_str):
        """Parse bounding box string to dictionary."""
        if pd.isna(bbox_str):
            return None
        try:
            # Handle both string representations and actual dicts
            if isinstance(bbox_str, str):
                bbox_dict = ast.literal_eval(bbox_str)
            else:
                bbox_dict = bbox_str
            
            # Ensure all required keys exist and are numeric
            required_keys = ['top', 'left', 'height', 'width']
            if all(key in bbox_dict for key in required_keys):
                return {
                    'top': float(bbox_dict['top']),
                    'left': float(bbox_dict['left']),
                    'height': float(bbox_dict['height']),
                    'width': float(bbox_dict['width'])
                }
        except (ValueError, SyntaxError, KeyError) as e:
            pass
        return None
    
    def get_sequence_info(self, sequence_id):
        """
        Get bounding box information for a specific sequence.
        
        Returns:
            dict: Contains sequence data, max dimensions, and frame mapping
        """
        # Filter data for this sequence
        seq_data = self.df_all[self.df_all['seq'] == sequence_id].copy()
        
        if seq_data.empty:
            print(f"❌ No data found for sequence: {sequence_id}")
            return None
        
        # Filter out rows with invalid bounding boxes
        seq_data = seq_data[seq_data['bbox_parsed'].notna()].copy()
        
        if seq_data.empty:
            print(f"❌ No valid bounding boxes found for sequence: {sequence_id}")
            return None
        
        # Sort by frame number
        seq_data = seq_data.sort_values('frame_num').reset_index(drop=True)
        
        # Extract bounding box information
        bboxes = []
        for _, row in seq_data.iterrows():
            bbox = row['bbox_parsed']
            if bbox:
                bboxes.append({
                    'frame_num': row['frame_num'],
                    'top': bbox['top'],
                    'left': bbox['left'],
                    'height': bbox['height'],
                    'width': bbox['width'],
                    'bottom': bbox['top'] + bbox['height'],
                    'right': bbox['left'] + bbox['width']
                })
        
        if not bboxes:
            print(f"❌ No valid bounding boxes extracted for sequence: {sequence_id}")
            return None
        
        # Find maximum dimensions and bounding region
        max_width = max(bbox['width'] for bbox in bboxes)
        max_height = max(bbox['height'] for bbox in bboxes)
        
        # Find the overall bounding region that encompasses all bboxes
        min_left = min(bbox['left'] for bbox in bboxes)
        max_right = max(bbox['right'] for bbox in bboxes)
        min_top = min(bbox['top'] for bbox in bboxes)
        max_bottom = max(bbox['bottom'] for bbox in bboxes)
        
        # Calculate frame mapping (GAVD frame numbers to video frame indices)
        frame_nums = [bbox['frame_num'] for bbox in bboxes]
        min_frame = min(frame_nums)
        max_frame = max(frame_nums)
        
        # Create frame mapping: GAVD frame number -> video frame index
        frame_mapping = {}
        for i, frame_num in enumerate(sorted(frame_nums)):
            frame_mapping[frame_num] = i
        
        return {
            'sequence_id': sequence_id,
            'bboxes': bboxes,
            'max_width': max_width,
            'max_height': max_height,
            'crop_region': {
                'left': min_left,
                'top': min_top,
                'width': max_right - min_left,
                'height': max_bottom - min_top
            },
            'frame_mapping': frame_mapping,
            'frame_range': (min_frame, max_frame),
            'total_frames': len(bboxes),
            'metadata': {
                'gait_pat': seq_data['gait_pat'].iloc[0],
                'dataset': seq_data['dataset'].iloc[0],
                'cam_view': seq_data['cam_view'].iloc[0]
            }
        }
    
    def crop_sequence_video(self, sequence_id, use_max_dimensions=True, center_crop=False, 
                           topleft_as_center=True):
        """
        Crop a sequence video using bounding box information.
        
        Args:
            sequence_id: ID of the sequence to process
            use_max_dimensions: If True, use max width/height for consistent crop size
            center_crop: If True, center the crop region around the person
            topleft_as_center: If True, treat bbox top-left as center point (overrides center_crop)
        
        Returns:
            bool: Success status
        """
        # Get sequence information
        seq_info = self.get_sequence_info(sequence_id)
        if not seq_info:
            return False
        
        # Find input video file
        input_video_path = self.sequences_dir / f"{sequence_id}.mp4"
        if not input_video_path.exists():
            print(f"❌ Video file not found: {input_video_path}")
            return False
        
        # Output video path
        output_video_path = self.output_dir / f"{sequence_id}_cropped.mp4"
        
        # Skip if already processed
        if output_video_path.exists():
            print(f"⏭️ Skipping {sequence_id} - already processed")
            return True
        
        print(f"\n🎬 Processing sequence: {sequence_id}")
        print(f"📊 Gait pattern: {seq_info['metadata']['gait_pat']}")
        print(f"📐 Max dimensions: {seq_info['max_width']:.0f}x{seq_info['max_height']:.0f}")
        print(f"🎯 Frame range: {seq_info['frame_range'][0]} to {seq_info['frame_range'][1]}")
        
        try:
            # Open input video
            cap = cv2.VideoCapture(str(input_video_path))
            if not cap.isOpened():
                print(f"❌ Cannot open video: {input_video_path}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"📹 Video info: {original_width}x{original_height}, {total_frames} frames, {fps:.2f} FPS")
            
            # Determine crop dimensions
            if use_max_dimensions:
                crop_width = int(seq_info['max_width'])
                crop_height = int(seq_info['max_height'])
            else:
                crop_width = int(seq_info['crop_region']['width'])
                crop_height = int(seq_info['crop_region']['height'])
            
            print(f"✂️ Crop size: {crop_width}x{crop_height}")
            
            # Setup output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (crop_width, crop_height))
            
            if not out.isOpened():
                print(f"❌ Cannot create output video: {output_video_path}")
                cap.release()
                return False
            
            # Create bbox interpolation for smooth cropping
            bboxes_dict = {bbox['frame_num']: bbox for bbox in seq_info['bboxes']}
            
            frames_written = 0
            frame_idx = 0
            
            # Progress bar
            pbar = tqdm(total=total_frames, desc="Processing frames")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Map video frame index to GAVD frame number
                # Since GAVD frames start from min_frame, we need to offset
                gavd_frame_num = seq_info['frame_range'][0] + frame_idx
                
                # Get bounding box for this frame (or interpolate)
                if gavd_frame_num in bboxes_dict:
                    bbox = bboxes_dict[gavd_frame_num]
                else:
                    # Find nearest bounding box
                    bbox = min(seq_info['bboxes'], 
                             key=lambda x: abs(x['frame_num'] - gavd_frame_num))
                
                # Calculate crop coordinates
                if topleft_as_center:
                    # Treat bbox top-left as center point
                    crop_left = int(bbox['left'] - crop_width / 2)
                    crop_top = int(bbox['top'] - crop_height / 2)
                elif center_crop:
                    # Center the crop around the bounding box center
                    bbox_center_x = bbox['left'] + bbox['width'] / 2
                    bbox_center_y = bbox['top'] + bbox['height'] / 2
                    
                    crop_left = int(bbox_center_x - crop_width / 2)
                    crop_top = int(bbox_center_y - crop_height / 2)
                else:
                    # Use bounding box top-left as crop origin
                    crop_left = int(bbox['left'])
                    crop_top = int(bbox['top'])
                
                # Ensure crop region is within frame bounds
                crop_left = max(0, min(crop_left, original_width - crop_width))
                crop_top = max(0, min(crop_top, original_height - crop_height))
                crop_right = crop_left + crop_width
                crop_bottom = crop_top + crop_height
                
                # Crop the frame
                cropped_frame = frame[crop_top:crop_bottom, crop_left:crop_right]
                
                # Handle edge cases where crop might be smaller than expected
                if cropped_frame.shape[:2] != (crop_height, crop_width):
                    # Resize to expected dimensions
                    cropped_frame = cv2.resize(cropped_frame, (crop_width, crop_height))
                
                out.write(cropped_frame)
                frames_written += 1
                frame_idx += 1
                pbar.update(1)
            
            # Cleanup
            cap.release()
            out.release()
            pbar.close()
            
            print(f"✅ Cropped video saved: {frames_written} frames written")
            print(f"💾 Output: {output_video_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cropping video {sequence_id}: {e}")
            return False
    
    def process_all_sequences(self, max_sequences=None, skip_existing=True, 
                             use_max_dimensions=True, center_crop=False, topleft_as_center=True):
        """
        Process all available sequences.
        
        Args:
            max_sequences: Maximum number of sequences to process (None for all)
            skip_existing: Whether to skip already processed sequences
            use_max_dimensions: If True, use max width/height for consistent crop size
            center_crop: If True, center the crop region around the person
            topleft_as_center: If True, treat bbox top-left as center point
        """
        # Get all unique sequence IDs from annotations
        sequence_ids = self.df_all['seq'].dropna().unique()
        
        print(f"\n🔍 Found {len(sequence_ids)} unique sequences in annotations")
        
        # Filter for sequences that have corresponding video files
        available_sequences = []
        for seq_id in sequence_ids:
            video_path = self.sequences_dir / f"{seq_id}.mp4"
            if video_path.exists():
                available_sequences.append(seq_id)
        
        print(f"📹 Found {len(available_sequences)} sequences with video files")
        
        if max_sequences:
            available_sequences = available_sequences[:max_sequences]
            print(f"🎯 Processing first {len(available_sequences)} sequences")
        
        # Process each sequence
        successful = 0
        failed = 0
        skipped = 0
        
        for i, seq_id in enumerate(available_sequences, 1):
            print(f"\n[{i}/{len(available_sequences)}] Processing: {seq_id}")
            
            # Check if already processed
            output_path = self.output_dir / f"{seq_id}_cropped.mp4"
            if skip_existing and output_path.exists():
                print(f"⏭️ Skipping - already exists")
                skipped += 1
                continue
            
            success = self.crop_sequence_video(
                seq_id, 
                use_max_dimensions=use_max_dimensions,
                center_crop=center_crop,
                topleft_as_center=topleft_as_center
            )
            if success:
                successful += 1
            else:
                failed += 1
        
        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"⏭️ Skipped: {skipped}")
        print(f"📁 Output directory: {self.output_dir}")
    
    def get_processing_stats(self):
        """Get statistics about the dataset and processing status."""
        sequence_ids = self.df_all['seq'].dropna().unique()
        
        stats = {
            'total_annotations': len(self.df_all),
            'unique_sequences': len(sequence_ids),
            'gait_patterns': self.df_all['gait_pat'].value_counts().to_dict(),
            'datasets': self.df_all['dataset'].value_counts().to_dict(),
            'camera_views': self.df_all['cam_view'].value_counts().to_dict()
        }
        
        # Check for available video files
        available_videos = 0
        processed_videos = 0
        
        for seq_id in sequence_ids:
            video_path = self.sequences_dir / f"{seq_id}.mp4"
            if video_path.exists():
                available_videos += 1
                
                output_path = self.output_dir / f"{seq_id}_cropped.mp4"
                if output_path.exists():
                    processed_videos += 1
        
        stats['available_videos'] = available_videos
        stats['processed_videos'] = processed_videos
        
        return stats
    
    def create_bbox_overlay_video(self, sequence_id, line_thickness=3, show_info_text=True):
        """
        Create a video with green bounding box overlays on the original sequence.
        
        Args:
            sequence_id: ID of the sequence to process
            line_thickness: Thickness of the bounding box outline
            show_info_text: Whether to show frame info and bbox dimensions
        
        Returns:
            bool: Success status
        """
        # Get sequence information
        seq_info = self.get_sequence_info(sequence_id)
        if not seq_info:
            return False
        
        # Find input video file
        input_video_path = self.sequences_dir / f"{sequence_id}.mp4"
        if not input_video_path.exists():
            print(f"❌ Video file not found: {input_video_path}")
            return False
        
        # Output video path
        overlay_output_dir = self.output_dir / "bbox_overlays"
        overlay_output_dir.mkdir(exist_ok=True)
        output_video_path = overlay_output_dir / f"{sequence_id}_bbox_overlay.mp4"
        
        # Skip if already processed
        if output_video_path.exists():
            print(f"⏭️ Skipping {sequence_id} - overlay already exists")
            return True
        
        print(f"\n🎬 Creating bbox overlay for: {sequence_id}")
        print(f"📊 Gait pattern: {seq_info['metadata']['gait_pat']}")
        print(f"📐 Max dimensions: {seq_info['max_width']:.0f}x{seq_info['max_height']:.0f}")
        print(f"🎯 Frame range: {seq_info['frame_range'][0]} to {seq_info['frame_range'][1]}")
        
        try:
            # Open input video
            cap = cv2.VideoCapture(str(input_video_path))
            if not cap.isOpened():
                print(f"❌ Cannot open video: {input_video_path}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"📹 Video info: {original_width}x{original_height}, {total_frames} frames, {fps:.2f} FPS")
            
            # Setup output video writer (same dimensions as input)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (original_width, original_height))
            
            if not out.isOpened():
                print(f"❌ Cannot create output video: {output_video_path}")
                cap.release()
                return False
            
            # Create bbox mapping for easy lookup
            bboxes_dict = {bbox['frame_num']: bbox for bbox in seq_info['bboxes']}
            
            frames_written = 0
            frame_idx = 0
            
            # Progress bar
            pbar = tqdm(total=total_frames, desc="Adding bbox overlays")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Map video frame index to GAVD frame number
                gavd_frame_num = seq_info['frame_range'][0] + frame_idx
                
                # Get bounding box for this frame (or nearest)
                if gavd_frame_num in bboxes_dict:
                    bbox = bboxes_dict[gavd_frame_num]
                else:
                    # Find nearest bounding box
                    bbox = min(seq_info['bboxes'], 
                             key=lambda x: abs(x['frame_num'] - gavd_frame_num))
                
                # Draw bounding box overlay
                bbox_left = int(bbox['left'])
                bbox_top = int(bbox['top'])
                bbox_right = int(bbox['left'] + bbox['width'])
                bbox_bottom = int(bbox['top'] + bbox['height'])
                
                # Draw green bounding box
                cv2.rectangle(frame, (bbox_left, bbox_top), (bbox_right, bbox_bottom), 
                            (0, 255, 0), line_thickness)  # Green color in BGR
                
                # Add corner markers for better visibility
                corner_size = 10
                # Top-left corner
                cv2.line(frame, (bbox_left, bbox_top), (bbox_left + corner_size, bbox_top), (0, 255, 0), line_thickness + 1)
                cv2.line(frame, (bbox_left, bbox_top), (bbox_left, bbox_top + corner_size), (0, 255, 0), line_thickness + 1)
                
                # Top-right corner
                cv2.line(frame, (bbox_right, bbox_top), (bbox_right - corner_size, bbox_top), (0, 255, 0), line_thickness + 1)
                cv2.line(frame, (bbox_right, bbox_top), (bbox_right, bbox_top + corner_size), (0, 255, 0), line_thickness + 1)
                
                # Bottom-left corner
                cv2.line(frame, (bbox_left, bbox_bottom), (bbox_left + corner_size, bbox_bottom), (0, 255, 0), line_thickness + 1)
                cv2.line(frame, (bbox_left, bbox_bottom), (bbox_left, bbox_bottom - corner_size), (0, 255, 0), line_thickness + 1)
                
                # Bottom-right corner
                cv2.line(frame, (bbox_right, bbox_bottom), (bbox_right - corner_size, bbox_bottom), (0, 255, 0), line_thickness + 1)
                cv2.line(frame, (bbox_right, bbox_bottom), (bbox_right, bbox_bottom - corner_size), (0, 255, 0), line_thickness + 1)
                
                if show_info_text:
                    # Add text overlay with frame and bbox info
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    text_color = (0, 255, 0)  # Green
                    text_thickness = 2
                    
                    # Frame info
                    frame_text = f"Frame: {gavd_frame_num} ({frame_idx}/{total_frames-1})"
                    cv2.putText(frame, frame_text, (10, 30), font, font_scale, text_color, text_thickness)
                    
                    # Bbox dimensions
                    bbox_text = f"BBox: {bbox['width']:.0f}x{bbox['height']:.0f} at ({bbox['left']:.0f},{bbox['top']:.0f})"
                    cv2.putText(frame, bbox_text, (10, 60), font, font_scale, text_color, text_thickness)
                    
                    # Gait pattern
                    gait_text = f"Gait: {seq_info['metadata']['gait_pat']}"
                    cv2.putText(frame, gait_text, (10, 90), font, font_scale, text_color, text_thickness)
                    
                    # Max dimensions reference
                    max_text = f"Max: {seq_info['max_width']:.0f}x{seq_info['max_height']:.0f}"
                    cv2.putText(frame, max_text, (10, 120), font, font_scale, text_color, text_thickness)
                
                out.write(frame)
                frames_written += 1
                frame_idx += 1
                pbar.update(1)
            
            # Cleanup
            cap.release()
            out.release()
            pbar.close()
            
            print(f"✅ Bbox overlay video saved: {frames_written} frames written")
            print(f"💾 Output: {output_video_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating bbox overlay for {sequence_id}: {e}")
            return False
    
    def create_multiple_bbox_overlays(self, num_videos=5, skip_existing=True):
        """
        Create bounding box overlay videos for multiple sequences.
        
        Args:
            num_videos: Number of videos to create overlays for
            skip_existing: Whether to skip already processed videos
        """
        # Get all unique sequence IDs from annotations
        sequence_ids = self.df_all['seq'].dropna().unique()
        
        print(f"\n🔍 Found {len(sequence_ids)} unique sequences in annotations")
        
        # Filter for sequences that have corresponding video files
        available_sequences = []
        for seq_id in sequence_ids:
            video_path = self.sequences_dir / f"{seq_id}.mp4"
            if video_path.exists():
                available_sequences.append(seq_id)
        
        print(f"📹 Found {len(available_sequences)} sequences with video files")
        
        # Limit to requested number
        sequences_to_process = available_sequences[:num_videos]
        print(f"🎯 Creating bbox overlays for {len(sequences_to_process)} sequences")
        
        # Process each sequence
        successful = 0
        failed = 0
        skipped = 0
        
        for i, seq_id in enumerate(sequences_to_process, 1):
            print(f"\n[{i}/{len(sequences_to_process)}] Processing overlay for: {seq_id}")
            
            # Check if already processed
            overlay_output_dir = self.output_dir / "bbox_overlays"
            output_path = overlay_output_dir / f"{seq_id}_bbox_overlay.mp4"
            if skip_existing and output_path.exists():
                print(f"⏭️ Skipping - overlay already exists")
                skipped += 1
                continue
            
            success = self.create_bbox_overlay_video(seq_id)
            if success:
                successful += 1
            else:
                failed += 1
        
        print(f"\n📊 BBOX OVERLAY SUMMARY:")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"⏭️ Skipped: {skipped}")
        print(f"📁 Output directory: {self.output_dir / 'bbox_overlays'}")

def main():
    """Main function to demonstrate usage."""
    print("🚀 GAVD Bounding Box Video Processor")
    print("=" * 50)
    
    # Initialize processor
    processor = GAVDBBoxProcessor()
    
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
    
    # Process a few sequences as demo
    print(f"\n🎬 Starting video processing...")
    processor.process_all_sequences(max_sequences=5)  # Process first 5 as demo
    
    # Create bbox overlay videos
    print(f"\n📹 Creating bounding box overlay videos...")
    processor.create_multiple_bbox_overlays(num_videos=100)

if __name__ == "__main__":
    main()
