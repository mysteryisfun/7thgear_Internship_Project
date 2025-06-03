"""
Optimized Output Manager for structured data storage and JSON export.

This module handles saving processed video analysis results to structured JSON files
with timestamps, metadata, and proper organization. Optimized to reduce file size
and improve organization.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path


class OutputManager:
    def __init__(self, output_dir: str):
        """
        Initialize the OutputManager with output directory.
        
        Args:
            output_dir: Base directory for saving outputs
        """
        self.output_dir = Path(output_dir)
        self.results_dir = self.output_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def save_processing_results(
        self,
        video_path: str,
        frame_data: List[Tuple[str, float]],
        scene_indices: List[int],
        extracted_texts: List[str],
        processed_texts: List[str],
        processing_params: Dict[str, Any]
    ) -> str:
        """
        Save complete processing results to a structured JSON file.
        
        Args:
            video_path: Path to the original video file
            frame_data: List of (frame_path, timestamp) tuples
            scene_indices: List of frame indices where scenes change
            extracted_texts: Raw extracted texts from all frames
            processed_texts: NLP-processed texts from scene frames
            processing_params: Parameters used for processing
            
        Returns:
            Path to the saved JSON file
        """
        # Generate output folder and filename based on video name and timestamp
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create individual folder for this analysis
        analysis_folder = f"{video_name}_analysis_{timestamp}"
        analysis_dir = self.results_dir / analysis_folder
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"{video_name}_analysis_{timestamp}.json"
        output_path = analysis_dir / output_filename
        
        # Build scene data with timestamp ranges
        scenes_data = []
        scene_frames_data = []  # Only store frame data for scene change frames
        
        for i, scene_idx in enumerate(scene_indices):
            # Calculate timestamp range for this scene
            start_timestamp = frame_data[scene_idx][1]
            
            # End timestamp is either next scene or end of video
            if i + 1 < len(scene_indices):
                end_timestamp = frame_data[scene_indices[i + 1]][1]
            else:
                end_timestamp = frame_data[-1][1] if frame_data else start_timestamp
            
            scene_data = {
                "scene_number": i + 1,
                "frame_index": scene_idx,
                "frame_path": frame_data[scene_idx][0],
                "timestamp_range": {
                    "start_seconds": start_timestamp,
                    "end_seconds": end_timestamp,
                    "duration_seconds": end_timestamp - start_timestamp
                },
                "raw_text": extracted_texts[scene_idx] if scene_idx < len(extracted_texts) else "",
                "processed_text": processed_texts[i] if i < len(processed_texts) else "",
                "text_length": len(processed_texts[i]) if i < len(processed_texts) else 0
            }
            scenes_data.append(scene_data)
            
            # Store frame data only for scene change frames
            scene_frames_data.append({
                "frame_index": scene_idx,
                "frame_path": frame_data[scene_idx][0],
                "timestamp_seconds": start_timestamp,
                "is_scene_change": True
            })
        
        # Build complete results structure (without full frame_data)
        results = {
            "metadata": {
                "video_file": {
                    "path": video_path,
                    "filename": Path(video_path).name,
                    "size_bytes": os.path.getsize(video_path) if os.path.exists(video_path) else None
                },
                "processing_info": {
                    "timestamp": datetime.now().isoformat(),
                    "processing_date": datetime.now().strftime("%Y-%m-%d"),
                    "processing_time": datetime.now().strftime("%H:%M:%S"),
                    "total_frames_extracted": len(frame_data),
                    "total_scenes_detected": len(scene_indices),
                    "video_duration_seconds": frame_data[-1][1] if frame_data else 0
                },
                "parameters": processing_params,
                "output_info": {
                    "output_directory": str(self.output_dir),
                    "analysis_folder": analysis_folder,
                    "results_file": output_filename,
                    "frames_directory": str(self.output_dir / "frames"),
                    "keyframes_directory": str(self.output_dir / "keyframes")
                }
            },
            "summary": {
                "total_scenes": len(scene_indices),
                "total_frames": len(frame_data),
                "video_duration": frame_data[-1][1] if frame_data else 0,
                "average_scene_duration": sum(scene["timestamp_range"]["duration_seconds"] for scene in scenes_data) / len(scenes_data) if scenes_data else 0,
                "total_text_length": sum(len(text) for text in processed_texts),
                "scenes_with_text": len([s for s in scenes_data if s["text_length"] > 0])
            },
            "scenes": scenes_data,
            "scene_change_frames": scene_frames_data  # Only frames where scenes change
        }
        
        # Save to JSON file with proper formatting
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return str(output_path)
    
    def load_processing_results(self, json_path: str) -> Dict[str, Any]:
        """
        Load previously saved processing results from JSON file.
        
        Args:
            json_path: Path to the JSON results file
            
        Returns:
            Dictionary containing the loaded results
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_results_files(self) -> List[str]:
        """
        Get list of all result files in the results directory.
        
        Returns:
            List of JSON result file paths
        """
        result_files = []
        for analysis_dir in self.results_dir.glob("*_analysis_*"):
            if analysis_dir.is_dir():
                json_files = list(analysis_dir.glob("*.json"))
                result_files.extend([str(f) for f in json_files])
        return result_files
    
    def get_analysis_folders(self) -> List[str]:
        """
        Get list of all analysis folders in the results directory.
        
        Returns:
            List of analysis folder paths
        """
        return [str(f) for f in self.results_dir.glob("*_analysis_*") if f.is_dir()]
    
    def create_summary_report(self, json_path: str) -> str:
        """
        Create a human-readable summary report from JSON results.
        
        Args:
            json_path: Path to the JSON results file
            
        Returns:
            Path to the created summary report
        """
        data = self.load_processing_results(json_path)
        
        # Create summary text
        summary_lines = [
            f"Video Analysis Summary Report",
            f"{'=' * 50}",
            f"",
            f"Video File: {data['metadata']['video_file']['filename']}",
            f"Processing Date: {data['metadata']['processing_info']['processing_date']}",
            f"Processing Time: {data['metadata']['processing_info']['processing_time']}",
            f"Analysis Folder: {data['metadata']['output_info']['analysis_folder']}",
            f"",
            f"Summary Statistics:",
            f"- Total Scenes: {data['summary']['total_scenes']}",
            f"- Total Frames: {data['summary']['total_frames']}",
            f"- Video Duration: {data['summary']['video_duration']:.2f} seconds",
            f"- Average Scene Duration: {data['summary']['average_scene_duration']:.2f} seconds",
            f"- Total Text Length: {data['summary']['total_text_length']} characters",
            f"- Scenes with Text: {data['summary']['scenes_with_text']}",
            f"",
            f"Scene Details:",
            f"{'=' * 30}"
        ]
        
        for scene in data['scenes']:
            # Handle processed_text as dict (structured context) or string
            processed_text = scene['processed_text']
            if isinstance(processed_text, dict):
                # Show a compact summary of key fields for the report
                processed_text_str = ", ".join(f"{k}: {str(v)[:30]}" for k, v in processed_text.items() if v)
            else:
                processed_text_str = str(processed_text)
            summary_lines.extend([
                f"",
                f"Scene {scene['scene_number']}:",
                f"  Time Range: {scene['timestamp_range']['start_seconds']:.2f}s - {scene['timestamp_range']['end_seconds']:.2f}s",
                f"  Duration: {scene['timestamp_range']['duration_seconds']:.2f}s",
                f"  Text Length: {scene['text_length']} characters",
                f"  Processed Text: {processed_text_str[:100]}{'...' if len(processed_text_str) > 100 else ''}"
            ])
        
        # Save summary report in the same analysis folder
        summary_path = Path(json_path).with_suffix('.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))
        
        return str(summary_path)
