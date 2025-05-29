"""
Image Detector Module for Visual Element Detection

This module implements YOLO11-Nano based detection of visual elements in presentation slides.
Part of Phase 1: Image Detection in the Image Processing Pipeline.

Classes:
    ImageDetector: Main class for detecting visual elements in frames using YOLO11-Nano

Dependencies:
    - ultralytics (YOLO11)
    - opencv-python
    - onnx
    - onnxruntime
    - numpy

Author: Intelligent Data Extraction System
Version: 2.0.0
"""

from typing import List, Dict, Tuple, Optional, Union
import cv2
import numpy as np
from pathlib import Path
import logging
import os
import time
from ultralytics import YOLO
import onnxruntime as ort
class ImageDetector:
    """
    Detects visual elements in presentation slides using YOLO11-Nano.
    
    This class handles the detection of charts, diagrams, images, and other
    visual content within slide frames, providing bounding box coordinates
    for further processing using the latest YOLO11-Nano model.
    """
    
    def __init__(self, 
                 model_type: str = "yolo11n", 
                 use_onnx: bool = True,
                 confidence_threshold: float = 0.5,
                 device: str = "auto"):
        """
        Initialize the ImageDetector with YOLO11-Nano model.
        
        Args:
            model_type (str): YOLO model type ('yolo11n', 'yolo11s', etc.)
            use_onnx (bool): Whether to use ONNX format for TensorFlow compatibility
            confidence_threshold (float): Minimum confidence for detections
            device (str): Device for inference ('auto', 'cpu', 'gpu', 'cuda')
        """
        self.model_type = model_type
        self.use_onnx = use_onnx
        self.confidence_threshold = confidence_threshold
        self.device = self._setup_device(device)
        
        # Initialize model
        self.model = None
        self.onnx_session = None
        self.model_path = None
        
        # Detection statistics
        self.total_detections = 0
        self.processing_times = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self._initialize_model()
    
    def _setup_device(self, device: str) -> str:
        """Setup optimal device for inference."""
        if device == "auto":
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                return "cuda"
            else:
                return "cpu"
        return device
    
    def _initialize_model(self) -> None:
        """Initialize YOLO11 model and optionally export to ONNX."""
        try:
            self.logger.info(f"Initializing {self.model_type} model...")
            
            # Load YOLO11 model
            self.model = YOLO(f"{self.model_type}.pt")
            
            if self.use_onnx:
                # Export to ONNX for TensorFlow compatibility
                onnx_path = f"{self.model_type}.onnx"
                if not os.path.exists(onnx_path):
                    self.logger.info("Exporting model to ONNX format...")
                    self.model.export(
                        format="onnx",
                        simplify=True,
                        dynamic=True,
                        half=False  # Use FP32 for better compatibility
                    )
                
                # Initialize ONNX Runtime session
                self.onnx_session = ort.InferenceSession(
                    onnx_path,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self.model_path = onnx_path
                self.logger.info(f"ONNX model loaded: {onnx_path}")
            else:
                self.model_path = f"{self.model_type}.pt"
                self.logger.info(f"PyTorch model loaded: {self.model_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            raise
      def detect_visual_elements(self, frame_path: str, frame_id: str, timestamp: float) -> Dict:
        """
        Detect visual elements in a single frame using YOLO11-Nano.
        
        Args:
            frame_path (str): Path to the frame image
            frame_id (str): Unique identifier for the frame
            timestamp (float): Timestamp of the frame in video
            
        Returns:
            Dict: Detection results with bounding boxes and metadata
        """
        start_time = time.time()
        
        try:
            # Preprocess frame
            frame = self.preprocess_frame(frame_path)
            if frame is None:
                return self._empty_result(frame_id, timestamp, "Failed to load frame")
            
            # Run detection
            if self.use_onnx:
                detections = self._detect_with_onnx(frame)
            else:
                detections = self._detect_with_pytorch(frame)
            
            # Validate and filter detections
            validated_detections = self.validate_detections(detections, frame.shape[:2])
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Update statistics
            self.total_detections += len(validated_detections)
            
            # Build result
            result = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "frame_path": frame_path,
                "detections": validated_detections,
                "detection_count": len(validated_detections),
                "processing_time": processing_time,
                "model_type": self.model_type,
                "confidence_threshold": self.confidence_threshold,
                "status": "success"
            }
            
            self.logger.info(f"Frame {frame_id}: {len(validated_detections)} detections in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Detection failed for frame {frame_id}: {str(e)}")
            return self._empty_result(frame_id, timestamp, f"Detection error: {str(e)}")
    
    def _detect_with_onnx(self, frame: np.ndarray) -> List:
        """Run detection using ONNX Runtime."""
        # Prepare input for ONNX
        input_data = self._prepare_onnx_input(frame)
        
        # Run inference
        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: input_data})
        
        # Process outputs
        return self._process_onnx_outputs(outputs[0], frame.shape[:2])
    
    def _detect_with_pytorch(self, frame: np.ndarray) -> List:
        """Run detection using PyTorch model."""
        results = self.model(frame, conf=self.confidence_threshold, device=self.device)
        return self._process_pytorch_outputs(results)
    
    def _prepare_onnx_input(self, frame: np.ndarray) -> np.ndarray:
        """Prepare frame for ONNX inference."""
        # Resize to model input size (640x640)
        resized = cv2.resize(frame, (640, 640))
        
        # Convert to RGB and normalize
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_frame.astype(np.float32) / 255.0
        
        # Add batch dimension and reorder to NCHW
        input_data = np.transpose(normalized, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)
        
        return input_data
    
    def _process_onnx_outputs(self, outputs: np.ndarray, original_shape: Tuple[int, int]) -> List:
        """Process ONNX model outputs to detection format."""
        detections = []
        
        # YOLO11 output format: [batch, anchors, 4+num_classes]
        # outputs shape: [1, 8400, 84] for COCO dataset
        
        for detection in outputs[0]:  # Remove batch dimension
            # Extract confidence scores for all classes
            class_scores = detection[4:]
            max_confidence = np.max(class_scores)
            
            if max_confidence >= self.confidence_threshold:
                class_id = np.argmax(class_scores)
                
                # Extract bounding box (center_x, center_y, width, height)
                x_center, y_center, width, height = detection[:4]
                
                # Convert to corner coordinates
                x1 = int((x_center - width / 2) * original_shape[1] / 640)
                y1 = int((y_center - height / 2) * original_shape[0] / 640)
                x2 = int((x_center + width / 2) * original_shape[1] / 640)
                y2 = int((y_center + height / 2) * original_shape[0] / 640)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(max_confidence),
                    'class_id': int(class_id),
                    'class_name': self._get_class_name(class_id)
                })
        
        return detections
    
    def _process_pytorch_outputs(self, results) -> List:
        """Process PyTorch model outputs to detection format."""
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                    confidence = boxes.conf[i].cpu().numpy()
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    detections.append({
                        'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': self._get_class_name(class_id)
                    })
        
        return detections
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID using COCO dataset labels."""
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        if 0 <= class_id < len(coco_classes):
            return coco_classes[class_id]
        return f"class_{class_id}"
    
    def _empty_result(self, frame_id: str, timestamp: float, error_msg: str) -> Dict:
        """Return empty result with error information."""
        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "frame_path": None,
            "detections": [],
            "detection_count": 0,
            "processing_time": 0.0,
            "model_type": self.model_type,
            "confidence_threshold": self.confidence_threshold,
            "status": "error",
            "error": error_msg
        }
      def preprocess_frame(self, frame_path: str) -> Optional[np.ndarray]:
        """
        Preprocess frame for YOLO11 detection.
        
        Args:
            frame_path (str): Path to frame image
            
        Returns:
            Optional[np.ndarray]: Preprocessed frame or None if failed
        """
        try:
            # Load frame
            frame = cv2.imread(frame_path)
            if frame is None:
                self.logger.error(f"Failed to load frame: {frame_path}")
                return None
            
            # Basic preprocessing
            # YOLO11 handles most preprocessing internally, but we can do basic cleanup
            
            # Apply slight noise reduction if needed
            if frame.shape[0] > 1080 or frame.shape[1] > 1920:
                # Apply noise reduction for high-resolution images
                frame = cv2.bilateralFilter(frame, 9, 75, 75)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Frame preprocessing failed for {frame_path}: {str(e)}")
            return None
    
    def validate_detections(self, detections: List, frame_shape: Tuple[int, int]) -> List:
        """
        Validate and filter detection results.
        
        Args:
            detections (List): Raw detection results from YOLO11
            frame_shape (Tuple[int, int]): Original frame dimensions (height, width)
            
        Returns:
            List: Filtered and validated detections
        """
        validated = []
        
        for detection in detections:
            # Check confidence threshold
            if detection['confidence'] < self.confidence_threshold:
                continue
            
            # Validate bounding box coordinates
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, frame_shape[1]))
            y1 = max(0, min(y1, frame_shape[0]))
            x2 = max(0, min(x2, frame_shape[1]))
            y2 = max(0, min(y2, frame_shape[0]))
            
            # Check minimum box size (avoid tiny detections)
            box_width = x2 - x1
            box_height = y2 - y1
            
            if box_width < 10 or box_height < 10:
                continue
            
            # Check maximum box size (avoid full-frame detections unless very confident)
            frame_area = frame_shape[0] * frame_shape[1]
            box_area = box_width * box_height
            
            if box_area > 0.8 * frame_area and detection['confidence'] < 0.9:
                continue
            
            # Update bbox with corrected coordinates
            detection['bbox'] = [x1, y1, x2, y2]
            detection['area'] = box_area
            detection['width'] = box_width
            detection['height'] = box_height
            
            validated.append(detection)
        
        # Apply Non-Maximum Suppression if we have overlapping detections
        if len(validated) > 1:
            validated = self._apply_nms(validated)
        
        return validated
    
    def _apply_nms(self, detections: List, iou_threshold: float = 0.5) -> List:
        """Apply Non-Maximum Suppression to remove overlapping detections."""
        if not detections:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        
        for detection in detections:
            # Check if this detection overlaps significantly with any kept detection
            should_keep = True
            
            for kept_detection in filtered:
                iou = self._calculate_iou(detection['bbox'], kept_detection['bbox'])
                if iou > iou_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                filtered.append(detection)
        
        return filtered
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def get_statistics(self) -> Dict:
        """Get detection statistics."""
        if not self.processing_times:
            return {"status": "no_data"}
        
        return {
            "total_detections": self.total_detections,
            "frames_processed": len(self.processing_times),
            "avg_processing_time": np.mean(self.processing_times),
            "min_processing_time": min(self.processing_times),
            "max_processing_time": max(self.processing_times),
            "model_type": self.model_type,
            "device": self.device,
            "use_onnx": self.use_onnx
        }
    
    def batch_detect(self, frame_paths: List[str], batch_size: int = 4) -> List[Dict]:
        """
        Process multiple frames in batches for better efficiency.
        
        Args:
            frame_paths (List[str]): List of frame paths to process
            batch_size (int): Number of frames to process in each batch
            
        Returns:
            List[Dict]: List of detection results
        """
        results = []
        
        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i:i + batch_size]
            
            for j, frame_path in enumerate(batch_paths):
                frame_id = f"frame_{i + j:05d}"
                timestamp = (i + j) * (1/30)  # Assuming 30 FPS
                
                result = self.detect_visual_elements(frame_path, frame_id, timestamp)
                results.append(result)
        
        return results
