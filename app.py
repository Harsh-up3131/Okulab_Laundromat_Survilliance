#!/usr/bin/env python3
"""
Enhanced Flask CCTV Surveillance System - Fixed Version
- Fixed pose detection stability with confidence-based tracking
- Fixed region coordinate mapping and scaling
- Added colored bounding boxes per person ID
- Fixed region position mapping
- Enhanced CSV export with login/logout times
- Fixed video playback in results
- Added hand detection and coin insertion tracking
"""

import os
import cv2
import numpy as np
import torch
import mediapipe as mp
from flask import Flask, request, render_template, jsonify, send_file, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import json
from collections import defaultdict, deque
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import uuid
import base64
from pathlib import Path
import logging
import csv
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle numpy arrays and sets
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

app = Flask(__name__)
app.json_encoder = CustomJSONEncoder
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global processing status
processing_status = {}

@dataclass
class Region:
    """Enhanced region with naming and analytics"""
    id: int
    name: str
    points: List[Tuple[int, int]]
    color: Tuple[int, int, int] = (0, 255, 0)
    entry_count: int = 0
    exit_count: int = 0
    current_occupants: Set[int] = field(default_factory=set)
    entry_log: List[Dict] = field(default_factory=list)

@dataclass
class PersonTracker:
    """Enhanced person tracker with path prediction and occlusion handling"""
    person_id: int
    entry_time: float
    last_seen_time: float
    exit_time: Optional[float] = None
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=30))
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=10))
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=20))
    pose_history: deque = field(default_factory=lambda: deque(maxlen=10))
    pose_confidence: Dict[str, float] = field(default_factory=dict)
    current_pose: str = "standing"
    is_occluded: bool = False
    occlusion_start: float = 0.0
    predicted_position: Optional[Tuple[float, float]] = None
    path_prediction: List[Tuple[float, float]] = field(default_factory=list)
    stable_detections: int = 0
    feature_vector: Optional[np.ndarray] = None
    region_entries: Dict[int, List[float]] = field(default_factory=dict)
    current_regions: Set[int] = field(default_factory=set)
    color: Tuple[int, int, int] = field(default_factory=lambda: (255, 255, 255))

class HandDetector:
    """Detect hand and return center point"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=4,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        
    def detect_hands(self, image_crop):
        """Return list of hands with centers and labels: [{"center": (x, y), "label": "Left"|"Right"}]"""
        try:
            if image_crop.size == 0:
                return []
                
            rgb_image = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_image)
            
            hands_out = []
            if results.multi_hand_landmarks:
                h, w = image_crop.shape[:2]
                handedness = results.multi_handedness if hasattr(results, 'multi_handedness') else []
                for idx, landmarks in enumerate(results.multi_hand_landmarks):
                    lm_list = landmarks.landmark
                    x_coords = [lm.x * w for lm in lm_list]
                    y_coords = [lm.y * h for lm in lm_list]
                    center_x = int(sum(x_coords) / len(x_coords))
                    center_y = int(sum(y_coords) / len(y_coords))
                    label = None
                    if handedness and idx < len(handedness):
                        try:
                            label = handedness[idx].classification[0].label
                        except Exception:
                            label = None
                    hands_out.append({"center": (center_x, center_y), "label": label})
            return hands_out
        except Exception as e:
            logger.warning(f"Hand detection error: {e}")
            return []

class CoinDetectionTracker:
    """Track hand points and detect coin insertions"""
    
    def __init__(self):
        self.hand_timers = {}  # person_id -> {region_id: {entered_at, last_seen, cooldown_until}}
        self.coin_insertions = []  # List of insertion events
        self.occlusion_grace = 0.3  # seconds to tolerate brief occlusion/misses
        self.required_dwell = 0.5   # seconds required inside region
        self.cooldown = 0.8         # seconds before allowing another coin event in same region
        
    def check_coin_insertion(self, person_id, hand_points, person_bbox, regions, current_time):
        """Check if any hand is in a region for required dwell time = coin insertion"""
        
        # Initialize person timers if needed
        if person_id not in self.hand_timers:
            self.hand_timers[person_id] = {}
        
        # Normalize input to list of centers
        centers = []
        if isinstance(hand_points, list):
            for hp in hand_points:
                if isinstance(hp, dict) and 'center' in hp:
                    centers.append(hp['center'])
                elif isinstance(hp, (tuple, list)) and len(hp) == 2:
                    centers.append(tuple(hp))
        elif hand_points is not None:
            # Backward compatibility: single point
            centers.append(hand_points)
        
        person_timers = self.hand_timers[person_id]
        
        # Check each region
        for region in regions:
            # Determine if any hand is inside this region
            hand_in_region = False
            for center in centers:
                abs_hand_x = person_bbox[0] + center[0]
                abs_hand_y = person_bbox[1] + center[1]
                absolute_point = (abs_hand_x, abs_hand_y)
                if self.point_in_polygon(absolute_point, region.points):
                    hand_in_region = True
                    break
            
            if hand_in_region:
                # Hand is in this region
                if region.id not in person_timers:
                    # Hand just entered - start timer
                    person_timers[region.id] = {
                        'entered_at': current_time,
                        'last_seen': current_time,
                        'cooldown_until': 0.0
                    }
                    logger.info(f"Hand entered {region.name} - Person {person_id}")
                else:
                    # Hand still in region - update last seen and check dwell
                    timer = person_timers[region.id]
                    timer['last_seen'] = current_time
                    if current_time >= timer.get('cooldown_until', 0.0):
                        duration = current_time - timer['entered_at']
                        if duration >= self.required_dwell:
                            # COIN INSERTION DETECTED!
                            self.record_coin_insertion(person_id, region.name, current_time, duration)
                            # Start cooldown and reset entry
                            timer['entered_at'] = current_time
                            timer['cooldown_until'] = current_time + self.cooldown
            else:
                # Hand not in this region - keep timer during brief occlusion, else remove
                if region.id in person_timers:
                    timer = person_timers[region.id]
                    # If we've been outside/undetected beyond grace, forget timer
                    if (current_time - timer['last_seen']) > self.occlusion_grace:
                        del person_timers[region.id]
    
    def point_in_polygon(self, point, polygon):
        """Check if point is inside polygon"""
        if len(polygon) < 3:
            return False
            
        x, y = point
        inside = False
        j = len(polygon) - 1
        
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    def record_coin_insertion(self, person_id, region_name, timestamp, duration):
        """Record coin insertion event"""
        insertion_event = {
            'person_id': person_id,
            'region_name': region_name,
            'timestamp': timestamp,
            'time_formatted': time.strftime('%H:%M:%S', time.gmtime(timestamp)),
            'duration': round(duration, 2)
        }
        
        self.coin_insertions.append(insertion_event)
        logger.info(f"🪙 COIN INSERTED: Person {person_id} in {region_name} at {insertion_event['time_formatted']}")
    
    def export_csv(self, video_id):
        """Export coin insertions to CSV"""
        csv_filename = os.path.join(app.config['RESULTS_FOLDER'], f"coin_insertions_{video_id}.csv")
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Person_ID', 'Region_Name', 'Timestamp', 'Time', 'Duration_Seconds']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for event in self.coin_insertions:
                writer.writerow({
                    'Person_ID': event['person_id'],
                    'Region_Name': event['region_name'],
                    'Timestamp': event['timestamp'],
                    'Time': event['time_formatted'],
                    'Duration_Seconds': event['duration']
                })
        
        logger.info(f"✅ Coin insertions exported to {csv_filename}")
        return csv_filename

class EnhancedTracker:
    """Optimized tracking with path prediction and occlusion handling"""
    
    def __init__(self, max_disappeared=60, max_distance=150, prediction_frames=10):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.prediction_frames = prediction_frames
        self.trackers = {}
        self.disappeared = {}
        self.next_id = 1
        self.feature_memory = {}
        self.occlusion_threshold = 30
        self.color_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0), (75, 0, 130),
            (255, 20, 147), (0, 191, 255), (50, 205, 50), (220, 20, 60), (255, 140, 0)
        ]
        
    def _get_color_for_id(self, person_id: int) -> Tuple[int, int, int]:
        """Get consistent color for person ID"""
        return self.color_palette[person_id % len(self.color_palette)]
        
    def _extract_features(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract robust features for tracking"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = height / max(1, width)
        area = width * height
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        features = np.array([width, height, aspect_ratio, area, center_x, center_y])
        return features / (np.linalg.norm(features) + 1e-8)
        
    def _predict_path(self, tracker: PersonTracker) -> List[Tuple[float, float]]:
        """Predict future path based on velocity history"""
        if len(tracker.velocity_history) < 3:
            return []
            
        velocities = list(tracker.velocity_history)[-5:]
        avg_vx = np.mean([v[0] for v in velocities])
        avg_vy = np.mean([v[1] for v in velocities])
        
        if len(tracker.position_history) == 0:
            return []
            
        current_pos = tracker.position_history[-1]
        path = []
        
        for i in range(1, self.prediction_frames + 1):
            pred_x = current_pos[0] + avg_vx * i
            pred_y = current_pos[1] + avg_vy * i
            path.append((pred_x, pred_y))
            
        return path
        
    def _calculate_velocity(self, tracker: PersonTracker) -> Optional[Tuple[float, float]]:
        """Calculate current velocity from position history"""
        if len(tracker.position_history) < 2:
            return None
            
        pos1 = tracker.position_history[-1]
        pos2 = tracker.position_history[-2]
        
        vx = pos1[0] - pos2[0]
        vy = pos1[1] - pos2[1]
        
        return (vx, vy)
        
    def register(self, bbox: Tuple[int, int, int, int], confidence: float) -> int:
        """Register new person tracker"""
        person_id = self.next_id
        self.next_id += 1
        
        features = self._extract_features(bbox)
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        
        self.trackers[person_id] = PersonTracker(
            person_id=person_id,
            entry_time=time.time(),
            last_seen_time=time.time(),
            feature_vector=features,
            color=self._get_color_for_id(person_id)
        )
        
        self.trackers[person_id].bbox_history.append(bbox)
        self.trackers[person_id].position_history.append(center)
        self.trackers[person_id].confidence_history.append(confidence)
        self.trackers[person_id].stable_detections = 1
        
        self.disappeared[person_id] = 0
        self.feature_memory[person_id] = features
        
        return person_id
        
    def update(self, detections: List[Tuple]) -> Dict[int, PersonTracker]:
        """Update trackers with enhanced matching and prediction"""
        current_time = time.time()
        
        # Update existing trackers with predictions
        for person_id, tracker in self.trackers.items():
            velocity = self._calculate_velocity(tracker)
            if velocity:
                tracker.velocity_history.append(velocity)
                
            tracker.path_prediction = self._predict_path(tracker)
            
            if self.disappeared[person_id] > self.occlusion_threshold:
                tracker.is_occluded = True
                if tracker.occlusion_start == 0:
                    tracker.occlusion_start = current_time
                    
                if tracker.path_prediction:
                    tracker.predicted_position = tracker.path_prediction[0]
        
        if len(detections) == 0:
            for person_id in list(self.disappeared.keys()):
                self.disappeared[person_id] += 1
                if self.disappeared[person_id] > self.max_disappeared:
                    self._deregister(person_id)
            return self.trackers
        
        # Extract detection info
        detection_centers = np.array([((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) 
                                    for box, _ in detections])
        detection_bboxes = [box for box, _ in detections]
        detection_confs = [conf for _, conf in detections]
        detection_features = [self._extract_features(box) for box, _ in detections]
        
        if len(self.trackers) == 0:
            for bbox, conf in detections:
                self.register(bbox, conf)
        else:
            self._match_detections_to_trackers(
                detection_centers, detection_bboxes, detection_confs, 
                detection_features, current_time
            )
        
        return self.trackers
        
    def _match_detections_to_trackers(self, detection_centers, detection_bboxes, 
                                    detection_confs, detection_features, current_time):
        """Enhanced multi-stage detection matching"""
        tracker_ids = list(self.trackers.keys())
        tracker_centers = []
        
        for person_id in tracker_ids:
            tracker = self.trackers[person_id]
            if tracker.predicted_position and tracker.is_occluded:
                tracker_centers.append(tracker.predicted_position)
            elif len(tracker.position_history) > 0:
                tracker_centers.append(tracker.position_history[-1])
            else:
                tracker_centers.append((0, 0))
        
        tracker_centers = np.array(tracker_centers)
        
        if len(tracker_centers) > 0 and len(detection_centers) > 0:
            # Calculate distance matrix
            distances = np.zeros((len(tracker_ids), len(detection_centers)))
            for i, tracker_id in enumerate(tracker_ids):
                for j, det_center in enumerate(detection_centers):
                    if self.trackers[tracker_id].predicted_position and self.trackers[tracker_id].is_occluded:
                        pred_pos = self.trackers[tracker_id].predicted_position
                        distances[i, j] = np.sqrt((pred_pos[0] - det_center[0])**2 + (pred_pos[1] - det_center[1])**2)
                    else:
                        distances[i, j] = np.sqrt((tracker_centers[i][0] - det_center[0])**2 + (tracker_centers[i][1] - det_center[1])**2)
            
            # Stage 1: Feature-based matching
            used_detections = set()
            used_trackers = set()
            
            for i, tracker_id in enumerate(tracker_ids):
                if i in used_trackers:
                    continue
                    
                for j, det_features in enumerate(detection_features):
                    if j in used_detections:
                        continue
                        
                    if distances[i, j] < self.max_distance * 1.2:
                        if tracker_id in self.feature_memory:
                            stored_features = self.feature_memory[tracker_id]
                            similarity = np.dot(det_features, stored_features)
                            
                            if similarity > 0.85:
                                self._update_tracker(tracker_id, detection_bboxes[j], 
                                                   detection_confs[j], current_time, det_features)
                                used_trackers.add(i)
                                used_detections.add(j)
                                break
            
            # Stage 2: Distance-based matching
            remaining_assignments = []
            for i in range(len(tracker_ids)):
                if i in used_trackers:
                    continue
                for j in range(len(detection_centers)):
                    if j in used_detections:
                        continue
                    if distances[i, j] < self.max_distance:
                        remaining_assignments.append((i, j, distances[i, j]))
            
            remaining_assignments.sort(key=lambda x: x[2])
            for tracker_idx, detection_idx, distance in remaining_assignments:
                if tracker_idx not in used_trackers and detection_idx not in used_detections:
                    tracker_id = tracker_ids[tracker_idx]
                    self._update_tracker(tracker_id, detection_bboxes[detection_idx], 
                                       detection_confs[detection_idx], current_time, 
                                       detection_features[detection_idx])
                    used_trackers.add(tracker_idx)
                    used_detections.add(detection_idx)
            
            # Handle unmatched detections and trackers
            for j in range(len(detection_centers)):
                if j not in used_detections:
                    self.register(detection_bboxes[j], detection_confs[j])
            
            for i in range(len(tracker_ids)):
                if i not in used_trackers:
                    tracker_id = tracker_ids[i]
                    self.disappeared[tracker_id] += 1
                    
    def _update_tracker(self, person_id: int, bbox: Tuple[int, int, int, int], 
                       conf: float, current_time: float, features: np.ndarray):
        """Update individual tracker"""
        tracker = self.trackers[person_id]
        tracker.bbox_history.append(bbox)
        tracker.position_history.append(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2))
        tracker.confidence_history.append(conf)
        tracker.last_seen_time = current_time
        tracker.stable_detections += 1
        tracker.is_occluded = False
        tracker.occlusion_start = 0.0
        tracker.predicted_position = None
        
        if person_id in self.feature_memory:
            self.feature_memory[person_id] = 0.8 * self.feature_memory[person_id] + 0.2 * features
        else:
            self.feature_memory[person_id] = features
        
        self.disappeared[person_id] = 0
        
    def _deregister(self, person_id: int):
        """Deregister person tracker"""
        if person_id in self.trackers:
            self.trackers[person_id].exit_time = time.time()
            del self.trackers[person_id]
        if person_id in self.disappeared:
            del self.disappeared[person_id]
        if person_id in self.feature_memory:
            del self.feature_memory[person_id]

class EnhancedPoseDetector:
    """Enhanced pose detection with better accuracy and stability"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )
        self.pose_history = defaultdict(lambda: deque(maxlen=10))
        self.pose_confidence_tracker = defaultdict(lambda: defaultdict(float))
    
    def detect_pose(self, image_crop, person_id: int) -> str:
        """Detect pose with confidence-based stability"""
        try:
            if image_crop.size == 0:
                return self._get_most_confident_pose(person_id)
                
            rgb_image = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_image)
            
            if results.pose_landmarks:
                pose, confidence = self._classify_pose_with_confidence(results.pose_landmarks.landmark)
                
                # Update pose confidence
                self.pose_confidence_tracker[person_id][pose] += confidence
                
                # Decay old confidences
                for p in self.pose_confidence_tracker[person_id]:
                    self.pose_confidence_tracker[person_id][p] *= 0.95
                
                # Add to history
                self.pose_history[person_id].append(pose)
                
                # Return most confident pose
                return self._get_most_confident_pose(person_id)
                    
            return self._get_most_confident_pose(person_id)
        except Exception as e:
            logger.warning(f"Pose detection error: {e}")
            return self._get_most_confident_pose(person_id)
    
    def _get_most_confident_pose(self, person_id: int) -> str:
        """Get pose with highest confidence, excluding 'unknown'"""
        if person_id not in self.pose_confidence_tracker:
            return "standing"
            
        confidences = self.pose_confidence_tracker[person_id]
        if not confidences:
            return "standing"
            
        # Filter out unknown poses and get the most confident
        valid_poses = {k: v for k, v in confidences.items() if k in ['sitting', 'standing', 'walking']}
        if not valid_poses:
            return "standing"
            
        return max(valid_poses, key=valid_poses.get)
    
    def _classify_pose_with_confidence(self, landmarks) -> Tuple[str, float]:
        """Enhanced pose classification with confidence score"""
        try:
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
            right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            # Calculate visibility-based confidence
            visibility_score = np.mean([
                nose.visibility, left_hip.visibility, right_hip.visibility,
                left_knee.visibility, right_knee.visibility,
                left_ankle.visibility, right_ankle.visibility
            ])
            
            hip_center_y = (left_hip.y + right_hip.y) / 2
            knee_center_y = (left_knee.y + right_knee.y) / 2
            ankle_center_y = (left_ankle.y + right_ankle.y) / 2
            
            body_height = abs(nose.y - hip_center_y)
            leg_bend = abs(hip_center_y - knee_center_y)
            leg_extension = abs(knee_center_y - ankle_center_y)
            
            # Enhanced pose classification with confidence
            if leg_bend < 0.06 * body_height and leg_extension > 0.15 * body_height:
                return "sitting", visibility_score * 0.8
            elif leg_bend > 0.2 * body_height or leg_extension < 0.1 * body_height:
                # Check for walking motion using ankle positions
                ankle_diff = abs(left_ankle.y - right_ankle.y)
                if ankle_diff > 0.05:
                    return "walking", visibility_score * 0.9
                else:
                    return "standing", visibility_score * 0.7
            else:
                return "standing", visibility_score * 0.8
                
        except Exception as e:
            logger.warning(f"Pose classification error: {e}")
            return "standing", 0.5

class EnhancedCCTVProcessor:
    """Enhanced CCTV processing with all improvements"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model = YOLO('yolo11n.pt').to(self.device)
        except:
            self.model = YOLO('yolov8n.pt').to(self.device)
            
        self.tracker = EnhancedTracker()
        self.pose_detector = EnhancedPoseDetector()
        self.hand_detector = HandDetector()
        self.coin_tracker = CoinDetectionTracker()
        self.regions = []
        self.region_analytics = {}
        self.frame_count = 0
        self.start_time = time.time()
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.person_logs = {}  # Store detailed person logs for CSV export
        
        self.process_every_n_frames = 1
        self.pose_detection_interval = 3
        self.hand_detection_interval = 1  # Process hand detection every frame for robustness
        
    def set_regions(self, regions_data: List[Dict], original_width: int, original_height: int, 
                   display_width: int, display_height: int):
        """Set regions with proper coordinate scaling"""
        self.regions = []
        self.region_analytics = {}
        
        # Calculate scaling factors
        self.scale_x = original_width / display_width
        self.scale_y = original_height / display_height
        
        for i, region_data in enumerate(regions_data):
            # Scale points back to original video dimensions
            scaled_points = []
            for point in region_data['points']:
                scaled_x = int(point[0] * self.scale_x)
                scaled_y = int(point[1] * self.scale_y)
                scaled_points.append((scaled_x, scaled_y))
            
            region = Region(
                id=i,
                name=region_data.get('name', f'Region {i+1}'),
                points=scaled_points,
                color=region_data.get('color', (0, 255, 0))
            )
            self.regions.append(region)
            self.region_analytics[i] = {
                'entries': 0,
                'exits': 0,
                'max_occupancy': 0,
                'total_time': 0,
                'entry_log': []
            }
    
    def _point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
        """Check if point is inside polygon"""
        if len(polygon) < 3:
            return False
            
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
            
        return inside
    
    def _check_region_crossing(self, person_id: int, tracker: PersonTracker, current_time: float):
        """Check and log region crossings"""
        if len(tracker.position_history) < 2:
            return
            
        current_pos = tracker.position_history[-1]
        prev_pos = tracker.position_history[-2]
        
        for region in self.regions:
            current_inside = self._point_in_polygon(current_pos, region.points)
            prev_inside = self._point_in_polygon(prev_pos, region.points)
            
            if current_inside != prev_inside:
                if current_inside:  # Entered region
                    region.entry_count += 1
                    region.current_occupants.add(person_id)
                    if person_id not in tracker.region_entries:
                        tracker.region_entries[region.id] = []
                    tracker.region_entries[region.id].append(current_time)
                    
                    entry_log = {
                        'person_id': person_id,
                        'timestamp': current_time,
                        'action': 'entered',
                        'region_name': region.name
                    }
                    region.entry_log.append(entry_log)
                    self.region_analytics[region.id]['entries'] += 1
                    self.region_analytics[region.id]['entry_log'].append(entry_log)
                    
                else:  # Exited region
                    region.exit_count += 1
                    region.current_occupants.discard(person_id)
                    
                    exit_log = {
                        'person_id': person_id,
                        'timestamp': current_time,
                        'action': 'exited',
                        'region_name': region.name
                    }
                    region.entry_log.append(exit_log)
                    self.region_analytics[region.id]['exits'] += 1
                    self.region_analytics[region.id]['entry_log'].append(exit_log)
                
                if current_inside:
                    tracker.current_regions.add(region.id)
                else:
                    tracker.current_regions.discard(region.id)
    
    def process_video(self, video_path: str, output_path: str, 
                     regions: List[Dict] = None, progress_callback=None,
                     original_width: int = None, original_height: int = None,
                     display_width: int = None, display_height: int = None) -> Dict:
        """Process video with enhanced features and progress tracking"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use actual video dimensions if not provided
        if original_width is None:
            original_width = width
        if original_height is None:
            original_height = height
        if display_width is None:
            display_width = width
        if display_height is None:
            display_height = height
        
        if regions:
            self.set_regions(regions, original_width, original_height, display_width, display_height)
        
        # Video writer setup with better codec handling
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            avi_path = output_path.replace('.mp4', '.avi')
            out = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))
            output_path = avi_path
        
        self.frame_count = 0
        self.start_time = time.time()
        pose_stats = defaultdict(int)
        processing_start = time.time()
        self.person_logs = {}
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                current_time = self.frame_count / fps
                
                # Progress callback
                if progress_callback and self.frame_count % 30 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    elapsed = time.time() - processing_start
                    eta = (elapsed / self.frame_count) * (total_frames - self.frame_count)
                    progress_callback(progress, eta)
                
                # YOLO detection
                try:
                    with torch.no_grad():
                        results = self.model(frame, classes=[0], conf=0.4, verbose=False)
                except Exception as e:
                    logger.warning(f"Detection error: {e}")
                    out.write(frame)
                    continue
                
                detections = []
                if results and len(results[0].boxes) > 0:
                    try:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confs = results[0].boxes.conf.cpu().numpy()
                        
                        for box, conf in zip(boxes, confs):
                            if conf > 0.5:
                                bbox = [int(x) for x in box]
                                if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                                    if area > 2000:
                                        detections.append((bbox, conf))
                    except Exception as e:
                        logger.warning(f"Detection processing error: {e}")
                
                # Update tracker
                try:
                    trackers = self.tracker.update(detections)
                except Exception as e:
                    logger.warning(f"Tracking error: {e}")
                    trackers = {}
                
                # Process each tracker
                for person_id, tracker in trackers.items():
                    if tracker.stable_detections > 3:
                        # Log person data for CSV export
                        if person_id not in self.person_logs:
                            self.person_logs[person_id] = {
                                'person_id': person_id,
                                'entry_time': tracker.entry_time,
                                'exit_time': None,
                                'total_duration': 0,
                                'poses': defaultdict(int),
                                'regions_visited': set()
                            }
                        
                        self.person_logs[person_id]['exit_time'] = tracker.last_seen_time
                        self.person_logs[person_id]['total_duration'] = tracker.last_seen_time - tracker.entry_time
                        self.person_logs[person_id]['poses'][tracker.current_pose] += 1
                        self.person_logs[person_id]['regions_visited'].update(tracker.current_regions)
                        
                        bbox = tracker.bbox_history[-1]
                        x1, y1, x2, y2 = bbox
                        
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        
                        if x2 > x1 and y2 > y1:
                            # Pose detection
                            if self.frame_count % self.pose_detection_interval == 0:
                                person_crop = frame[y1:y2, x1:x2]
                                pose = self.pose_detector.detect_pose(person_crop, person_id)
                                tracker.current_pose = pose
                                pose_stats[pose] += 1
                            
                            # Hand detection and coin insertion tracking
                            if self.frame_count % self.hand_detection_interval == 0:
                                person_crop = frame[y1:y2, x1:x2]
                                hands = self.hand_detector.detect_hands(person_crop)
                                
                                # Check for coin insertion using all detected hands
                                self.coin_tracker.check_coin_insertion(
                                    person_id, hands, (x1, y1, x2, y2), self.regions, current_time
                                )
                                
                                # Draw detected hands on frame
                                self._draw_hand_points(frame, (x1, y1, x2, y2), hands)
                            
                            # Region crossing detection
                            self._check_region_crossing(person_id, tracker, current_time)
                            
                            # Draw enhanced annotations
                            self._draw_enhanced_annotations(frame, tracker, (x1, y1, x2, y2))
                
                # Draw regions
                self._draw_regions(frame)
                
                # Draw statistics
                self._draw_statistics(frame, pose_stats, len([t for t in trackers.values() if t.stable_detections > 3]))
                
                out.write(frame)
                
        except Exception as e:
            logger.error(f"Processing error: {e}")
            raise
        finally:
            cap.release()
            out.release()
        
        # Finalize person logs
        for person_id, tracker in self.tracker.trackers.items():
            if person_id in self.person_logs and tracker.exit_time:
                self.person_logs[person_id]['exit_time'] = tracker.exit_time
        
        # Generate comprehensive report
        processing_time = time.time() - processing_start
        return self._generate_comprehensive_report(processing_time, total_frames, fps, pose_stats)
    
    def _draw_hand_points(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], hands: List[Dict]):
        """Draw multiple hands as points with optional handedness label"""
        if not hands:
            return
        for hand in hands:
            center = hand.get('center') if isinstance(hand, dict) else None
            label = hand.get('label') if isinstance(hand, dict) else None
            if center:
                abs_x = bbox[0] + int(center[0])
                abs_y = bbox[1] + int(center[1])
                # Yellow filled dot with red ring
                cv2.circle(frame, (abs_x, abs_y), 5, (0, 255, 255), -1)
                cv2.circle(frame, (abs_x, abs_y), 8, (0, 0, 255), 2)
                if label:
                    cv2.putText(frame, label, (abs_x + 6, abs_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    def _draw_enhanced_annotations(self, frame: np.ndarray, tracker: PersonTracker, bbox: Tuple[int, int, int, int]):
        """Draw enhanced annotations with colored boxes and thin borders"""
        x1, y1, x2, y2 = bbox
        
        # Use tracker's unique color
        color = tracker.color
        
        if tracker.is_occluded:
            color = tuple(int(c * 0.5) for c in color)
        
        # Draw thin bounding box
        thickness = 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        label = f"ID:{tracker.person_id} {tracker.current_pose}"
        if tracker.is_occluded:
            label += " (O)"
        
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_height - 8), (x1 + text_width + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
        # Draw path prediction for occluded persons
        if tracker.is_occluded and tracker.path_prediction:
            for i, pred_pos in enumerate(tracker.path_prediction[:5]):
                pred_x, pred_y = int(pred_pos[0]), int(pred_pos[1])
                alpha = 0.3 + (i * 0.1)
                pred_color = tuple(int(c * alpha) for c in color)
                cv2.circle(frame, (pred_x, pred_y), 2, pred_color, -1)
        
        # Draw movement trail with same color as bounding box
        if len(tracker.position_history) > 1:
            points = list(tracker.position_history)[-10:]
            for i in range(1, len(points)):
                pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                alpha = i / len(points)
                trail_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, pt1, pt2, trail_color, 1)
    
    def _draw_regions(self, frame: np.ndarray):
        """Draw regions with analytics"""
        for region in self.regions:
            if len(region.points) >= 3:
                pts = np.array(region.points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, region.color, 2)
                
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], region.color)
                cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
                
                if len(region.points) > 0:
                    info_x, info_y = region.points[0]
                    info_text = f"{region.name}: {region.entry_count} entries, {len(region.current_occupants)} current"
                    cv2.putText(frame, info_text, (info_x, info_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, region.color, 2)
    
    def _draw_statistics(self, frame: np.ndarray, pose_stats: Dict, active_tracks: int):
        """Draw comprehensive statistics"""
        y = 30
        cv2.putText(frame, f"Active Tracks: {active_tracks}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y += 30
        for pose, count in pose_stats.items():
            cv2.putText(frame, f"{pose}: {count}", 
                       (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 25
        
        if self.regions:
            y += 10
            cv2.putText(frame, "Regions:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 25
            for region in self.regions:
                cv2.putText(frame, f"{region.name}: {region.entry_count} entries", 
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, region.color, 2)
                y += 20
    
    def _generate_comprehensive_report(self, processing_time: float, total_frames: int, 
                                     fps: int, pose_stats: Dict) -> Dict:
        """Generate comprehensive analysis report"""
        
        region_summary = {}
        for region in self.regions:
            region_summary[region.id] = {
                'name': region.name,
                'entries': region.entry_count,
                'exits': region.exit_count,
                'max_occupancy': max(len(region.current_occupants), 
                                   self.region_analytics[region.id]['max_occupancy']),
                'entry_log': region.entry_log,
                'current_occupants': list(region.current_occupants)  # Convert set to list
            }
        
        active_trackers = {pid: tracker for pid, tracker in self.tracker.trackers.items()
                          if tracker.stable_detections > 5}
        
        # Prepare person tracking data
        person_data = []
        for person_id, tracker in active_trackers.items():
            if person_id in self.person_logs:
                log_data = self.person_logs[person_id]
                person_data.append({
                    'person_id': person_id,
                    'login_time': time.strftime('%H:%M:%S', time.gmtime(log_data['entry_time'])),
                    'logout_time': time.strftime('%H:%M:%S', time.gmtime(log_data['exit_time'])) if log_data['exit_time'] else 'Still Active',
                    'total_duration': log_data['total_duration'],
                    'stable_detections': tracker.stable_detections,
                    'final_pose': tracker.current_pose,
                    'regions_visited': list(tracker.current_regions)  # Convert set to list
                })
        
        total_persons = len(active_trackers)
        avg_confidence = np.mean([np.mean(list(t.confidence_history)) 
                                for t in active_trackers.values() if t.confidence_history])
        
        return {
            'video_info': {
                'total_frames': total_frames,
                'fps': fps,
                'duration': total_frames / fps,
                'processing_time': processing_time,
                'processing_fps': self.frame_count / processing_time if processing_time > 0 else 0
            },
            'tracking_summary': {
                'total_persons_detected': total_persons,
                'active_trackers': len(active_trackers),
                'average_confidence': float(avg_confidence) if not np.isnan(avg_confidence) else 0.0,
                'stable_detections': sum(t.stable_detections for t in active_trackers.values())
            },
            'pose_statistics': dict(pose_stats),
            'region_analytics': region_summary,
            'person_data': person_data,  # Add person tracking data
            'coin_insertions': {
                'total_coins': len(self.coin_tracker.coin_insertions),
                'events': self.coin_tracker.coin_insertions
            },
            'performance': {
                'gpu_used': self.device == 'cuda',
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                'model_used': 'yolo11n.pt' if 'yolo11n.pt' in str(self.model) else 'yolov8n.pt'
            }
        }
    
    def generate_csv_report(self, video_id: str) -> str:
        """Generate CSV report with person login/logout times"""
        csv_path = os.path.join(app.config['RESULTS_FOLDER'], f"person_logs_{video_id}.csv")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Person ID', 'Entry Time', 'Exit Time', 'Duration (seconds)', 
                         'Primary Pose', 'Regions Visited', 'Total Detections']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for person_id, log_data in self.person_logs.items():
                entry_time = time.strftime('%H:%M:%S', time.gmtime(log_data['entry_time']))
                exit_time = time.strftime('%H:%M:%S', time.gmtime(log_data['exit_time'])) if log_data['exit_time'] else 'Still Active'
                duration = f"{log_data['total_duration']:.2f}"
                
                # Get primary pose
                poses = log_data['poses']
                primary_pose = max(poses, key=poses.get) if poses else 'unknown'
                
                # Get regions visited
                regions_visited = ', '.join([f"Region {r}" for r in log_data['regions_visited']]) if log_data['regions_visited'] else 'None'
                
                # Get total detections
                total_detections = sum(poses.values())
                
                writer.writerow({
                    'Person ID': person_id,
                    'Entry Time': entry_time,
                    'Exit Time': exit_time,
                    'Duration (seconds)': duration,
                    'Primary Pose': primary_pose,
                    'Regions Visited': regions_visited,
                    'Total Detections': total_detections
                })
        
        return csv_path

# Global processor instance
processor = EnhancedCCTVProcessor()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload_temp', methods=['POST'])
def upload_temp():
    """Temporarily upload video for first frame extraction"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = secure_filename(file.filename)
    video_id = str(uuid.uuid4())
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}_{filename}")
    file.save(video_path)
    
    return jsonify({
        'success': True,
        'video_id': video_id
    })

@app.route('/process', methods=['POST'])
def process_video():
    """Handle video processing with progress tracking"""
    try:
        video_id = request.form.get('video_id')
        if not video_id:
            return jsonify({'error': 'No video ID provided'}), 400
        
        regions_data = request.form.get('regions', '[]')
        original_width = int(request.form.get('original_width', 0))
        original_height = int(request.form.get('original_height', 0))
        display_width = int(request.form.get('display_width', 0))
        display_height = int(request.form.get('display_height', 0))
        
        try:
            regions = json.loads(regions_data)
        except:
            regions = []
        
        video_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(video_id)]
        if not video_files:
            return jsonify({'error': 'Video file not found'}), 404
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_files[0])
        
        processing_status[video_id] = {
            'progress': 0,
            'eta': 0,
            'status': 'processing',
            'error': None
        }
        
        output_path = os.path.join(app.config['RESULTS_FOLDER'], f"processed_{video_id}.mp4")
        
        def progress_callback(progress, eta):
            processing_status[video_id]['progress'] = progress
            processing_status[video_id]['eta'] = eta
        
        stats = processor.process_video(
            video_path, output_path, regions, progress_callback,
            original_width, original_height, display_width, display_height
        )
        
        processing_status[video_id]['progress'] = 100
        processing_status[video_id]['status'] = 'completed'
        
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f"results_{video_id}.json")
        with open(results_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        # Generate CSV report
        csv_path = processor.generate_csv_report(video_id)
        
        # Generate coin insertion CSV report
        coin_csv_path = processor.coin_tracker.export_csv(video_id)
        
        return jsonify({
            'success': True,
            'video_id': video_id,
            'stats': stats,
            'output_video': f"processed_{video_id}.mp4",
            'csv_report': f"person_logs_{video_id}.csv",
            'coin_csv_report': f"coin_insertions_{video_id}.csv"
        })
    
    except Exception as e:
        logger.error(f"Processing error: {e}")
        if video_id in processing_status:
            processing_status[video_id]['status'] = 'error'
            processing_status[video_id]['error'] = str(e)
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/progress/<video_id>')
def get_progress(video_id):
    """Get processing progress"""
    if video_id in processing_status:
        return jsonify(processing_status[video_id])
    return jsonify({'progress': 0, 'eta': 0, 'status': 'unknown'})

@app.route('/get_first_frame/<video_id>')
def get_first_frame(video_id):
    """Extract and return first frame for region setup"""
    try:
        video_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(video_id)]
        if not video_files:
            return jsonify({'error': 'Video not found'}), 404
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_files[0])
        cap = cv2.VideoCapture(video_path)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return jsonify({'error': 'Could not read first frame'}), 500
        
        height, width = frame.shape[:2]
        max_width, max_height = 800, 400
        
        display_width, display_height = width, height
        if width > max_width or height > max_height:
            scale = min(max_width/width, max_height/height)
            display_width = int(width * scale)
            display_height = int(height * scale)
            frame = cv2.resize(frame, (display_width, display_height))
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'frame': f"data:image/jpeg;base64,{frame_base64}",
            'width': display_width,
            'height': display_height,
            'original_width': width,
            'original_height': height
        })
        
    except Exception as e:
        return jsonify({'error': f'Frame extraction failed: {str(e)}'}), 500

@app.route('/results/<video_id>')
def results(video_id):
    """Results page"""
    return render_template('results.html', video_id=video_id)

@app.route('/get_results/<video_id>')
def get_results(video_id):
    """Get processing results for a video"""
    try:
        results_file = os.path.join(app.config['RESULTS_FOLDER'], f"results_{video_id}.json")
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            return jsonify(results)
        else:
            return jsonify({
                'pose_statistics': {},
                'region_analytics': {},
                'tracking_summary': {'total_persons_detected': 0}
            })
    
    except Exception as e:
        return jsonify({'error': f'Failed to load results: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed video or CSV report"""
    file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

@app.route('/video/<video_id>')
def serve_video(video_id):
    """Serve video for streaming playback"""
    filename = f"processed_{video_id}.mp4"
    file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    
    # Try .avi if .mp4 doesn't exist
    if not os.path.exists(file_path):
        filename = f"processed_{video_id}.avi"
        file_path = os.path.join(app.config['RESULTS_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return "File not found", 404

    def generate():
        with open(file_path, 'rb') as f:
            data = f.read(1024)
            while data:
                yield data
                data = f.read(1024)

    return Response(generate(), 
                   mimetype='video/mp4' if filename.endswith('.mp4') else 'video/avi',
                   headers={'Content-Disposition': f'inline; filename={filename}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)