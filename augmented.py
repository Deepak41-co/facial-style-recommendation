import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import hashlib
import os
app = Flask(__name__)
CORS(app)
from flask import send_from_directory


class FaceShapeDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,  # Lowered for better detection
            min_tracking_confidence=0.5
        )
        
        # All 7 face shapes including Triangle
        self.face_shapes = {
            'oval': 'Oval',
            'round': 'Round', 
            'square': 'Square',
            'heart': 'Heart',
            'diamond': 'Diamond',
            'oblong': 'Oblong',
            'triangle': 'Triangle'
        }
        
        self.analysis_cache = {}
        
        # Updated landmark indices for better accuracy
        self.landmark_indices = {
            'forehead_left': 54,
            'forehead_right': 284,
            'forehead_center': 10,
            'chin': 152,
            'jaw_left': 172,
            'jaw_right': 397,
            'cheek_left': 234,
            'cheek_right': 454,
            'nose_tip': 1,
            'left_eye_left': 33,
            'right_eye_right': 263,
            'jaw_mid_left': 136,
            'jaw_mid_right': 365,
            'left_eyebrow_upper': 65,
            'right_eyebrow_upper': 295
        }

    def get_image_hash(self, image):
        return hashlib.md5(image.tobytes()).hexdigest()

    def get_face_landmarks(self, image):
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None
                
            landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            landmark_points = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmark_points.append((x, y))
                
            return landmark_points
            
        except Exception as e:
            print(f"Error in landmark detection: {e}")
            return None

    def calculate_face_ratios(self, landmarks):
        if not landmarks:
            return None
        
        try:
            idx = self.landmark_indices
            
            # Calculate face width at different levels
            forehead_width = np.linalg.norm(
                np.array(landmarks[idx['forehead_left']]) - 
                np.array(landmarks[idx['forehead_right']])
            )
            
            cheek_width = np.linalg.norm(
                np.array(landmarks[idx['cheek_left']]) - 
                np.array(landmarks[idx['cheek_right']])
            )
            
            jaw_width = np.linalg.norm(
                np.array(landmarks[idx['jaw_left']]) - 
                np.array(landmarks[idx['jaw_right']])
            )
            
            # Face length
            face_length = np.linalg.norm(
                np.array(landmarks[idx['forehead_center']]) - 
                np.array(landmarks[idx['chin']])
            )
            
            # Jaw proportions
            jaw_mid_width = np.linalg.norm(
                np.array(landmarks[idx['jaw_mid_left']]) - 
                np.array(landmarks[idx['jaw_mid_right']])
            )
            
            # Forehead to jaw ratio
            forehead_to_jaw_ratio = forehead_width / jaw_width if jaw_width > 0 else 1.0
            
            # Key ratios for classification
            length_width_ratio = face_length / max(forehead_width, cheek_width, jaw_width)
            jaw_forehead_ratio = jaw_width / forehead_width if forehead_width > 0 else 1.0
            cheek_jaw_ratio = cheek_width / jaw_width if jaw_width > 0 else 1.0
            jaw_progression = jaw_width / jaw_mid_width if jaw_mid_width > 0 else 1.0
            
            return {
                'length_width_ratio': length_width_ratio,
                'jaw_forehead_ratio': jaw_forehead_ratio,
                'cheek_jaw_ratio': cheek_jaw_ratio,
                'jaw_progression': jaw_progression,
                'forehead_to_jaw_ratio': forehead_to_jaw_ratio,
                'face_length': face_length,
                'jaw_width': jaw_width,
                'forehead_width': forehead_width,
                'cheek_width': cheek_width,
                'jaw_mid_width': jaw_mid_width
            }
            
        except Exception as e:
            print(f"Error calculating ratios: {e}")
            return None

    def classify_face_shape(self, ratios):
        """Improved classification for all 7 face shapes"""
        if not ratios:
            return "Unknown"
            
        try:
            lw_ratio = ratios['length_width_ratio']
            jf_ratio = ratios['jaw_forehead_ratio']
            cj_ratio = ratios['cheek_jaw_ratio']
            ftj_ratio = ratios['forehead_to_jaw_ratio']
            
            print(f"Ratios - L/W: {lw_ratio:.2f}, Jaw/Forehead: {jf_ratio:.2f}, Cheek/Jaw: {cj_ratio:.2f}, Forehead/Jaw: {ftj_ratio:.2f}")
            
            # Improved classification logic
            if lw_ratio > 1.5:
                # Long faces
                if jf_ratio > 0.95:
                    shape = 'oblong'
                else:
                    shape = 'heart'
                    
            elif lw_ratio < 1.1:
                # Short/wide faces
                if abs(jf_ratio - 1.0) < 0.15:
                    shape = 'round'
                else:
                    shape = 'square'
                    
            else:
                # Medium length faces
                if cj_ratio > 1.15:
                    shape = 'diamond'
                elif jf_ratio > 1.1:
                    shape = 'triangle'
                elif ftj_ratio > 1.1:
                    shape = 'heart'
                else:
                    # Balanced proportions = Oval
                    shape = 'oval'
            
            # Additional validation checks
            if shape == 'diamond' and cj_ratio < 1.1:
                shape = 'oval'
            elif shape == 'triangle' and jf_ratio < 1.05:
                shape = 'square'
            elif shape == 'heart' and ftj_ratio < 1.05:
                shape = 'oval'
                    
            print(f"Detected shape: {shape}")
            return self.face_shapes.get(shape, 'Unknown')
            
        except Exception as e:
            print(f"Error in classification: {e}")
            return "Unknown"

    def calculate_confidence(self, ratios, detected_shape):
        """Calculate confidence based on how well ratios match ideal ranges"""
        # Expanded ideal ranges for better accuracy
        ideal_ranges = {
            'Oval': {
                'lw_min': 1.3, 'lw_max': 1.5, 
                'jf_min': 0.85, 'jf_max': 1.05,
                'cj_min': 0.9, 'cj_max': 1.1
            },
            'Round': {
                'lw_min': 1.0, 'lw_max': 1.2, 
                'jf_min': 0.9, 'jf_max': 1.1,
                'cj_min': 0.9, 'cj_max': 1.1
            },
            'Square': {
                'lw_min': 1.1, 'lw_max': 1.3, 
                'jf_min': 0.95, 'jf_max': 1.15,
                'cj_min': 0.9, 'cj_max': 1.1
            },
            'Heart': {
                'lw_min': 1.4, 'lw_max': 1.7, 
                'jf_min': 0.7, 'jf_max': 0.9,
                'cj_min': 1.0, 'cj_max': 1.2
            },
            'Diamond': {
                'lw_min': 1.3, 'lw_max': 1.5, 
                'jf_min': 0.8, 'jf_max': 1.0,
                'cj_min': 1.1, 'cj_max': 1.4
            },
            'Oblong': {
                'lw_min': 1.5, 'lw_max': 1.8, 
                'jf_min': 0.9, 'jf_max': 1.1,
                'cj_min': 0.9, 'cj_max': 1.1
            },
            'Triangle': {
                'lw_min': 1.2, 'lw_max': 1.4, 
                'jf_min': 1.05, 'jf_max': 1.3,
                'cj_min': 0.8, 'cj_max': 1.0
            }
        }
        
        if detected_shape not in ideal_ranges:
            return 75
        
        ideal = ideal_ranges[detected_shape]
        lw_ratio = ratios['length_width_ratio']
        jf_ratio = ratios['jaw_forehead_ratio']
        cj_ratio = ratios['cheek_jaw_ratio']
        
        # Calculate how close ratios are to ideal
        lw_score = self.calculate_score(lw_ratio, ideal['lw_min'], ideal['lw_max'])
        jf_score = self.calculate_score(jf_ratio, ideal['jf_min'], ideal['jf_max'])
        cj_score = self.calculate_score(cj_ratio, ideal['cj_min'], ideal['cj_max'])
        
        confidence = (lw_score + jf_score + cj_score) / 3
        return min(95, max(65, confidence))

    def calculate_score(self, value, min_val, max_val):
        """Calculate how close a value is to the ideal range"""
        ideal_center = (min_val + max_val) / 2
        ideal_range = max_val - min_val
        
        # Calculate distance from ideal center
        distance = abs(value - ideal_center)
        
        # Convert to score (0-100)
        if distance <= ideal_range / 2:
            score = 100 - (distance / (ideal_range / 2)) * 25
        else:
            score = max(0, 75 - (distance - ideal_range / 2) * 20)
            
        return score

    def analyze_image(self, image, gender="male"):
        image_hash = self.get_image_hash(image)
        
        if image_hash in self.analysis_cache:
            return self.analysis_cache[image_hash]
        
        landmarks = self.get_face_landmarks(image)
        
        if not landmarks:
            return None
        
        ratios = self.calculate_face_ratios(landmarks)
        if not ratios:
            return None
        
        face_shape = self.classify_face_shape(ratios)
        confidence = self.calculate_confidence(ratios, face_shape)
        
        result = {
            'face_shape': face_shape,
            'confidence': round(confidence, 1),
            'measurements': {
                'face_length': round(ratios['face_length'], 1),
                'face_width': round(max(ratios['forehead_width'], ratios['jaw_width']), 1),
                'length_width_ratio': round(ratios['length_width_ratio'], 2),
                'jaw_forehead_ratio': round(ratios['jaw_forehead_ratio'], 2),
                'cheek_jaw_ratio': round(ratios['cheek_jaw_ratio'], 2)
            },
            'debug_ratios': {  # For debugging
                'length_width': round(ratios['length_width_ratio'], 2),
                'jaw_forehead': round(ratios['jaw_forehead_ratio'], 2),
                'cheek_jaw': round(ratios['cheek_jaw_ratio'], 2)
            }
        }
        
        self.analysis_cache[image_hash] = result
        return result



@app.route('/analyze-face', methods=['POST'])
def analyze_face():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image data provided'
            }), 400

        image_data = data['image']
        gender = data.get('gender', 'male')

        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                'success': False,
                'error': 'Invalid image data'
            }), 400

        # Resize image for consistent processing
        height, width = image.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = 800
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))

        detector = FaceShapeDetector()
        result = detector.analyze_image(image, gender)

        if result:
            return jsonify({
                'success': True,
                'data': result
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No face detected in the image'
            })

    except Exception as e:
        print(f"Error in analyze_face: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500
@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'front.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Face Shape API is running"})
if __name__ == '__main__':
    print("🚀 Starting Face Shape Analysis API...")
    print("📍 Backend URL: http://localhost:5000")
    print("📱 Open front.html in your browser to use the application")
    print("⚡ Features: 7 Face Shapes, Advanced Measurements, Personalized Recommendations")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)