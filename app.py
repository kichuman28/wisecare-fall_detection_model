from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = YOLO('yolo11s-pose.pt')

@app.route('/detect_fall', methods=['POST'])
def detect_fall():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the uploaded video
    filename = secure_filename(video_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(filepath)
    
    try:
        # Process video with YOLO model
        results = model(filepath, save=True)  # Enable saving of detection results
        
        fall_detected = False
        fall_frames = []
        
        # Process each frame's results
        for r in results:
            # Your fall detection logic here
            # This is a placeholder - implement your actual fall detection criteria
            keypoints = r.keypoints.data  # Get pose keypoints
            if len(keypoints) > 0:
                # Example: Check if any person's head position is close to ground level
                # Implement your actual fall detection logic here
                fall_detected = True
                fall_frames.append(r.orig_img)  # Store the frame where fall was detected
        
        if fall_detected and fall_frames:
            # Save the fall detection video
            fall_video_name = f"fall_detection_{filename}"
            fall_video_path = os.path.join(app.config['UPLOAD_FOLDER'], fall_video_name)
            
            # Create video from frames
            height, width = fall_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(fall_video_path, fourcc, 30.0, (width, height))
            
            for frame in fall_frames:
                out.write(frame)
            out.release()
            
            # Read the video file as bytes to send in response
            with open(fall_video_path, 'rb') as f:
                fall_video_bytes = f.read()
            
            # Clean up
            os.remove(filepath)
            os.remove(fall_video_path)
            
            # Return both JSON data and video file
            response = {
                'status': 'success',
                'message': 'Fall detected',
                'fall_video': fall_video_bytes.hex()  # Convert bytes to hex string for JSON
            }
            return jsonify(response)
        
        # Clean up if no fall detected
        os.remove(filepath)
        return jsonify({
            'status': 'success',
            'message': 'No fall detected'
        })
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000) 