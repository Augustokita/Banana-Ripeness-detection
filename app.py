from flask import Flask, render_template, request, redirect, url_for, flash, session, Response, jsonify
from functools import wraps
import cv2
from ultralytics import YOLO
import numpy as np
import os
import torch
import base64

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key')  # Better secret key handling

# Temporary user storage (in-memory)
users = {}

# Initialize the YOLO model
try:
    model = YOLO('best.pt')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Global variables for camera
conf_threshold = 0.5

def process_frame(frame_data):
    try:
        if model is None:
            print("Model not loaded")
            return None

        # Decode base64 image
        nparr = np.frombuffer(base64.b64decode(frame_data.split(',')[1]), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("Failed to decode frame")
            return None

        # Run YOLOv8 inference with specific parameters
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Get the first result
        result = results[0]
        
        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        # Draw boxes and labels
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get confidence and class
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{result.names[cls]} {conf:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return None

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
@login_required
def home():
    return render_template('home.html', username=session['username'])

@app.route('/process_frame', methods=['POST'])
@login_required
def process_frame_route():
    try:
        frame_data = request.json.get('frame')
        if not frame_data:
            return jsonify({'error': 'No frame data received'}), 400
        
        processed_frame = process_frame(frame_data)
        if processed_frame:
            return jsonify({'frame': processed_frame})
        else:
            return jsonify({'error': 'Error processing frame'}), 500
    except Exception as e:
        print(f"Error in process_frame_route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/update_confidence', methods=['POST'])
@login_required
def update_confidence():
    global conf_threshold
    data = request.get_json()
    if 'confidence' in data:
        conf_threshold = float(data['confidence'])
        return jsonify({'status': 'success', 'confidence': conf_threshold})
    return jsonify({'status': 'error', 'message': 'Invalid confidence value'}), 400

@app.route('/signin', methods=['POST'])
def signin():
    try:
        username = request.form.get('username')
        password = request.form.get('password')

        if not username or not password:
            flash("Please fill in all fields", "error")
            return redirect(url_for('index'))

        user = users.get(username)
        
        if user and user['password'] == password:  # In a real app, use proper password hashing
            session['username'] = username
            flash("Login Successful!", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid username or password", "error")
            return redirect(url_for('index'))
    except Exception as e:
        flash("An error occurred during login. Please try again.", "error")
        return redirect(url_for('index'))

@app.route('/signup', methods=['POST'])
def signup():
    try:
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if not username or not email or not password:
            flash("Please fill in all fields", "error")
            return redirect(url_for('index'))

        # Check if username or email already exists
        if username in users:
            flash("Username already taken", "error")
            return redirect(url_for('index'))
        
        if any(user['email'] == email for user in users.values()):
            flash("Email already registered", "error")
            return redirect(url_for('index'))

        # Create new user
        users[username] = {
            'email': email,
            'password': password  # In a real app, use proper password hashing
        }
        
        flash("Sign-up successful! Please log in.", "success")
        return redirect(url_for('index'))
    except Exception as e:
        flash("An error occurred during registration. Please try again.", "error")
        return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
