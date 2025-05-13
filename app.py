from flask import Flask, render_template, request, redirect, url_for, flash, session, Response, jsonify
from functools import wraps
import cv2
from ultralytics import YOLO
import numpy as np
import os
import torch

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key')  # Better secret key handling

# Temporary user storage (in-memory)
users = {}

# Initialize the YOLO model
model = YOLO('best.pt')

# Global variables for camera
camera = None
conf_threshold = 0.5

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        # Set camera properties for better quality
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
    return camera

def generate_frames():
    camera = get_camera()
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Run YOLOv8 inference
            results = model(frame, conf=conf_threshold)
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            # Convert the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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
    global camera
    if camera is not None:
        camera.release()
        camera = None
    session.pop('username', None)
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
