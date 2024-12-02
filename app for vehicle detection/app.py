import os
from flask import Flask, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
import cv2
import torch
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import threading

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load YOLOv5 model for vehicle detection
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# Load custom model for traffic sign detection
traffic_sign_model = load_model('model.h5')

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Traffic sign label mappings
label_mapping = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

# To handle the video capture in a separate thread and stop it gracefully
stop_video_flag = threading.Event()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_traffic_sign(filepath):
    # Preprocess image for traffic sign model
    img = image.load_img(filepath, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict traffic sign
    prediction = traffic_sign_model.predict(img_array)
    predicted_label = label_mapping.get(np.argmax(prediction, axis=1)[0], "Unknown")

    return predicted_label

def predict_vehicle(filepath):
    # Load image and predict using YOLOv5
    img = Image.open(filepath)
    results = yolo_model(img)
    processed_img = np.array(results.render()[0])  # Render image with bounding boxes

    # Extract detected vehicle labels
    labels = [yolo_model.names[int(cls)] for *_, cls in results.xyxy[0]]

    return processed_img, labels

def generate_frames():
    cap = cv2.VideoCapture(0)  # Open the camera
    while cap.isOpened() and not stop_video_flag.is_set():
        success, frame = cap.read()
        if not success:
            break

        # Save frame temporarily to make predictions
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_frame.jpg')
        cv2.imwrite(filepath, frame)
        
        # Run vehicle detection on the frame
        vehicle_img, vehicle_labels = predict_vehicle(filepath)
        
        # Run traffic sign prediction
        traffic_sign_prediction = predict_traffic_sign(filepath)

        # Display text on frame for detected traffic sign and vehicles
        cv2.putText(vehicle_img, f"Traffic Sign: {traffic_sign_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(vehicle_img, f"Vehicles: {', '.join(vehicle_labels)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Convert the processed frame (with bounding boxes) to JPEG format
        ret, buffer = cv2.imencode('.jpg', vehicle_img)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Run both models: predict traffic signs and vehicles
    traffic_sign_prediction = predict_traffic_sign(filepath)
    vehicle_img, vehicle_labels = predict_vehicle(filepath)

    # Save the image with YOLOv5 detections for display
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
    Image.fromarray(vehicle_img).save(output_path)

    return render_template('result.html', 
                           filename='result_' + filename, 
                           traffic_sign_prediction=traffic_sign_prediction, 
                           vehicle_labels=vehicle_labels)

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    global stop_video_flag
    stop_video_flag.clear()  # Reset the stop flag to allow video feed
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video')
def stop_video():
    global stop_video_flag
    stop_video_flag.set()  # Set the flag to stop the video feed
    return redirect(url_for('camera'))

if __name__ == '__main__':
    app.run(debug=True)
