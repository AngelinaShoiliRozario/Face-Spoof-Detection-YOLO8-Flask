from flask import Flask, render_template, Response, request, session
from werkzeug.serving import run_simple
from flask_socketio import SocketIO, emit, join_room, leave_room
import cv2
import math
import time
import cvzone
import threading
from ultralytics import YOLO

app = Flask(__name__)
print(cv2.__version__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

cap = cv2.VideoCapture(1)
model = YOLO("./models/best.pt")


# cap.set(3, 640)
# cap.set(4, 480)

# below 60 spoof should go for manual testing
# count frame to 30

# def detect_spoof():
#     confidence = 0.9
#
#     # cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video
#
#     classNames = ["real", "fake"]
#
#     while True:
#         success, img = cap.read()
#         ret, buffer = cv2.imencode('.jpg', img)  # image to buffer
#         frame = buffer.tobytes()
#
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#         results = model(img, stream=True,verbose=False)
#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 # Bounding Box
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
#                 w, h = x2 - x1, y2 - y1
#                 # Confidence
#                 conf = math.ceil((box.conf[0] * 100)) / 100
#                 print('confidence ',conf)
#                 # Class Name
#                 cls = int(box.cls[0])
#
#                 print('class ', cls)
#                 # if conf > confidence:
#                     # print('class ', cls)

# Define a function to perform prediction
def perform_prediction(cap, model, confidence):
    while True:
        success, img = cap.read()
        results = model(img, stream=True, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                print('confidence ',conf)
                # Class Name
                cls = int(box.cls[0])
                print('class ',cls)
        # time.sleep(0.1)  # Adjust the delay as needed

def detect_spoof():
    confidence = 0.9
    cap.set(3, 640)
    cap.set(4, 480)

    prediction_thread = threading.Thread(target=perform_prediction, args=(cap, model, confidence))
    prediction_thread.daemon = True
    prediction_thread.start()

    while True:
        success, img = cap.read()
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# on connection make a room 
@socketio.on('connect')
def handle_connect():
    session['room'] = request.sid
    join_room(session['room'])
    print(f"User connected with session ID: {session['room']}")

# on disconnection remove a room 
@socketio.on('disconnect')
def handle_disconnect():
    print(f"User disconnected with session ID: {session['room']}")
    leave_room(session['room'])

@socketio.on('message')
def handle_message(data):
    print('message received by the server. MSG: ', data)
    message = f'Hello Client {session["room"]}!'
    socketio.emit('response', message, room=session['room'])

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(detect_spoof(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    socketio.run(app)