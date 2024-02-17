from flask import Flask, render_template, Response, request, session
from werkzeug.serving import run_simple
from flask_socketio import SocketIO, emit, join_room, leave_room
import cv2
import math
import time
import cvzone
import threading
import  mediapipe as mp
import numpy as np
from ultralytics import YOLO
import queue
import os
import base64
import io
from io import StringIO
from PIL import Image




app = Flask(__name__)
print(cv2.__version__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

model = YOLO("./models/best7000.pt")

# Dictionary to store frames for each user
frames_all = {}



result_queue = queue.Queue() # Create a queue for communication between threads


# on connection make a room 
@socketio.on('connect')
def handle_connect():
    user_id = request.sid
    session['room'] = request.sid
    join_room(session['room'])
    frames_all[user_id] = []
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


def pose_detection(images, required_pose, room , conf): 
    print('in pose detection function')
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    for i in range(len(images)):
        image = images[i]
        np_image = np.array(image)
        
        # Convert RGB image to BGR (if necessary)
        if np_image.shape[2] == 3:  # RGB image
            image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        # print(image)
        
        image= cv2.cvtColor(cv2.flip(image,1 ), cv2.COLOR_BGR2RGB)  # convert to grayscale
        image.flags.writeable = False

        results = face_mesh.process(image)

        img_h, img_w, img_c = image.shape

        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    # print(idx)
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x + (img_w/2), lm.y * img_h)
                            node_3d = (lm.x + img_w, lm.y * img_h, lm.z * 3000)

                        x,y = int(lm.x * img_w), int(lm.y * img_h)

                        face_2d.append([x,y])

                        face_3d.append([x,y,lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                    [0, focal_length, img_w/2],
                    [0,0,1]
                ])

                dist_matrix = np.zeros((4,1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                rmat, jac = cv2.Rodrigues(rot_vec)

                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                if y < -10:
                    text = 'left'
                elif y > 10:
                    text = 'right'
                elif x < -10:
                    text = 'down'
                elif x > 10:
                    text = 'up'
                else:
                    text = 'forward'
                
                if(required_pose == text):
                    print(f'in function detected pose {text}')
                    return True





# Define a function to perform prediction ..........returns true for normal and false for spoof
def perform_prediction1(imgs, confidence, room, userId):
    
    
    normal_score = 0
    spoof_score = 0
    for i in range(len(imgs)):
        
        
        OUTPUT_FOLDER = 'output'
        output_path = os.path.join(OUTPUT_FOLDER, f'{i}.jpg')
        imgs[i].save(output_path)
        
        results = model(imgs[i], stream=True, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                print('class ' , cls)
                if(conf > 0.5):
                    if(cls == 0):
                        # return 'normal'
                        print(f'normal user {userId}')   
                        normal_score= normal_score + 1
                    if(cls == 1):
                        # return 'spoof'
                        print(f'spoof user {userId}')
                        spoof_score = spoof_score + 1
                else:
                    print('special case')
    # Open a file in write mode
    with open('database.txt', 'a') as file:
        # Write a line to the file
        file.write(f'{userId} : Spoof Score= {spoof_score} & Normal Score= {normal_score}.\n')

    # if(spoof_score > normal_score):
    #     socketio.emit('spoof_response', 'spoof', room=room)
    # else:
    #     socketio.emit('spoof_response', 'normal', room=room)
 
# @socketio.on('pose_check')
# def handle_pose_check(data):
#     print('need to check for pose. Required Pose: ', data)
#     detected_pose = pose_detection(data)
#     print(detected_pose)
#     socketio.emit('POSE_response', detected_pose, room=session['room'])          


@socketio.on('stream')
def handle_stream(data):
    allImages = []
    user_id = request.sid
    frame_arr = data['image']
    frameCount = data['frameCount']
    requiredPose = data['requiredPose']
    print(f'requeuing pose {requiredPose}')
    # print(frames_all[user_id])
    for i in range(len(frame_arr)):
        image_data = base64.b64decode(frame_arr[i].split(',')[1]) # Decode the base64-encoded image data
        image = Image.open(io.BytesIO(image_data)) # Create a PIL Image object from the decoded image data
        allImages.append(image)
    # print(allImages)
    frames_all[user_id].extend(allImages)
    confidence = 0.85
    room = session['room']

    if(requiredPose != 0):
        print('pose needed')
        print(requiredPose)
        success = pose_detection(allImages, requiredPose, room, confidence)
        if(success== True):
            socketio.emit('pose_in_response', 'sucess', room=room)

            
    # print(frameCount)

    if(requiredPose == 'forward'):
        print('completed the interval')
        all = frames_all[user_id]
        perform_prediction1(all,confidence,room, user_id)
    # print(frames_all[user_id])
    
 

@app.route('/')
def index():
    return render_template('index.html')




if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0',debug=True, ssl_context="adhoc")