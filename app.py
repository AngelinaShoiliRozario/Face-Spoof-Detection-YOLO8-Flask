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

model = YOLO("./models/best.pt")
cap = cv2.VideoCapture(1)

cap.set(3, 640)
cap.set(4, 480)

result_queue = queue.Queue() # Create a queue for communication between threads

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

def pose_detection(required_pose): 
    print('in pose detection function')
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    while cap.isOpened():
        # print('in pose detection function 2')
        success, image = cap.read()
        # start = time.time()
        image= cv2.cvtColor(cv2.flip(image,1 ), cv2.COLOR_BGR2RGB)  # convert to grayscale
        image.flags.writeable = False

        results = face_mesh.process(image)

        # image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
                # print(text)
                if(required_pose == text):
                    print(f'in function detected pose {text}')
                    return True

                # node_3d_projection, jacobian = cv2.projectPoints(node_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                # p1 = ( int(nose_2d[0]), int(nose_2d[1]) )
                # p2 = ( int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))



            # end = time.time()
            # totalTime = end - start
            # fps = 1/totalTime
            # print("FPS ", fps)

            

            # mp_drawing.draw_landmarks(
            #     image = image,
            #     landmark_list= face_landmarks,
            #     landmark_drawing_spec= drawing_spec,
            #     connection_drawing_spec= drawing_spec

            # )
        # cv2.imshow('HeadPose', image)
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break;
    cap.release()




# Define a function to perform prediction ..........returns true for normal and false for spoof
def perform_prediction(img, confidence, room):
    countFrame = 20
    spoof_score = 0
    normal_score = 0
    while countFrame > 0:
        countFrame -= 1
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
                # print('confidence ',conf)
                # Class Name
                cls = int(box.cls[0])
                if(cls == 0):
                    print('normal')
                    return 'normal'
                    normal_score+=1
                if(cls == 1):
                    print('spoof')
                    return 'spoof'

                    spoof_score+-1
                
                
                # print('class ',cls)
        # time.sleep(0.1)  # Adjust the delay as needed
    # if(normal_score > spoof_score):
    #     socketio.emit('spoof_response', 'normal', room=room)
    # else:
    #     socketio.emit('spoof_response', 'spoof', room=room)

def detect_spoof():
   

   


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

@socketio.on('pose_check')
def handle_pose_check(data):
    print('need to check for pose. Required Pose: ', data)
    detected_pose = pose_detection(data)
    print(detected_pose)
    socketio.emit('POSE_response', detected_pose, room=session['room'])

@socketio.on('spoof_check')
def handle_spoof_check(data):
    print('need to check for spoof.')
    confidence = 0.9
    room=session['room']
    print(room)
    prediction_thread = threading.Thread(target=perform_prediction, args=(cap, model, confidence, room))
    prediction_thread.daemon = True
    prediction_thread.start()



# Define a function to perform prediction ..........returns true for normal and false for spoof
def perform_prediction1(imgs, confidence, room):
    spoof_score = 0
    normal_score = 0
    
    
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
                if(conf >= confidence):
                    if(cls == 0):
                        # return 'normal'
                        # print('normal')   
                        normal_score+=1
                    if(cls == 1):
                        # return 'spoof'
                        # print('spoof')
                        spoof_score+-1
    
    if(spoof_score > normal_score):
        socketio.emit('spoof_response', 'spoof', room=room)
    else:
        socketio.emit('spoof_response', 'normal', room=room)
 
            

allResults = []
@socketio.on('stream')
def handle_stream(data):
    
    allImages = []
    frame_arr = data['image']
    # count = data['count']
    for i in range(len(frame_arr)):
        image_data = base64.b64decode(frame_arr[i].split(',')[1]) # Decode the base64-encoded image data
        image = Image.open(io.BytesIO(image_data)) # Create a PIL Image object from the decoded image data
        allImages.append(image)

    print('need to check for spoof.')
    confidence = 0.85
    room=session['room']
    # print(room)
    # result = perform_prediction(image, confidence, room)
    # allResults.append(result)
    print('all results')
    print(allImages)

    
    # if(count == 2):
    perform_prediction1(allImages, confidence, room)
        
    
    # prediction_thread = threading.Thread(target=, args=(image, confidence, room))
    # prediction_thread.daemon = True
    # prediction_thread.start()

    # # Save the image to the backend
    # OUTPUT_FOLDER = 'output'
    # output_path = os.path.join(OUTPUT_FOLDER, f'{count}.jpg')
    # image.save(output_path)

   



































@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(detect_spoof(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0',debug=True)