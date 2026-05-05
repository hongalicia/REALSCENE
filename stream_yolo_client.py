import zmq
import cv2
import numpy as np
import json

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:9999")
socket.subscribe("annotated_frame")  # or "detections" for JSON data

while True:
    topic, data = socket.recv_multipart()
    if topic == b"annotated_frame":
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        cv2.imshow("ZMQ Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    elif topic == b"detections":
        detections = json.loads(data.decode())
        print(detections)