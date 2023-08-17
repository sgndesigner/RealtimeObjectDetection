import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')

# Load COCO class labels
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Load image and perform object detection
cap = cv2.VideoCapture(0)  # Use webcam (change to video file if needed)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:  # Adjust confidence threshold as needed
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # Apply non-maximum suppression
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = (0, 255, 0)  # Green color for bounding boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Real-Time Object Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
