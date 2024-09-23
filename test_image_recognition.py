import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Get names of all layers in the YOLO network
layer_names = net.getLayerNames()

# OpenCV 4.x fix: Make sure that getUnconnectedOutLayers returns indices properly
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except AttributeError:
    # Older OpenCV versions might return different formats
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load class names (COCO dataset)
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load the image
test_image_path = 'C:/Users/dell/Desktop/imagerecog/catdog.jpg'
img = cv2.imread(test_image_path)
height, width, channels = img.shape

# Prepare the image for YOLO
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Information for drawing bounding boxes
class_ids = []
confidences = []
boxes = []

# Iterate through each detection
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Only consider detections above a certain confidence threshold (e.g., 50%)
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Coordinates for bounding box
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-Max Suppression to filter overlapping bounding boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Drawing bounding boxes and labels
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        
        # Only draw boxes for cats and dogs
        if label == 'cat' or label == 'dog':
            # Draw bounding box in green
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Label the detected objectr
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Display the result
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output image with predictions
output_image_path = 'C:/Users/dell/Desktop/imagerecog/catdog_detected.jpg'
cv2.imwrite(output_image_path, img)
