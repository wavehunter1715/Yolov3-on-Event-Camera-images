import cv2
import numpy as np

# Load the pre-trained YOLO model
net = cv2.dnn.readNetFromDarknet('/Users/chandrikajadon/Downloads/image - reconstruct/yolov3.cfg', '/Users/chandrikajadon/Downloads/image - reconstruct/yolov3.weights')

# Set the classes for object detection
classes = []
with open('/Users/chandrikajadon/Downloads/image - reconstruct/coco.data', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load the event camera image
image = cv2.imread('/Users/chandrikajadon/Downloads/image - reconstruct/eg1.png')

# Convert the image to blob format for input to the model
blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)

# Set the input to the model
net.setInput(blob)

# Forward pass through the network
output_layers = net.getUnconnectedOutLayersNames()
layer_outputs = net.forward(output_layers)

# Process the outputs to get object detections
conf_threshold = 0.5  # Confidence threshold for detection
nms_threshold = 0.4  # Non-maximum suppression threshold
boxes = []
confidences = []
class_ids = []

# Iterate over each output layer
for output in layer_outputs:
    # Iterate over each detection
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > conf_threshold:
            # Calculate the bounding box coordinates
            center_x = int(detection[0] * image.shape[1])
            center_y = int(detection[1] * image.shape[0])
            width = int(detection[2] * image.shape[1])
            height = int(detection[3] * image.shape[0])
            x = int(center_x - width/2)
            y = int(center_y - height/2)

            # Add the bounding box coordinates, confidence, and class ID to the respective lists
            boxes.append([x, y, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply non-maximum suppression to remove redundant overlapping bounding boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Draw the bounding boxes on the image
colors = np.random.uniform(0, 255, size=(len(classes), 3))

if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        color = colors[class_ids[i]]

        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

# Display the image with object detections
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
