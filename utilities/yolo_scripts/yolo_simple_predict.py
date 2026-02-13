from ultralytics import YOLO

# Load a model
model = YOLO("path/to/your/model/best.pt")  # path to your trained YOLO model

# Run batched inference on a list of images
results = model(["path/to/image1.png",
                 "path/to/image2.png",  # return a list of Results objects
                 "path/to/image3.png",
                 "path/to/image4.png",
                 "path/to/image5.png"])

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk