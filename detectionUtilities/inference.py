from ultralytics import YOLO
import cv2
classes = ['Big Bus', 'Heavy Truck', 'Medium Truck', 'Microbus', 'Minibus-Coaster', 'Motor Cycle', 'Pickup-4 Wheeler', 'Private Car-Sedan Car', 'Small Truck', 'Trailer Truck', 'A-10-S-TANDEM', 'A-10-TRIDEM', 'AXLE', 'COMBINED', 'TYPE-2', 'TYPE-2-S2', 'TYPE-3', 'UC', 'Bus', 'Car', 'Truck', 'autorickshaw', 'tractor', 'truck', 'rickshaw']
# Load a pretrained YOLOv8n model
model_path = "/home/rv/Documents/datasets/trfc-combined-dataset/results/train/weights/best.pt"
# Define path to video file
source = "/home/rv/Downloads/hiv00031 (1).mp4"


def yolo_v8_inference(model_path, source_path):

    model = YOLO(model_path)

    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(source_path)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video file")

    # Read until video is completed
    while cap.isOpened():

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Display the resulting frame
            results = model(frame, verbose=False, conf=0.2)
            annotated_frame = results[0].plot()
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs
                masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints  # Keypoints object for pose outputs
                probs = result.probs  # Probs object for classification outputs
                conf, cls = list(boxes.conf.cpu().numpy()), list(boxes.cls.cpu().numpy())
                if conf:
                    for c in cls:
                        print(classes[int(c)])
                # if boxes.conf >= 0.2:
                #     print(int(boxes.conf))
                # if probs is not None:
                #     pass
            cv2.imshow('Frame', annotated_frame)

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()