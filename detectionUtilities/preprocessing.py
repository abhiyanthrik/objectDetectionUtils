import os
import cv2
import numpy as np

# KNN
KNN_subtractor = cv2.createBackgroundSubtractorKNN(
    detectShadows=True)  # detectShadows=True : exclude shadow areas from the objects you detected

# MOG2
MOG2_subtractor = cv2.createBackgroundSubtractorMOG2(
    detectShadows=True)  # exclude shadow areas from the objects you detected

# Kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# choose your subtractor
bg_subtractor = KNN_subtractor


def video_to_frames(video_path: str) -> None:
    base_path = os.path.dirname(video_path)
    file_name = os.path.basename(video_path).split('.')[0]
    frames_path = os.path.join(base_path, f'{file_name}-frames')
    os.makedirs(frames_path, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video file")
        return
    iterator = 1
    save = False
    while cap.isOpened():
        ret, original_frame = cap.read()
        if not ret or original_frame is None:
            print("Could not read frame")
            break
        frame = original_frame.copy()
        frame = cv2.resize(frame, (780, 540), interpolation=cv2.INTER_LINEAR)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        points = np.array([[180, 380], [165, 280], [210, 210], [685, 110], [780, 215], [780, 540]], dtype=np.int32)
        cv2.fillPoly(mask, [points], color=(255, 255, 255))
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        # Every frame is used both for calculating the foreground mask and for updating the background.
        foreground_mask = bg_subtractor.apply(masked)
        masked_fg = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)

        k = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(masked_fg, k, iterations=1)

        # threshold if it is bigger than 240 pixel is equal to 255 if smaller pixel is equal to 0
        # create binary image , it contains only white and black pixels
        ret, threshold = cv2.threshold(erosion.copy(), 120, 255, cv2.THRESH_BINARY)

        # find contours
        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # check every contour if are exceed certain value draw bounding boxes
        for contour in contours:
            # if area exceed certain value then draw bounding boxes
            if cv2.contourArea(contour) > 220:
                # print(cv2.contourArea(contour))
                # (x, y, w, h) = cv2.boundingRect(contour)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                save = True
        if save:
            frame_path = os.path.join(frames_path, f'frame-{iterator}.jpg')
            print(f"Saving Frame: {iterator}")
            iterator += 1
            cv2.imwrite(frame_path, original_frame)
            save = False

    #     cv2.imshow("Subtracted", foreground_mask)
    #     cv2.imshow("threshold", threshold)
    #     cv2.imshow("detection", frame)
    #     cv2.imshow("masked", masked)
    #
    #     if cv2.waitKey(30) & 0xff == 27:
    #         break
    #
    # cap.release()
    # cv2.destroyAllWindows()


# def frame_segregation(frames_path: str) -> None:
#     for frame_name in os.listdir(frames_path):
#         frame_path = os.path.join(frames_path, frame_name)
#         original_frame = cv2.imread(frame_path)
#         frame = original_frame.copy()
#         frame = cv2.resize(frame, (780, 540), interpolation=cv2.INTER_LINEAR)
#         mask = np.zeros(frame.shape[:2], dtype=np.uint8)
#         points = np.array([[315, 415], [345, 340], [555, 225], [675, 200], [630, 500]], dtype=np.int32)
#         cv2.fillPoly(mask, [points], color=(255, 255, 255))
#         # cv2.polylines(frame, [points], True, color=(255, 0, 0), thickness=5)
#         masked = cv2.bitwise_and(frame, frame, mask=mask)
#         # Every frame is used both for calculating the foreground mask and for updating the background.
#         foreground_mask = bg_subtractor.apply(masked)
#         masked_fg = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
#
#         k = np.ones((3, 3), np.uint8)
#         erosion = cv2.erode(masked_fg, k, iterations=1)
#
#         # threshold if it is bigger than 240 pixel is equal to 255 if smaller pixel is equal to 0
#         # create binary image , it contains only white and black pixels
#         ret, threshold = cv2.threshold(erosion.copy(), 120, 255, cv2.THRESH_BINARY)
#
#         # find contours
#         contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         # check every contour if are exceed certain value draw bounding boxes
#         remove = True
#         area = 0
#         for contour in contours:
#             if cv2.contourArea(contour) > 450:
#                 area = cv2.contourArea(contour)
#                 remove = False
#                 break
#
#         if remove:
#             print(f"Removing file: {frame_name} as area: {area}")
#             os.remove(frame_path)
#
#     #     cv2.imshow("Subtracted", foreground_mask)
#     #     cv2.imshow("threshold", threshold)
#     #     cv2.imshow("detection", frame)
#     #     cv2.imshow("masked", masked)
#     #
#     #     a = cv2.waitKey(0)
#     #     if a & 0xFF == ord('q'):
#     #         break
#     # cv2.destroyAllWindows()


def manual_segregation(frames_path: str) -> None:
    for frame in os.listdir(frames_path):
        frame_path = os.path.join(frames_path, frame)
        frame = cv2.imread(frame_path)
        cv2.imshow('frame', frame)
        a = cv2.waitKey(0)
        if a & 0xFF == ord('y'):
            print("Yes")
        if a & 0xFF == ord('n'):
            print(f"Removing file: {frame_path}")
            os.remove(frame_path)
    cv2.destroyAllWindows()
