'''
Author: WANG Zichen

The CoorExtraction class takes video as input and generates the 
detection result of all hand keypoints as a log file. Totally,
there are 21 keypoints are detected.
'''
from collections import deque
import argparse
import cv2
import numpy as np
import time
import logging
from glob import glob

NET_INPUT_HEIGHT = 368
HAND_SECTION =["PALM", "THUMB_1", "THUMB_2", "THUMB_3", "THUMB_TIP", "INDEX_FINGER_1", "INDEX_FINGER_2", "INDEX_FINGER_3", "INDEX_FINGER_TIP", "MIDDLE_FINGER_1", "MIDDLE_FINGER_2", "MIDDLE_FINGER_3", "MIDDLE_FINGER_TIP", "RING_FINGER_1", "RING_FINGER_2", "RING_FINGER_3", "RING_FINGER_TIP", "PINKY_FINGER_1", "PINKY_FINGER_2", "PINKY_FINGER_3", "PINKY_FINGER_TIP" ]


class CoorExtraction():
    '''
    The CoorExtraction class takes video as input and generates the 
    detection result of all hand keypoints as a log file. Totally,
    there are 21 keypoints are detected.

    Parameters: 
        proto_file_path (str): the path of the proto file
        weights_file_path (str): the path of the weight file
        input_video_path (str): the path of the input video
        threshold (double): the detection threshold. the higher the 
        more accurate
    '''
    def __init__(self, proto_file_path, weights_file_path, input_video_path, threshold):
        self._net = cv2.dnn.readNetFromCaffe(proto_file_path, weights_file_path)
        self._input_video_path = input_video_path
        self._max_hands_detected = 1
        self._threshold = threshold
        self._frame_number = 0
        log_name = input_video_path.replace(".MOV", ".txt")
        self._log_file = open(log_name, "w")
        self._log_to("LOG FILE: " + log_name)
        self._log_to("PROCESSING_STARTS")
        self._log_to(str("INPUT FILE: " + self._input_video_path))
        self._log_to("DETECTION THRESHOLD: {}".format(self._threshold))

    def _run_extraction(self):      
        '''
        Initiate the extraction procedure
        Return the name of lof file for further evaluation
        '''
        print("Extraction has been initialized")
        t = time.time()

        input_video = cv2.VideoCapture(self._input_video_path)
        if not input_video.isOpened():
            raise FileNotFoundError("The input video path you provided is invalid.")
        video_fps, frame_h, frame_w, frame_count = self._get_video_properties(input_video)
        self._log_to("TOTAL NUMBER OF FRAMES: " + str(frame_count) + "\n")
        finger_ixs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        while input_video.isOpened():
            grabbed, frame = input_video.read()
            if not grabbed:
                break

            finger_coords = self._get_all_finger_coords_from_frame(frame, finger_ixs, self._max_hands_detected, self._threshold)
            #frame_finger_coords_queue.append(finger_coords)

        input_video.release()
        print("Total time taken for extraction: {:.3f}".format((time.time() - t) / 60) + " MINUTES")
        return self._log_file

    def _get_all_finger_coords_from_frame(self, frame, finger_ixs, max_hands_detected, threshold):
        all_finger_coords = []

        frame_height, frame_width = frame.shape[:2]
        aspect_ratio = frame_width / frame_height
        net_input_width = int(aspect_ratio * NET_INPUT_HEIGHT)

        input_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (net_input_width, NET_INPUT_HEIGHT),
                                           (0, 0, 0), swapRB=False, crop=False)

        self._net.setInput(input_blob)
        prediction = self._net.forward()

        print("Frame Number: ", self._frame_number)
        self._log_to("Frame Number: " + str(self._frame_number))
        self._frame_number += 1

        for finger_ix in finger_ixs:
            probability_map = prediction[0, finger_ix, :, :]

            finger_coords = self._get_single_finger_coords(probability_map, frame_height, frame_width, max_hands_detected, threshold)
            all_finger_coords.append(finger_coords)
            self._log_to(HAND_SECTION[finger_ix] + ": " + (str(finger_coords)))
            #print(HAND_SECTION[finger_ix] + ": " + (str(finger_coords)))

        self._log_to("")
        #print("")

        return all_finger_coords
    
    def _get_max_prob_coordinate(self, probability_map):
        probability = np.max(probability_map)
        coord = np.unravel_index(probability_map.argmax(), probability_map.shape)

        return probability, coord

    def _get_circular_mask(self, frame, radius, coordinate):
        img_h, img_w = frame.shape[0], frame.shape[1]
        center_y, center_x = coordinate

        y, x = np.ogrid[-center_y: img_h - center_y, -center_x: img_w - center_x]
        circular_mask = x * x + y * y <= radius * radius

        return circular_mask

    def _get_single_finger_coords(self, probability_map, frame_height, frame_width, max_hands_detected, threshold):
        """
        Returns up to max_hands_detected pairs of coordinates for a single type of finger.
        e.g. If max_hands_detected is 2 and if there are 2 instances of a ring finger in an image, the coordinate points
        for each instance will be returned.
        """
        # The resize ratio values will be used to rescale the coordinate points
        # from the probability maps to the original image frame.
        prob_map_height_ratio = frame_height / probability_map.shape[0]
        prob_map_width_ratio = frame_width / probability_map.shape[1]

        finger_coords = []

        for _ in range(max_hands_detected):
            probability, coordinate = self._get_max_prob_coordinate(probability_map)
            if probability < threshold:
                break

            coord_resized = self._resize_coordinate(coordinate, prob_map_height_ratio, prob_map_width_ratio)
            finger_coords.append(coord_resized)

            # Remove coordinate from the probability map so that it is not retrieved again.
            mask = self._get_circular_mask(probability_map, radius=3, coordinate=coordinate)
            probability_map[mask] = 0.0

        return finger_coords
        
    def _log_to(self, line):
        '''
        Record lines to the log file

        Parameter:
            line (str): the line to be recorded
        '''
        self._log_file.write(line + "\n")

    def _get_video_properties(self, video):
        video_fps = round(video.get(cv2.CAP_PROP_FPS))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        return video_fps, frame_height, frame_width, frame_count

    def _resize_coordinate(self, coordinate, height_ratio, width_ratio):
        coord_y, coord_x = coordinate
        coord_resized_y = int(coord_y * height_ratio)
        coord_resized_x = int(coord_x * width_ratio)

        return coord_resized_y, coord_resized_x

