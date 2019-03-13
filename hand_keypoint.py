from collections import deque
import argparse
import cv2
import numpy as np
import time
import logging
from glob import glob

KEYPOINT_DETECTION_PROB_THRESHOLD = 0.3
NET_INPUT_HEIGHT = 368

LIGHT_MAX_ALPHA = 1.0
LIGHT_GAUSSIAN_BLUR_K_SIZE = (17, 17)

PALM = 0
THUMB_IX = 4
INDEX_FINGER_IX = 8
MIDDLE_FINGER_IX = 12
RING_FINGER_IX = 16
PINKY_FINGER_IX = 20

POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]
HAND_SECTION =["PALM", "THUMB_1", "THUMB_2", "THUMB_3", "THUMB_TIP", "INDEX_FINGER_1", "INDEX_FINGER_2", "INDEX_FINGER_3", "INDEX_FINGER_TIP", "MIDDLE_FINGER_1", "MIDDLE_FINGER_2", "MIDDLE_FINGER_3", "MIDDLE_FINGER_TIP", "RING_FINGER_1", "RING_FINGER_2", "RING_FINGER_3", "RING_FINGER_TIP", "PINKY_FINGER_1", "PINKY_FINGER_2", "PINKY_FINGER_3", "PINKY_FINGER_TIP" ]
JOINT_ANGLE = [[1,2,3],[2,3,4],[5,6,7],[6,7,8],[9,10,11],[10,11,12],[13,14,15],[14,15,16],[17,18,19],[18,19,20]]


BGR_BLUE = [255, 102, 70]
BGR_GREEN = [57, 255, 20]
BGR_RED = [58, 7, 255]
BGR_WHITE = [250, 250, 255]
BGR_YELLOW = [21, 243, 243]
BGR_BLACK = [255, 255, 255]
BGR_COLORS = {
    "black" : BGR_BLACK,     
    "blue": BGR_BLUE,    
    "green": BGR_GREEN,    
    "red": BGR_RED,   
    "white": BGR_WHITE,
    "yellow": BGR_YELLOW
}

class HandLights():

    def __init__(self, proto_file_path, weights_file_path, log_name):
        self._net = cv2.dnn.readNetFromCaffe(proto_file_path, weights_file_path)
        self._frame_number = 0
        file_name = time.strftime(log_name)
        self._log_file = open(file_name, "w")
        self._log_to("LOG FILE: " + file_name + "\n")
        
    def run_video(self, input_video_path, output_video_path="output_video.mp4", hasDemo=False, fingers="all", max_hands_detected=2,
                  light_color="green", light_duration_n_secs=0.1, light_radius_frame_height_ratio=0.015,
                  light_same_alpha=False, background_alpha=1.0, mirror=False, threshold = 0.5):
        input_video = cv2.VideoCapture(input_video_path)
        if not input_video.isOpened():
            raise FileNotFoundError("The input video path you provided is invalid.")

        video_fps, frame_h, frame_w, frame_count = self._get_video_properties(input_video)
        light_duration_n_frames = int(light_duration_n_secs * video_fps)
        frame_finger_coords_queue = deque(maxlen=light_duration_n_frames)

        print("ESTIMATE TIME: {:.3f} MINUTES".format(frame_count * 3 / 60))
        self._log_to("TOTAL NUMBER OF FRAMES: " + str(frame_count) + "\n")

        if hasDemo:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_video = cv2.VideoWriter(output_video_path, fourcc, video_fps,  (frame_w, frame_h))
            

        finger_ixs = self._get_finger_ixs(fingers)
        current_frame_ix = 0

        while input_video.isOpened():
            grabbed, frame = input_video.read()
            if not grabbed:
                break

            finger_coords = self._get_all_finger_coords_from_frame(frame, finger_ixs, max_hands_detected, threshold)
            frame_finger_coords_queue.append(finger_coords)


            if hasDemo:
                frame_drawn = self._draw_lights_on_frame_using_coords_queue(frame, frame_finger_coords_queue, light_color,
                                                                        light_radius_frame_height_ratio, light_same_alpha,
                                                                        background_alpha, mirror)
                output_video.write(frame_drawn)

            current_frame_ix += 1
            

        input_video.release()
        
        if hasDemo:
            output_video.release()


    def _blur_lights(self, frame, lights_mask):
        """
        A Gaussian Blur is applied to all of the lights to smoothen out the lines that are created
        from overlapping lights.
        """
        frame_copy = np.copy(frame)
        frame_blurred = cv2.GaussianBlur(frame, LIGHT_GAUSSIAN_BLUR_K_SIZE, 0)
        frame_copy[lights_mask] = frame_blurred[lights_mask]

        return frame_copy

    def _darken_frame(self, frame, background_alpha):
        """
        The entire frame will be darkened based on the background_alpha value.

        :param background_alpha: 0.0 creates a solid black frame. 1.0 leaves the frame unmodified/opaque.
        """
        black_frame = np.zeros_like(frame)
        frame_darkened = cv2.addWeighted(frame, background_alpha, black_frame, 1.0 - background_alpha, 0)

        return frame_darkened

    def _draw_lights_on_frame_using_coords_queue(self, frame, frame_finger_coords_queue, light_color,
                                                 light_radius_frame_height_ratio, light_same_alpha, background_alpha, mirror):
        light_alphas = self._get_light_alphas_for_frame(light_same_alpha, frames_queue_size=len(frame_finger_coords_queue))
        masks_merged_all_frames = np.full(frame.shape[:2], fill_value=False, dtype=bool)
        frame = self._darken_frame(frame, background_alpha)

        for frame_finger_coords, light_alpha in zip(frame_finger_coords_queue, light_alphas):
            frame, masks_merged = self._draw_lights_on_frame_using_single_frame_coords(frame, frame_finger_coords,
                                                                                       light_color, light_alpha,
                                                                                       light_radius_frame_height_ratio)
            masks_merged_all_frames = np.logical_or(masks_merged_all_frames, masks_merged)


        frame = self._blur_lights(frame, masks_merged_all_frames)
        if mirror:
            frame = np.fliplr(frame)

        return  frame

    def _draw_lights_on_frame_using_single_frame_coords(self, frame, frame_finger_coords, light_color, light_alpha,
                                                        light_radius_frame_height_ratio):
        """
        Using the finger coordinates of a single frame, lights are drawn in a circular shape with the alpha parameter.
        The circular mask for each finger coordinate is saved into a single mask called masks_merged.
        """
        frame_with_lights = np.copy(frame)
        light_radius = light_radius_frame_height_ratio * frame.shape[0]
        masks_merged = np.full(frame.shape[:2], fill_value=False, dtype=bool)
        light_colors_BGR_for_fingers = self._get_light_colors_BGR_for_fingers(light_color, n_fingers=len(frame_finger_coords))

        for single_finger_coords, light_color_BGR_for_finger in zip(frame_finger_coords, light_colors_BGR_for_fingers):
            for finger_coord in single_finger_coords:
                mask = self._get_circular_mask(frame_with_lights, radius=light_radius, coordinate=finger_coord)
                frame_with_lights[mask] = light_color_BGR_for_finger

                masks_merged = np.logical_or(masks_merged, mask)

        frame_with_transparent_lights = cv2.addWeighted(frame_with_lights, light_alpha, frame, 1.0-light_alpha, 0)

        return frame_with_transparent_lights, masks_merged

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

    def _get_circular_mask(self, frame, radius, coordinate):
        img_h, img_w = frame.shape[0], frame.shape[1]
        center_y, center_x = coordinate

        y, x = np.ogrid[-center_y: img_h - center_y, -center_x: img_w - center_x]
        circular_mask = x * x + y * y <= radius * radius

        return circular_mask

    def _get_finger_ixs(self, fingers):
        if fingers == "index":
            finger_ixs = [PALM, INDEX_FINGER_IX]
        elif fingers == "tips":
            finger_ixs = [PALM, THUMB_IX, INDEX_FINGER_IX, MIDDLE_FINGER_IX, RING_FINGER_IX, PINKY_FINGER_IX]
        else: # all
            finger_ixs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        return finger_ixs

    def _get_light_alphas_for_frame(self, light_same_alpha, frames_queue_size):
        if light_same_alpha:
            light_alphas = [LIGHT_MAX_ALPHA] * frames_queue_size
        else:
            light_alphas = np.linspace(0.0, LIGHT_MAX_ALPHA, num=frames_queue_size + 1)[1:]  # skip first alpha value (0.0)

        return light_alphas

    def _get_light_colors_BGR_for_fingers(self, light_color, n_fingers):
        if light_color == "all":
            light_colors_BGR = list(BGR_COLORS.values())[:n_fingers]
        else:
            light_colors_BGR = [BGR_COLORS[light_color]] * n_fingers

        return light_colors_BGR

    def _get_max_prob_coordinate(self, probability_map):
        probability = np.max(probability_map)
        coord = np.unravel_index(probability_map.argmax(), probability_map.shape)

        return probability, coord

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

    def _get_video_properties(self, video):
        video_fps = round(video.get(cv2.CAP_PROP_FPS))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        return video_fps, frame_height, frame_width, frame_count

    def _log_to(self, line):
        self._log_file.write(line + "\n")

    def _resize_coordinate(self, coordinate, height_ratio, width_ratio):
        coord_y, coord_x = coordinate
        coord_resized_y = int(coord_y * height_ratio)
        coord_resized_x = int(coord_x * width_ratio)

        return coord_resized_y, coord_resized_x


def restricted_float_0_to_1(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        error_msg = "{} not in range [0.0, 1.0]".format(x)
        raise argparse.ArgumentTypeError(error_msg)
    return x


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Draws virtual lights using bare hands.')
    parser.add_argument("--proto_file_path", type=str, default="caffe_model/pose_deploy.prototxt",
                        help='Path to the Caffe Pose prototxt file.')
    parser.add_argument("--weights_file_path", type=str, default="caffe_model/pose_iter_102000.caffemodel",
                        help='Path to the Caffe Pose model.')

    parser.add_argument("--input_folder", type=str, required=True,
                        help='Path to the video that will be processed. e.g. media/videos/my_video.mp4')
    parser.add_argument("--output_folder", type=str, required=True,
                        help='Where to save the processed video. e.g. media/videos/my_video_output.mp4')
    parser.add_argument("--output_demo", type=bool, default=False,
                        help='Where to save the processed video. e.g. media/videos/my_video_output.mp4')    
    
    
    parser.add_argument("--fingers", type=str, default="all", choices=["all", "index", "tips"],
                        help='Specifies which finger(s) will be detected in the video.')
    parser.add_argument("--max_hands_detected", type=int, default=2, choices=[1, 2, 3, 4],
                        help="Maximum number of hands that can be detected.")
    parser.add_argument("--light_color", type=str, default="green",
                        choices=["all", "blue", "green", "red", "white", "yellow"],
                        help="The color of the lights that are drawn. 'all' should only be used when fingers=all.")
    parser.add_argument("--light_duration_n_secs", type=float, default=0.1,
                        help="How long a light will be visible for.")
    parser.add_argument("--light_radius_frame_height_ratio", type=restricted_float_0_to_1, default=0.015,
                        help="Ratio of the light radius relative to the frame height.")
    parser.add_argument("--light_same_alpha", type=bool, default=False,
                        help='If set to True, light alpha values will not decrease. LIGHT_MAX_ALPHA will be used on all lights.')
    parser.add_argument("--background_alpha", type=restricted_float_0_to_1, default=1.0,
                        help="0.0 creates a solid black background. 1.0 leaves the background unmodified/opaque.")
    parser.add_argument("--mirror", type=bool, default=False,
                        help="Each frame in the output video will be a mirror image.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Set the determination accuracy rate. The higher the more accurate")
    
    args = parser.parse_args()

  #  hand_lights = HandLights(args.proto_file_path, args.weights_file_path)
    
    input_folder = args.input_folder
    output_folder = args.output_folder
    hasDemo = False
    hasDemo = args.output_demo

    input_mask = input_folder + "/*.MOV"
    input_video_names = glob(input_mask)

    #print(input_mask)

    t = time.time()

    log_file_mask = output_folder + "/*.txt"
    log_file_names = glob(log_file_mask)

    for input_video_name in input_video_names:

        file_name = str(input_video_name).replace(input_folder, "")
        log_file_name = output_folder + file_name.replace("MOV", "txt")
        output_video_name = output_folder + file_name.replace(".MOV", "_new.MOV")
        print(input_video_name)
        print(log_file_name)

        jump = False

        for log_file in log_file_names:
            if log_file_name == log_file:
                jump = True

        if jump:
            continue

        hand_lights = HandLights(args.proto_file_path, args.weights_file_path, log_file_name)
        hand_lights._log_to("PROCESSING_STARTS")
        hand_lights._log_to(str("INPUT FILE: " + input_video_name))
        hand_lights._log_to("log FILE: " + log_file_name)
        hand_lights._log_to("DETECTION MODE: " + args.fingers)
        hand_lights._log_to("DETECTION THRESHOLD: " + str(args.threshold))
        
        
        #hand_lights.run_video(input_video_name, log_file_name)

        hand_lights.run_video(input_video_name, output_video_name, hasDemo, args.fingers, args.max_hands_detected, args.light_color,
                            args.light_duration_n_secs, args.light_radius_frame_height_ratio, args.light_same_alpha,
                            args.background_alpha, args.mirror, args.threshold)

        hasDemo = False

   # hand_lights.run_video(args.input_video_path, args.output_video_path, args.fingers, args.max_hands_detected, args.light_color,
  #                        args.light_duration_n_secs, args.light_radius_frame_height_ratio, args.light_same_alpha,
  #                        args.background_alpha, args.mirror, args.threshold)
   # 
   # hand_lights._log_to("TOTAL TIME TAKEN : {:.3f}".format((time.time() - t) / 60) + " MINUTES")
    print("Total time taken : {:.3f}".format((time.time() - t) / 60) + " MINUTES")