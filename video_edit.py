'''
Author: WANG Zichen

Edit the input video with certain cutting invertal (CI), which means 
    keep one frame per every CI frames, to remove unnesessary parts. 
    
    The class is implemented for dataset preparation with pre-defined 
    valid length (1080 frames) and unwanted sections (before the 120th 
    frame and after the valid period).

    Thus, this class is optional to users, depending on the purposes and 
    the charateristics of input videos.
'''
from collections import deque
import argparse
import cv2
import numpy as np
import time
import logging

class EditVideo():
    '''
    Edit the input video with certain cutting invertal (CI), which means 
    keep one frame per every CI frames, to remove unnesessary parts. 
    
    The class is implemented for dataset preparation with pre-defined 
    valid length (1080 frames) and unwanted sections (before the 120th 
    frame and after the valid period).

    Thus, this class is optional to users, depending on the purposes and 
    the charateristics of input videos.

    Parameters:
        folder ->(str): name of the input folder. All videos inside of the 
                        folder will be processed.
        mode ->(str): modes of operation. 
            count
    '''
    def __init__(self, folder, mode, interval=5):
        self._folder = folder
        self._mode = mode
        self._interval = interval

    def run(self):
        print("STARTING...")
        print("FOLDER: " + self._folder)
        print("MODE: " + mode)
        print("INTERVAL(IF VALID): " + str(interval) + "\n")
        
        if self._mode == "interval_cut":
            self._interval_cut()
        elif self._mode == "count":
            self._count_number_of_frame()

        print("\nFINISHED")

    def _count_number_of_frame(self):
        video_mask = self._folder + "/*.MOV"
        video_names = glob(video_mask)

        for video_name in video_names:
            if str(video_name).find("new") < 0:
                continue

            cap = cv2.VideoCapture(video_name)

            if cap.isOpened():
                print(video_name + " is opened\nStart to count the number of frame")
            else:
                raise FileNotFoundError("The input video path you provided is invalid.")

            fps, f_height, f_width, count = self._get_video_properties(cap)
            print("FPS: " + str(fps) + "\nFrame Height: " + str(f_height) + "\nFrame Width: " +str(f_width) + "\nTotal Frame Number: " + str(count) + "\n")
            
            cap.release()

    def _interval_cut(self):
        '''
        Perform the interval cut operation
        '''
        video_mask = self._folder + "/*.MOV"
        video_names = glob(video_mask)

        for video_name in video_names:
            if str(video_name).find("new") > 0:
                continue

            cap = cv2.VideoCapture(video_name)

            if cap.isOpened():
                print(video_name + " is opened\nStart to perform interval cutting")
            else:
                raise FileNotFoundError("The input video path you provided is invalid.")

            fps, f_height, f_width, count = self._get_video_properties(cap)
            print(str(fps) + " " + str(f_height) + " " +str(f_width) + " " +str(count / fps))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_name = str(video_name).replace(".MOV", "_new_{0:d}.MOV".format(self._interval)).replace("original", "modified")
            print("Output Path: " + output_name)

            output_video = cv2.VideoWriter(output_name, fourcc, fps / 2,  (f_width, f_height))

            frame_count = 0
            skip_mode = True

            while cap.isOpened():
                grabbed, frame = cap.read()
                if not grabbed:
                    break

                if frame_count < 120:
                   frame_count += 1
                   continue 
                else:
                    if frame_count % self._interval == 0:
                        output_video.write(frame)
                    frame_count += 1

                    if frame_count > 1080:
                        break

            cap.release()
            output_video.release()
            print("Finished")

    def _get_video_properties(self, video):
        video_fps = round(video.get(cv2.CAP_PROP_FPS))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        return video_fps, frame_height, frame_width, frame_count
'''
For Test Only
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select the folder to be processed')

    parser.add_argument("--folder", type=str, required=True,
                        help='the folder which contains the videos to be edited')
    parser.add_argument("--mode", type=str, required=True, choices=["interval_cut", "count"],
                        help="Mode of the program.")
    parser.add_argument("--interval", type=int, default=5,
                        help='the frame interval. Since in some part of the video, the change is not obvious.')
   
    args = parser.parse_args()
    editVideo = EditVideo(args.folder, args.mode, args.interval)

    editVideo.run()
'''
