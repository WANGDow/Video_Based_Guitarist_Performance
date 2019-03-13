from collections import deque
import argparse
import cv2
import numpy as np
import time
import logging
from glob import glob

OUTPUT_WIDTH = 720

class EditVideo():
    def __init__(self, folder):
        self._folder = folder

    def run(self, mode, interval=5):
        print("STARTING...")
        print("FOLDER: " + self._folder)
        print("MODE: " + mode)
        print("INTERVAL(IF VALID): " + str(interval) + "\n")
        
        if mode == "interval_cut":
            self._interval_cut(interval)
        elif mode == "count":
            self._count_number_of_frame()
        else: #num_cut
            self._num_cut()

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

    def _interval_cut(self, interval):
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
            output_name = str(video_name).replace(".MOV", "_new_{0:d}.MOV".format(interval)).replace("original", "modified")
            print("Output Path: " + output_name)

            #r = f_height / f_width
            #o_height = int(r * OUTPUT_WIDTH)

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
                    if frame_count % interval == 0:
                        output_video.write(frame)
                    frame_count += 1

                    if frame_count > 1080:
                        break


            cap.release()
            output_video.release()
            print("Finished")

    def _num_cut(self):
        return

    def _get_video_properties(self, video):
        video_fps = round(video.get(cv2.CAP_PROP_FPS))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        return video_fps, frame_height, frame_width, frame_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Select the folder to be processed')

    parser.add_argument("--folder", type=str, required=True,
                        help='the folder which contains the videos to be edited')

    parser.add_argument("--interval", type=int, default=5,
                        help='the frame interval. Since in some part of the video, the change is not obvious.')
    parser.add_argument("--mode", type=str, required=True, choices=["interval_cut", "count", "num_cut"],
                        help="Mode of the program.")

    args = parser.parse_args()

    editVideo = EditVideo(args.folder)

    editVideo.run(args.mode, args.interval)
