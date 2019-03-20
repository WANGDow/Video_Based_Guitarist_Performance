'''
Author: WANG Zichen

The CSV_Generation class produces a csv file, which will
be fed to the classification model, based on the coordinate 
extracted from the video. 
By default, the

The Point class provide a data structure to store 2D value.
'''
import csv
import numpy as np
import math
import time

#For testing purposse only
#INPUT_FORLDER = "cases\s_10_1_new_3.txt"

POINT_PAIRS = [[6,7],[7,8],[10,11],[11,12],[14,15],[15,16],[18,19],[19,20]]
THUMB_TIP = 4

class CSV_Generation():
    '''
    The CSV_Generation class produces a csv file, which will
    be fed to the classification model, based on the coordinate 
    extracted from the video.

    Parameter:
        log_file (str): the path of the log file to be processed
    '''
    def __init__(self, log_file):
        self._log_file = log_file
        self._output_file = log_file.replace(".txt", ".csv")

    def _start_csv(self):
        '''
        Initiate the evaluation procedure.
        Return the csv path
        '''
        t = time.time()
        #print("CSV Generation has been initialized")
        with open(self._output_file, "w", newline="") as dataset_file:
            dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            file = open(self._log_file, "r")
            new_row = True
            coor_buf = []
            result_line = []
            start_record = False
            count = 0
            for line in file:
                if new_row:
                    if count == 21:
                        thumb = coor_buf[THUMB_TIP]
                        if not thumb._is_zero():
                            result_line.append(1)
                        else:
                            result_line.append(0)

                        distance_list = self._get_distance_from_coor_tips(coor_buf)
                        zero_count = 0
                
                        for distance in distance_list:
                            result_line.append(distance)
                            if distance == 0:
                                zero_count += 1

                        if zero_count <= 2:                
                            dataset_writer.writerow(result_line)
                        result_line = []
                        coor_buf = []
                        count = 0
                        new_row = False

                if str(line).find("Frame") >= 0:
                    dump1, dump2, frame_no = str(line).split(" ")
                    d_name = self._log_file + "_" +str(frame_no)
                    chord = self._get_chord_from_frame_no(int(frame_no))
                    result_line.append(chord)
                    start_record = True
                    continue

                if str(line).find("[") >= 0:
                    count += 1
                    dump1, cor_line = str(line).split(" ", 1)
                    x_cor, y_cor = self._get_cors_from_line(cor_line)
                    point = Point(x_cor, y_cor)
                    coor_buf.append(point)
                    if count == 21:
                        new_row = True
        
        #print("Total time taken for csv generation: {:.3f}".format((time.time() - t) / 60) + " MINUTES")
        return self._output_file

    def _get_cors_from_line(self,line):
        '''
        Return coordinates from the input lines

        Parameter:
            line (str): the line in the txt file
        '''
        x_cor = "0"
        y_cor = "0"
        if line == "[]\n": #No values are found
            x_cor = "0"
            y_cor = "0"
        else: #Values are found
            first_half = line.split(",")[0]
            second_half = line.split(",")[1]
            first_half = str(first_half).replace("(", "")
            second_half = str(second_half).replace(")", "")
            first_half = str(first_half).replace("[", "")
            second_half = str(second_half).replace("]", "")
            second_half = str(second_half).replace("\n", "")
            x_cor = first_half
            y_cor = second_half

        return int(x_cor), int(y_cor)

    def _get_chord_from_frame_no(self, frame_no):
        '''
        Return the chord name according to the frame number with chord
        sequence C G Am Em C Dm7 Gsus4. Each chord lasts for 60 frames.
        
        Parameter:
            frame_no (int): the number of the frame
        '''
        if int(frame_no / 60) == 0 or int(frame_no / 40) == 5:
            return 0 #C
        elif int(frame_no / 60) == 1:
            return 1 #G
        elif int(frame_no / 60) == 2:
            return 2 #Am
        elif int(frame_no / 60) == 3:
            return 3 #Em
        elif int(frame_no / 60) == 4:
            return 4 #F
        elif int(frame_no / 60) == 6:
            return 5 #Dm7
        else:
            return 6 #Gsus4

    def _get_distance_from_coor_tips(self, coor_buf):
        '''
        Calculate the distance between two points, which will return 0 if
        the input point contains 0.
        
        Parameter:
            coor_buf (Point[]): a list of Points
        '''
        distance_list = []
    
        for point_pair in POINT_PAIRS:
            key_a = coor_buf[point_pair[0]]
            key_b = coor_buf[point_pair[1]]
            if key_a._is_zero() or key_b._is_zero():
                distance_list.append(0)
                continue
            len_ab = float(math.sqrt((key_a._get_x() - key_b._get_x()) * (key_a._get_x() - key_b._get_x()) + (key_a._get_y() - key_b._get_y()) * (key_a._get_y() - key_b._get_y())))
            distance_list.append(len_ab)

        return distance_list

class Point():
    '''
    The Point class provide a data structure to store 2D value.

    Parameter:
        x_coor (int): x coordinate of the Point
        y_coor (int): y coordinate of the Point
    '''
    def __init__(self, x_coor, y_coor):
        self.x_coor = x_coor
        self.y_coor = y_coor
    
    def _get_x(self):
        '''
        Return the x coordinate
        '''
        return self.x_coor

    def _get_y(self):
        '''
        Return the y coordinate
        '''
        return self.y_coor

    def _is_zero(self):
        '''
        Return if the point contains 0
        '''
        return self.x_coor == 0 or self.y_coor == 0

    #Print the point
    def __str__(self):
        return str(self.x_coor) + " " + str(self.y_coor)
    def __repr__(self):
        return str(self.x_coor) + " " + str(self.y_coor)

#For testing purposes only
'''
if __name__ == "__main__":
    csv_generation = CSV_Generation(INPUT_FILE)
    csv_generation._start_csv()
'''
