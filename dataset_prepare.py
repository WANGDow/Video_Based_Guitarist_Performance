import csv
from glob import glob
import numpy as np
import math

JOINT_ANGLE = [[1,2,3],[2,3,4],[5,6,7],[6,7,8],[9,10,11],[10,11,12],[13,14,15],[14,15,16],[17,18,19],[18,19,20]]
POINT_PAIRS = [[6,7],[7,8],[10,11],[11,12],[14,15],[15,16],[18,19],[19,20]]
THUMB_TIP = 4

def get_angle_from_coor(coor_buf):
    angle_list = []
    for angle_index in JOINT_ANGLE:
        key_b = coor_buf[angle_index[0]]
        key_a = coor_buf[angle_index[1]]
        key_c = coor_buf[angle_index[2]]
        if key_a._is_zero() or key_b._is_zero() or key_c._is_zero():
            angle_list.append(0)
            continue
        
        len_ac = math.sqrt((key_a._get_x() - key_c._get_x()) * (key_a._get_x() - key_c._get_x()) + (key_a._get_y() - key_c._get_y()) * (key_a._get_y() - key_c._get_y())) 
        len_ab = math.sqrt((key_a._get_x() - key_b._get_x()) * (key_a._get_x() - key_b._get_x()) + (key_a._get_y() - key_b._get_y()) * (key_a._get_y() - key_b._get_y())) 
        angle = math.acos(np.dot([key_a._get_x() - key_c._get_x(),key_a._get_y() - key_c._get_y()],[key_a._get_x() - key_b._get_x(),key_a._get_y() - key_b._get_y()]) / (len_ac * len_ab))
        angle_list.append(angle)
        print(angle)
    return angle_list

def get_distance_from_coor_tips(coor_buf):
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

def get_level(level_code):
    if level_code == "2" or level_code == "11" or level_code == "15":
        return 0 #"New"
    elif level_code == "4" or level_code == "7" or level_code == "9" or level_code == "10" or level_code == "14":
        return 1 #"Fluent"
    else: # level_code == "3" or level_code == "5" or level_code == "4"
        return 2 #"Skilled"

def get_level_with_name(level_code):
    if level_code == "2" or level_code == "11" or level_code == "15":
        return 0, "New" #"New"
    elif level_code == "4" or level_code == "7" or level_code == "9" or level_code == "10" or level_code == "14":
        return 1, "Fluent" #"Fluent"
    else: # level_code == "3" or level_code == "5" or level_code == "4"
        return 2, "Skilled" #"Skilled"

def get_cors_from_line(line):
    x_cor = "0"
    y_cor = "0"
    #print(line)
    if line == "[]\n": #No values are found
        x_cor = "0"
        y_cor = "0"
    else: #Values are found
        first_half = line.split(",")[0]
        second_half = line.split(",")[1]
        #print(second_falf)
       
        
        first_half = str(first_half).replace("(", "")
        second_half = str(second_half).replace(")", "")
        first_half = str(first_half).replace("[", "")
        second_half = str(second_half).replace("]", "")
        #line = str(line).replace(",", "")
        second_half = str(second_half).replace("\n", "")
        x_cor = first_half
        y_cor = second_half

        
        #print(line)
    #print(x_cor + " " + y_cor)
    return int(x_cor), int(y_cor)

def get_chord_from_frame_no(frame_no):
    if int(frame_no / 40) == 0 or int(frame_no / 40) == 5:
        return 0 #C
    elif int(frame_no / 40) == 1:
        return 1 #G
    elif int(frame_no / 40) == 2:
        return 2 #Am
    elif int(frame_no / 40) == 3:
        return 3 #Em
    elif int(frame_no / 40) == 4:
        return 4 #F
    elif int(frame_no / 40) == 6:
        return 5 #Dm7
    else:
        return 6 #Gsus4

class Point():
    def __init__(self, x_coor, y_coor):
        self.x_coor = x_coor
        self.y_coor = y_coor
    
    def _get_x(self):
        return self.x_coor

    def _get_y(self):
        return self.y_coor

    def _is_zero(self):
        return self.x_coor == 0 or self.y_coor == 0

    def __str__(self):
        return str(self.x_coor) + " " + str(self.y_coor)
    def __repr__(self):
        return str(self.x_coor) + " " + str(self.y_coor)


with open("dataset_distance_nomarker_name.csv", "w", newline="") as dataset_file:
    dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    #impliment the first row
    #dataset_writer.writerow(["sequence", "level", "chord", "thumb_angle_0", "thumb_angle_1", "index_angle_0", "index_angle_1","middle_angle_0", "middle_angle_1","ring_angle_0", "ring_angle_1","pinky_angle_0", "pinky_angle_1"])
    #dataset_writer.writerow(["sequence", "level", "chord", "thumb_tip_x", "thumb_tip_y", "index_distance_0", "index_distance_1","middle_distance_0", "middle_distance_1","ring_distance_0", "ring_distance_1","pinky_distance_0", "pinky_distance_1"])
    
    input_folder = "output/test"
   

    log_file_mask = input_folder + "\*.txt"
    log_file_names = glob(log_file_mask)

    for file_name in log_file_names:
        t_file_name = str(file_name).replace(input_folder, "")
        t_file_name = str(t_file_name).replace(".txt", "")
        dump1, level_code, dump3, dump4, interval = t_file_name.split("_")
        #if interval != "3":
         #   continue    

        #file_name = "0.5_thresh\s_2_1_new_3.txt"

       # print(t_file_name)

        #t_file_name = str(file_name).replace("0.5_thresh", "")
        #t_file_name = str(t_file_name).replace(".txt", "")
        #dump1, level_code, dump3, dump4, interval = t_file_name.split("_")
        
        print(file_name)
        #print(level_code)

        file = open(file_name, "r")
        new_row = True
        coor_buf = []
        result_line = []
        #level_code = "2"
        start_record = False
        count = 0
        for line in file:
            if new_row:
                if count == 21:
                    #angle_list = get_angle_from_coor(coor_buf)
                    thumb = coor_buf[THUMB_TIP]
                    if not thumb._is_zero():
                        #result_line.append(thumb._get_x())
                        #result_line.append(thumb._get_y())
                        result_line.append(1)
                    else:
                        #result_line.append(0)
                        #result_line.append(0)
                        result_line.append(0)

                    distance_list = get_distance_from_coor_tips(coor_buf)
                    zero_count = 0
                
                    for distance in distance_list:
                        result_line.append(distance)
                        if distance == 0:
                            zero_count += 1

                    if zero_count <= 2:                
                        dataset_writer.writerow(result_line)
                #print(coor_buf)
                    result_line = []
                    coor_buf = []
                    count = 0
                    new_row = False

            if str(line).find("Frame") >= 0:
                dump1, dump2, frame_no = str(line).split(" ")
                #print(frame_no.replace("\n", ""))
                d_name = file_name + "_" +str(frame_no)
                #result_line.append(d_name)
                level= get_level(level_code)
                #level, name = get_level_with_name(level_code)
                #result_line.append(name)
                result_line.append(level)
                chord = get_chord_from_frame_no(int(frame_no))
                result_line.append(chord)
            #print(result_line)
                start_record = True
                continue

            if str(line).find("[") >= 0:
                #print(line)
                count += 1
                dump1, cor_line = str(line).split(" ", 1)
                x_cor, y_cor = get_cors_from_line(cor_line)
                point = Point(x_cor, y_cor)
                coor_buf.append(point)
            #result_line.append(x_cor)
            #result_line.append(y_cor)
           # print(result_line)
            #print(cor_line)
                if count == 21:
                    new_row = True


'''
with open("dataset.csv", "w", newline="") as dataset_file:
  #  spamwriter = csv.writer(csvfile, delimiter=' ',
  #                          quotechar='|', quoting=csv.QUOTE_MINIMAL)
    dataset_writer = csv.writer(dataset_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    #impliment the first row
    dataset_writer.writerow(["sequence", "level", "chord", "k0_x", "k0_y", "k1_x", "k1_y", "k2_x", "k2_y", "k3_x", "k3_y", "k4_x", "k4_y", "k5_x", "k5_y", "k6_x", "k6_y", "k7_x", "k7_y", "k8_x", "k8_y", "k9_x", "k9_y", "k10_x", "k10_y", "k11_x", "k11_y", "k12_x", "k12_y", "k13_x", "k13_y", "k14_x", "k14_y", "k15_x", "k15_y", "k16_x", "k16_y", "k17_x", "k17_y", "k18_x", "k18_y", "k19_x", "k19_y", "k20_x", "k20_y"])
    
    input_folder = "output_0.5"
   

    log_file_mask = input_folder + "\*.txt"
    log_file_names = glob(log_file_mask)
   
    for file_name in log_file_names:
        t_file_name = str(file_name).replace(input_folder, "")
        t_file_name = str(t_file_name).replace(".txt", "")
        dump1, level_code, dump3, dump4, interval = t_file_name.split("_")
        if interval != "3":
            continue    

        print(file_name)
        #print(level_code)

        file = open(file_name, "r")
        new_row = True
        result_line = []
        #level_code = "2"
        start_record = False
        count = 0

        for line in file:
            if new_row:
                if count == 21:
                    dataset_writer.writerow(result_line)
                    #print("record")
                result_line = []
                count = 0
                new_row = False

            if str(line).find("Frame") >= 0:
                dump1, dump2, frame_no = str(line).split(" ")
                #print(frame_no.replace("\n", ""))
                d_name = file_name + "_" +str(frame_no)
                result_line.append(d_name)
                level = get_level(level_code)
                result_line.append(level)
                chord = get_chord_from_frame_no(int(frame_no))
                result_line.append(chord)
                #print(result_line)
                start_record = True
                continue

            if str(line).find("[") >= 0:
                #print(line)
                count += 1
                dump1, cor_line = str(line).split(" ", 1)
                x_cor, y_cor = get_cors_from_line(cor_line)
                result_line.append(x_cor)
                result_line.append(y_cor)
           # print(result_line)
            #print(cor_line)
                if count == 21:
                    new_row = True
       # print(result_line)
   '''