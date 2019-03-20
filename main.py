'''
Author: WANG Zichen

The Main function with testing parameter
'''
import evaluation as e
from training import NotSimpleNet
from glob import glob

def get_level(level_code):
    if level_code == "2" or level_code == "11" or level_code == "15":
        return "New"
    elif level_code == "4" or level_code == "7" or level_code == "9" or level_code == "10" or level_code == "14":
        return "Fluent"
    else: # level_code == "3" or level_code == "5" or level_code == "4"
        return "Skilled"

if __name__ == "__main__":
    MODEL_PATH = "torch_model/model_0.7608108108108108.model"
    VIDEO_PATH = "cases/s_10_1_new_3.MOV"
    PROTO_FILE_PATH = "caffe_model/pose_deploy.prototxt"
    WEIGHTS_FILE_PATH = "caffe_model/pose_iter_102000.caffemodel"
    evaluation = e.Evaluation(VIDEO_PATH, MODEL_PATH, PROTO_FILE_PATH, WEIGHTS_FILE_PATH)
    
    input_folder = "test_samples"
    log_file_mask = input_folder + "\*.txt"
    log_file_names = glob(log_file_mask)

    t_conf = 0
    acc_count = 0
    for file_name in log_file_names:
        print(file_name)
        dump0, dump1, level_code, dump3, dump4, dump5 = file_name.split("_")
        prof, conf = evaluation._start_evaluation(file_name)
        if get_level(level_code) == prof:
            acc_count += 1

        t_conf += conf
    print("The model performs {} accuracy with {} confidence over {} files.".format(acc_count / len(log_file_names), t_conf / len(log_file_names), len(log_file_names)))
