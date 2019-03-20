'''
Author: WANG Zichen

The Main function
'''
import evaluation as e

if __name__ == "__main__":
    MODEL_PATH = "torch_model/model_0.7608108108108108.model"
    VIDEO_PATH = "cases/s_10_1_new_3.MOV"
    PROTO_FILE_PATH = "caffe_model/pose_deploy.prototxt"
    WEIGHTS_FILE_PATH = "caffe_model/pose_iter_102000.caffemodel"
    evaluation = e.Evaluation(VIDEO_PATH, MODEL_PATH, PROTO_FILE_PATH, WEIGHTS_FILE_PATH)
    evaluation._start_evaluation()