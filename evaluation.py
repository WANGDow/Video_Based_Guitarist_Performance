'''
Author: WANG Zichen

The Evaluation class which carries out the whole evaluation procedure, 
including video modification, keypoints extraction, data preparation,
and result generation.
'''
import torch
import numpy as np
from training import NotSimpleNet
from video_edit import EditVideo
from coor_extraction import CoorExtraction
import dataset_prepare as dp
import time

#For testing purposes only
MODEL_PATH = "torch_model/model_0.7608108108108108.model"
VIDEO_PATH = "cases/s_10_1_new_3.MOV"
PROTO_FILE_PATH = "caffe_model/pose_deploy.prototxt"
WEIGHTS_FILE_PATH = "caffe_model/pose_iter_102000.caffemodel"

class Evaluation():
    '''
    Carry out the whole evaluation procedure, including
    video modification, keypoints extraction, data preparation
    result generation.

    Parameters:
        video_path (str): the path of the input video
        model_path (str): the path of pre-trained model
        proto_file_path (str): the path of the proto file
        weights_file_path (str): the path of the weight file
        threshold (double): the detection threshold. the higher the 
        more accurate. default=0.5
    '''
    def __init__(self, video_path, model_path, proto_file_path, weights_file_path, threshold=0.5):
        self._video_path = video_path
        #self._model_path = model_path
        self._model = NotSimpleNet()
        self._model = torch.load(model_path)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._proto_file_path = proto_file_path
        self._weights_file_path = weights_file_path
        self._threshold = threshold

    def _start_evaluation(self, log_file):
        '''
        Initiate the evaluation procedure
        '''
        t = time.time()
        print("Evaluation has been initialized")
        
        #extraction = CoorExtraction(self._proto_file_path, self._weights_file_path, self._video_path, self._threshold)
        #log_file = extraction._run_extraction()
        #log_file = "cases/s_9_9_new_1.txt"
        csv_generation = dp.CSV_Generation(log_file)
        #csv_file = "cases/s_10_1_new_3.csv"
        csv_file = csv_generation._start_csv()
        input = self._data_loader(csv_file)
        prof, conf = self._eval_result(input)
        print("Proficency: " + prof)
        print("Confidence: {}".format(conf))
        print("Total time taken for evaluation: {:.3f}".format((time.time() - t) / 60) + " MINUTES")
        return prof, conf

    def _data_loader(self, csv_path):
        '''
        Read the csv file to generate the test set
        Return the input as Tense array

        Parameter:
            csv_path (str): the path of the input csv file
        '''
        xy = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
        len = xy.shape[0]
        input = torch.from_numpy(xy[:,0:])
        return input

    def _eval_result(self, testset):
        '''
        Generate the evaluation result..
        Return the proficiency level and detection confidence

        Parameter:
            testset (tense[]): the dataset read from the csv file
        '''
        self._model.eval()
        predictions = []

        for data in testset:
            data = data.to(self._device)
            outputs = self._model(data)
            prediction = outputs.data.max(1)[1].cpu().numpy()[0]
            predictions.append(prediction)

        result = [0,0,0]
        for predict in predictions:
            result[predict] += 1

        if result[0] > result[1] and result[0] > result[2]:
            return "New", (result[0] / len(predictions))
        elif result[1] > result[0] and result[1] > result[2]:
            return "Fluent", (result[1] / len(predictions))
        else:
            return "Skilled", (result[2] / len(predictions))

#For testing purposes only
'''
if __name__ == "__main__":
    evaluation = Evaluation(VIDEO_PATH, MODEL_PATH, PROTO_FILE_PATH, WEIGHTS_FILE_PATH)
    evaluation._start_evaluation()
    print("Finish")
'''
