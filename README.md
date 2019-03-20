# Video_Based_Guitarist_Performance

Currently, it is an unfinished project. More files and examples will be uploaded later on.

This project aims to classify the performance of a guitarist according to his/her finger gesture in 2D. Now, the model is able to achieve around 70% accuracy.

#Testing

1. Make sure your the version of your libraries are the same (IMPORTANT!!!)
2. Modify the evaluation.py by commenting the video processing part and using the txt as input to generate the csv file directly. Since the video and detection models are not uploaded
3. Use the evaluation.py to test or comment the main function in it and run the main.py for testing purposes

# Library Involved:

OpenCV v=4.0.0
1. Frame-by-frame video processing
2. Hand keypoints detection using pretrained Caffa model

PyTorch v=1.0.1
1. The deep learning framework responses for the training of perfermance classifier

CUDA v=1.0
1. Enhance the pytorch operation 

Python v=3.6.5
1. The language the project is implimented in 
