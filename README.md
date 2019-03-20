# Video_Based_Guitarist_Performance

Currently, it is an unfinished project. More files and examples will be uploaded later on.

This project aims to classify the performance of a guitarist according to his/her finger gesture using 2D videos. Now, the model performs 0.84 accuracy with 0.769 confidence among 100 samples. 

# Testing

1. Make sure the version of your libraries are exactly the same. (IMPORTANT!!!)
2. The train_1.zip contains 100 test cases. Simply unzip it to the folder. 
3. Run main.py for testing purposes. It takes use the txt files (in the train_1.zip) as the input to generate the csv files directly, since the video and hand gesture detection models are not uploaded. (The txt files are generated coor_extraction.py from videos. Examples will be uploaded later on.)
4. Modified the evaluation.py for custom input and settings (Optional).

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
