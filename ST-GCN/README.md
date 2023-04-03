This folder contains files for the training and testing the ST-GCN model on the ASL Citizen Dataset.  

### File Description:

**pose.py**: This contains code to extract MediaPipe Holistic keypoints from the videos. It saves keypoints as .npy files.

**architecture [subfolder]** contains code for the ST-GCN architecture. It was sourced from [link]

**pose_transforms.py**: This contains code for transforms that are applied to the pose graph while training and testing such as shear transform.

**aslcitizen_dataset_pose.py**: This is the dataset file. It loads the keypoints from data csvs, downsampling and padding videos as needed, and returns samples for testing along with one-hot encoded labels.

**stgcn_training.py**: This file contains code to train a ST-GCN model on the ASLCitizen dataset. 

**stgcn_testing.py**: This file contains code to test the trained ST-GCN model on the ASLCitizen test dataset. For the task of dictionary retrieval, the goal is to get a ranked list of glosses for any given video. This code outputs the average top-1, top-5, top-15, top-20 accuracy, Discounted Cumulative Gain (DCG), Mean Reciprocal Rank on the whole dataset. It additionally outputs two kinds of confusion matrices to help with analysis -- one complete confusion matrix, and a confusion matrix mini that highlights top five confusions for a given gloss.

### Instructions:

Download the dataset from [link] and unzip. 

To extract the MediaPipe Holistic keypoints from the videos, open pose.py. Update paths to videos, data_csvs and destination path (lines 14-16). Use the following command on the command line. It takes approximately 7 days to complete extracting all keypoints. 

```
python3 pose.py
```

Open stgcn_training.py and update the file paths to dataset and datacsvs as needed (lines 35-38). Update names for log and weights folders as needed (lines 40-43). 

To train, use the following command on the command line. With a single GPU, it takes approximately 3 days to complete training. 
```
python3 stgcn_training.py 
```

Once done with training, you can find the saved model weights in a folder (*'saved_weights'* as default). The weights are named such that the last set of digits is the validation accuracy (e.g., *'trainingfull_jan1a75_0.736444.pt'* had validation accuracy of 73.64%). 

To evaluate chosen model weights, open stgcn_testing.py. Update the file paths to dataset and datacsvs as needed (lines 54-56). Update names for output files as needed (lines 57-58). Update path to model weights (line 79). 

To test, use the following command on the command line. With a single GPU, it takes approximately 45 minutes to complete testing.
```
python3 stgcn_testing.py
```

The results can be found in .txt and .csv files generated in the same folder. 


