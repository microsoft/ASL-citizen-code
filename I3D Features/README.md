This folder contains files for the I3D Features experiment on the ASL Citizen Dataset.

### File description:

**pytorch_i3d.py** : This is the network file, containing code for i3d architecture. It was sourced from [PyTorch I3D](https://github.com/piergiaj/pytorch-i3d)

**videotransforms.py**: This contains code for videotransforms that are applied while training and testing such as random crop and center crop.

**aslcitizen_dataset.py**: This is the dataset file. It loads the videos from data csvs, downsampling and padding videos as needed, and returns samples for testing along (excluding labels).

**extract_features_from_i3d.py**: Using a trained model, this saves i3d features for all videos in the dataset as a tsv file. 

**test_features.py**: This file implements the nearest neighbour search on the extracted train and test features. For the task of dictionary retrieval, the goal is to get a ranked list of glosses for any given video. This code outputs the average top-1, top-5, top-15, top-20 accuracy, Discounted Cumulative Gain (DCG), Mean Reciprocal Rank on the whole dataset. It additionally outputs two kinds of confusion matrices to help with analysis -- one complete confusion matrix, and a confusion matrix mini that highlights top five confusions for a given gloss.

### Instructions:

Open extract_features_from_i3d.py and update the type of model and model weights to use for experiment (lines 46-58). 

To extract features for a given dataset split, use the following command on the command line
```
python3 extract_features_from_i3d.py [path to data] [path to data csv] [output .tsv file name]
```

Extract I3d features for both the training data split and test data split as shown below:
```
python3 extract_features_from_i3d.py ../data/ASLCitizen/videos/ ../data_csv/aslcitizen_training_set.csv train_features_aslcitizen.tsv
```
```
python3 extract_features_from_i3d.py ../data/ASLCitizen/videos/ ../data_csv/aslcitizen_test_set.csv test_features_aslcitizen.tsv
```
To run nearest neighbour algorithm on the dataset and output dictionary metrics, use the following command on the command line
```
python3 test_features.py [path to train data tsv] [path to test data tsv] [output file name, no extension]
```
```
python3 test_features.py train_features_aslcitizen.tsv test_features_aslcitizen.tsv features_aslcitizen
```
The results can be found in .txt and .csv files generated in the same folder.
