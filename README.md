# Machine Learning for COVID-19 Radiography Diagnosis using Convolutional Neural Networks
![COVID-32](https://user-images.githubusercontent.com/39009079/224980789-ea399f73-49ad-4681-aa4f-4fff0d58cc43.png)

This project implements three approaches to Convolutional Neural Networks (CNNs) to classify chest radiography images into four classes: COVID-19, Normal, Lung Opacity and Viral Pneumonia. The dataset used in this project is the [COVID-19 Radiography Dataset](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database), which contains chest radiography images categorized into those four classes.

## Dataset
The dataset is already included in the repository under the COVID-19_Radiography_Dataset directory. The dataset contains 4 sub-directories, one for each class. The dataset was split into four subsets, `devel_ds`, `train_ds`, `val_ds`, and `test_ds`, with `train_pct=0.6`, `val_pct=0.2`, and `test_pct=0.2`.

## Models
The code defines three different convolutional neural network models: simple_cnn, deeper_cnn, and trained_cnn.
1. `CNN1`: A basic CNN architecture consisting of a small amount of convolution layers and filters, resulting in 2,805,268 network parameters.
2. `CNN2`: A larger and slightly more sophisticated model with more convolution layers and filters, resulting in 11,336,228 network parameters.
3. `resized_EfficientNetB0`: An EfficientNetB0 model with a resizing layer added to adapt to the input image size. This model is a smaller (5,288,548 parameters) but efficient model that is pre-trained on ImageNet.

## Structure
The program begins with two constants: DATA_DIR, which specifies the path to the x-ray folder, and DISPLAY_CONFUSION_MATRICES, which is a boolean that determines whether or not confusion matrices should be calculated and displayed.
To improve code readability and reusability, a data class and several helper functions are used throughout the code.

* `ProcessedDataset` is used as a simple data structure to organize and access the dataset and its individual useful fields.
* `clamp` is used for input validation on percentages of subsets of the data.
* `resized_EfficientNetB0` returns an EfficientNetB0 model with an additional input plane transform to 244x244.
* `display_confusion_matrix` accepts a trained model and a test dataset, and it calculates and displays the confusion matrix.
* `execute_cnn` compiles the chosen input model, runs the fit method, and calls `display_confusion_matrix`.

Finally, the code defines three additional functions: `simple_cnn`, `deeper_cnn`, and `trained_cnn`. These functions construct the aforementioned neural networks with the required parameters.

## Results
### CNN1: **83% accuracy**
![image](https://user-images.githubusercontent.com/39009079/224971753-2c546758-2d13-4e1b-988e-ffcfab122e7b.png)

The fit run terminated prematurely in the 10th epoch with: training accuracy 0.9681 and **valuation accuracy 0.8338**.

### CNN2: **89% accuracy**
![image](https://user-images.githubusercontent.com/39009079/224971836-fdc9e10d-745b-4035-b95e-d1723724d45d.png)

The fit run terminated prematurely at epoch 10 with: training accuracy 0.9458 and **valuation accuracy 0.8930**.

### resized_EfficientNetB0: **94% accuracy**
![image](https://user-images.githubusercontent.com/39009079/224972750-8d3e7f13-cc45-4bb0-a1de-4633f437ae18.png)

The fit run finished in the 5th epoch with: training accuracy 0.9676 and **valuation accuracy 0.9441**.

## Acknowledgements
This project was completed as part of a course on Machine Learning and Applications at Harokopio University.
