# Brain Tumor Segmentation using UNet with EfficientNetB7

This repository contains a Jupyter Notebook (`brain-tumor-segmentation-unet-efficientnetb7.ipynb`) that demonstrates the process of brain tumor segmentation using a UNet architecture with an EfficientNetB7 backbone. The notebook covers data preprocessing, model building, training, and evaluation of the segmentation model.

## Table of Contents


1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Building](#model-building)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Conclusion](#conclusion)
9. [References](#references)
    
## Introduction

Brain tumor segmentation is a critical task in medical image analysis. This project uses the UNet architecture, a popular deep learning model for semantic segmentation, enhanced with an EfficientNetB7 backbone to improve the segmentation accuracy. The model is trained and evaluated on a dataset of brain tumor images.

![UNet Architecture with EfficientNetB7 Backbone]( )![Architecture-of-proposed-Eff-UNet-with-EfficientNetB7-framework-for-semantic](https://github.com/user-attachments/assets/ffb7d9c7-d13c-42da-aca2-fec6118a9efa)


## Dataset

The dataset used for this project is the **LGG MRI Segmentation** dataset, which is available on Kaggle. The data is stored in the following directory:

`/kaggle/input/lgg-mri-segmentation`

This dataset consists of MRI images of gliomas (brain tumors) and their corresponding segmentation masks.

- **Data Folder Structure**:
  - **Images**: MRI images of brain scans.
  - **Masks**: Segmentation masks that mark the tumor regions.




## Data Preprocessing

In this project, data preprocessing is an essential step before training the model. The dataset consists of both image data and associated metadata (CSV file). Below are the key preprocessing steps applied to the data:

### 1. File Collection

The first step involves gathering all the necessary files from the dataset directory. The file paths for the MRI image files are collected based on specific naming patterns, such as `.gz` and `.tif` file types.

```python
files_dir = '/kaggle/input/lgg-mri-segmentation/lgg-mri-segmentation/kaggle_3m/'
file_paths = glob.glob(f'{files_dir}/*[a-g].tif')
```

### 2. Reading Metadata

The metadata (such as tumor location, grade, and other clinical data) is stored in a CSV file. This file is read into a pandas DataFrame for further processing.

```python
csv_path = '/kaggle/input/lgg-mri-segmentation/kaggle_3m/data.csv'
df = pd.read_csv(csv_path)

df.info()
```

This gives an overview of the dataset, including the number of missing and non-null values for each column.

### 3. Handling Missing Values

Missing values are handled using the `SimpleImputer` from Scikit-learn. The strategy used for imputation is the **most frequent** value, which fills missing entries with the most common value in each column.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="most_frequent")
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

This step ensures that the DataFrame is fully populated, and any missing values are handled appropriately to avoid issues during training.

### 4. Final Preprocessed Data

After imputation, the DataFrame contains the cleaned data, ready to be used for model training. The processed data ensures that all features are filled, and the data is now ready to be used for the next steps, such as model building.

The resulting data will look like this:

```text
   patient_id  MethylationCluster  ... ethnicity  death01
0   TCGA_CU_AA_494  1.0    2.0     ...   1.0
1   TCGA_CU_AA_495  1.0    2.0     ...   1.0
2   TCGA_CU_AA_496  1.0    2.0     ...   1.0
...
```

This ensures that there are no missing values in the dataset and all features are ready for training the model.

---

This content explains the preprocessing steps that were performed, including file handling, reading the CSV metadata, missing value imputation, and the final cleaned data. You can modify the code snippets based on your actual implementation if needed.

  



## Model Building

The model is built using the **UNet** architecture with an **EfficientNetB7** backbone. The EfficientNetB7 model is pre-trained on ImageNet and serves as the encoder, which helps in extracting high-level features from the input MRI images. The decoder part of the UNet model reconstructs the segmentation mask.

### UNet with EfficientNetB7 Architecture


- **UNet Architecture Diagram:**

  
  ![unet](https://github.com/user-attachments/assets/b1c5c8ec-3c20-4e57-be69-af848b29ea5e)



The model consists of the following parts:
- **Encoder**: EfficientNetB7, which extracts features from input images.
- **Bottleneck**: A convolutional layer to process the extracted features.
- **Decoder**: Upsampling and convolution layers to generate the segmentation map.
- **Output Layer**: A sigmoid activation function to predict the probability of tumor presence.

## Training

The model is trained using a combination of:
- **Loss Function**: Binary Cross-Entropy (for binary segmentation).
- **Optimizer**: Adam optimizer with a learning rate of 0.0001.
- **Metrics**: Dice Coefficient to evaluate segmentation performance.

### Training Process

- The model is trained for a specified number of epochs (e.g., 50).
- Early stopping is used to prevent overfitting.

### Example of Training Progress

- **Training Loss**
  
![TL](https://github.com/user-attachments/assets/78538b57-9700-44a8-920a-2d120711b538)

- **Dice Score**
  
![DS](https://github.com/user-attachments/assets/d7936d0c-5c7f-4c15-bb4a-85a9e3690f21)


## Evaluation

After training, the model is evaluated on a test set of brain tumor MRI images. The evaluation metrics include:

- **Dice Coefficient**: A measure of overlap between the predicted mask and the ground truth mask.
- **IoU (Intersection over Union)**: Another metric to evaluate the segmentation accuracy.

### Example of Segmentation Results

#### Predicted Mask vs Ground Truth

- **Ground Truth:**

  ![GDT](https://github.com/user-attachments/assets/bcdd95a0-54b3-44e4-97b0-f091ee95ba55)


- **Predicted Mask:**

  ![mask](https://github.com/user-attachments/assets/1c3e70e1-2c64-4b93-a01f-909954f8678d)


#### Performance Metrics

- **Dice Coefficient**: 0.93
- **IoU**: 0.92

## Results

The model demonstrates strong performance in segmenting brain tumors from MRI scans. The segmentation masks closely resemble the ground truth, achieving a high Dice Coefficient and IoU score. 



## Conclusion

In this project, a UNet model with an EfficientNetB7 backbone was successfully used for brain tumor segmentation. The results show promising segmentation accuracy, and this model can be further improved by tuning hyperparameters, using more advanced augmentation techniques, or training on a larger dataset.



### How to Run the Notebook

1. Clone this repository:

   ```bash
   git clone https://github.com/manushukla2/Brain--Tumor-Segmentation-Academics-Minor-project-.git



**Note**: Replace `path/to/your/image.png` with the actual paths to your images in the repository. To ensure that GitHub renders them correctly, you can upload your images to the repository and link them accordingly.

This README file provides a detailed overview of your project, including explanations of the architecture, training process, results, and how to run the notebook.
