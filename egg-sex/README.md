# Reproducing Results for Egg Sex Classification Models

This directory contains resources and scripts to reproduce the results from the egg sex determination project. Follow the instructions below to run the respective models.

## Models and Instructions

### 1. Cross-Validation Model
- **File:** `egg_sex_cross_validation_model.ipynb`  
- **Description:** Runs the cross-validation model for evaluating performance metrics.  
- **Steps to Run:**  
  1. Open the notebook in Google Colab: `egg_sex_cross_validation_model.ipynb`.  
  2. Execute the cells sequentially to reproduce cross-validation results.  

### 2. Train, Validation, and Test Split Model
- **File:** `egg_sex_train_validation_test_model.ipynb`  
- **Description:** Runs the model with data split into train, validation, and test sets.  
- **Steps to Run:**  
  1. Open the notebook in Google Colab: `egg_sex_train_validation_test_model.ipynb`. Includes visualizations of the data and results.
  2. Execute the cells sequentially to generate and validate the train/test split results.  

### 3. YOLOv5 Model
- **File:** `egg_sex_YOLOv5.ipynb`  
- **Description:** Implements YOLOv5 for object detection-based egg sex determination.  
- **Steps to Run:**  
  1. Open the notebook in Google Colab: `egg_sex_YOLOv5.ipynb`.  
  2. Execute the cells sequentially to run the YOLOv5 model and visualize results.

### 4. Heatmap
- **File:** `egg_sex_heatmap.ipynb`  
- **Description:** Visualize model attention heatmaps using Grad-CAM.  
- **Steps to Run:**  
  1. Open the notebook: `egg_sex_heatmap.ipynb`.
  2. Download data with testing, training, and validation pickle files to environment and set dataset_path.
  3. Download saved model pytorch file to environment and set model_path. 
  4. Execute the cells sequentially to run and visualize results.  

## Notes
- The data is not provided in this repository. Please reach out for access.
