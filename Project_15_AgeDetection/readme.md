# Project: Age Detection

## Introduction
In this project, we delve into the realm of computer vision and data science to address a challenge faced by the supermarket chain Good Seed. The objective is to leverage computer vision methods to determine the age of individuals purchasing alcohol, ensuring compliance with alcohol laws and preventing sales to underage customers. By analyzing a dataset containing photographs of individuals along with their ages, we aim to build and evaluate a model capable of verifying people's ages accurately.

## Data Description
The dataset is stored in the `/datasets/faces/` folder, which contains:
- The `final_files` folder with 7.6k photos
- The `labels.csv` file with labels, with two columns: `file_name` and `real_age`

## EDA
- Loaded and inspected the data, checking for duplicates and missing values.
- Plotted the age distribution, revealing a right-skewed distribution with the majority of ages between 10-40 years.
- Displayed a sample of images, noting variations in dimensions, orientation, and placement.

## Findings
- Augmenting the images can enhance performance. Suitable augmentations include horizontal flips, slight rotations, and adjustments for centering faces.
- The black borders around some images raise concerns, but instead of removing them, adjusting images horizontally and vertically is preferable.
- Implemented a 20% zoom to account for varying distances of faces from the camera.

## Modelling
### Functions for Model Training on GPU
Defined functions to load training and test data, create the model, and train it on the GPU platform. The model is built using the ResNet50 architecture with additional layers for fine-tuning.

### Prepare the Script to Run on the GPU Platform
Prepared and saved a script (`run_model_on_gpu.py`) to run the defined functions on the GPU platform.

### Model Training
Trained the model for 10 epochs, achieving the following results:
```python
Epoch 1/10 
474/474 - 128s - loss: 271.5880 - mae: 12.6584 - val_loss: 481.5121 - val_mae: 14.1965 

Epoch 2/10 
474/474 - 127s - loss: 204.2316 - mae: 10.8302 - val_loss: 274.7711 - val_mae: 10.7063 

Epoch 3/10 
474/474 - 128s - loss: 177.2655 - mae: 10.1848 - val_loss: 180.3962 - val_mae: 9.7349 

Epoch 4/10 
474/474 - 127s - loss: 159.1187 - mae: 9.6414 - val_loss: 203.3812 - val_mae: 9.9163 

Epoch 5/10 
474/474 - 128s - loss: 146.5263 - mae: 9.2951 - val_loss: 157.7959 - val_mae: 9.2561 

Epoch 6/10 
474/474 - 128s - loss: 138.4282 - mae: 9.0660 - val_loss: 155.0604 - val_mae: 9.2219 

Epoch 7/10 
474/474 - 128s - loss: 132.6969 - mae: 8.8797 - val_loss: 156.1251 - val_mae: 9.2651 

Epoch 8/10 
474/474 - 128s - loss: 124.4189 - mae: 8.6180 - val_loss: 155.2914 - val_mae: 9.2344 

Epoch 9/10 
474/474 - 128s - loss: 120.2625 - mae: 8.4833 - val_loss: 145.0242 - val_mae: 8.9851 

Epoch 10/10 
474/474 - 128s - loss: 117.0227 - mae: 8.3784 - val_loss: 140.6544 - val_mae: 8.8673 

Test MAE: 6.9393
## Conclusions and Recommendations

With a mean absolute error (MAE) of about 7, the model typically predicts a person’s age within a 7-year margin from their actual age. The primary challenge lies in distinguishing individuals under 21 from those who are 21 or older. Although the model shows promising results, relying solely on it for age verification carries significant risks due to the high stakes involved in legal compliance.

### Recommendations

- Consider using the system to identify customers clearly over a certain age, such as 35, to standardize age verification across all locations.
- Collect additional demographic details like race and gender to analyze the error distribution and understand model performance better.
- Use the system alongside mandatory ID checks for a year, collecting data on predicted versus actual ages to evaluate real-world performance.
- Continue to ensure compliance through traditional ID checks for legal age verification while utilizing the system for low-risk applications like targeted marketing.