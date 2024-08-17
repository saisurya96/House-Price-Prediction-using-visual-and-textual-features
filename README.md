# House Price Prediction with Scene Classification and Ablation Studies

## Overview

This project aims to predict house prices using a combination of image data and traditional numerical features. The project employs a scene classification model (AlexNet trained on Places365) to extract features from images of different rooms in the houses, such as the living room, kitchen, and exterior. These extracted features are then used to build and refine predictive models for house prices.

## Project Structure

The project is organized into two main parts:

1. **House Price Prediction:**

   - **Scene Classification:** We use a pre-trained AlexNet model trained on the Places365 dataset to classify images of different parts of the house (e.g., bathroom, bedroom, living room, kitchen, exterior). The extracted features from these images serve as inputs to the price prediction model.
   - **Feature Engineering:** The features extracted from the images are further processed, including L2 normalization, to create meaningful inputs for the predictive models.
   - **Modeling:** Various machine learning models, including Support Vector Regression (SVR), are trained using the image features along with traditional numerical features (like square footage, number of bedrooms, etc.).

2. **Ablation Studies:**

   - **Feature Splitting:** The image data is split into five different sets based on the room type (bathroom, bedroom, living room, kitchen, exterior). This allows for analyzing the impact of each room type on the overall prediction accuracy.
   - **Performance Evaluation:** The models are evaluated using metrics such as Mean Absolute Error (MAE) to determine the contribution of each set of features to the overall performance.

## Notebooks

- **`house_price_prediction.ipynb`**: This notebook contains the primary code for loading the pre-trained AlexNet model, extracting features from house images, and training the predictive models.
- **`price_prediction+ablation_studies.ipynb`**: This notebook is focused on conducting ablation studies to analyze the impact of each room's image features on the house price prediction accuracy.

## Key Components

- **AlexNet Model**: A convolutional neural network used for scene classification.
- **L2 Normalization**: Applied to the image features to ensure consistency and improve model performance.
- **Support Vector Regression (SVR)**: A machine learning algorithm used to predict house prices based on the processed features.

## Installation

To run the notebooks, you need to have Python installed along with the following libraries:

```bash
pip install torch torchvision numpy pandas scikit-learn pillow.
```
## How to Use

1. Clone the repository to your local machine.
2. Run the `house_price_prediction.ipynb` notebook to generate the image features and train the initial predictive model.
3. Run the `price_prediction+ablation_studies.ipynb` notebook to perform the ablation studies and analyze the impact of different room types on price prediction.

## Results

The ablation studies revealed that the **living room** and **kitchen** images contributed the most to the model's predictive power, highlighting the importance of these spaces in determining house prices. The **exterior** images also played a significant role, particularly in differentiating properties based on curb appeal and location features.
