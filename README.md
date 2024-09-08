# 🔢 Handwritten Persian Number Recognition using TensorFlow

## 📘 Overview

This project is designed to recognize **handwritten Persian digits** using **machine learning** techniques, specifically utilizing **Artificial Neural Networks (ANN)** and **Convolutional Neural Networks (CNN)**. The system has been optimized through **data preprocessing**, **model training**, and **data augmentation**, ultimately achieving a high accuracy of **95%**. Additionally, the program can handle multi-digit numbers by detecting individual digits using **contours** and recognizing them one by one. 

This project was implemented in **Python** with the help of **TensorFlow** for machine learning, and **OpenCV** for image processing.

## ✨ Key Features

- 🔢 **Single-Digit Persian Number Recognition**: Recognizes individual Persian digits (0-9) from preprocessed images.
- 🧠 **ANN & CNN Models**: Includes two models for training:
  - ANN: Provides basic digit recognition with 80% accuracy.
  - CNN: A deeper network for more complex feature extraction, achieving 90% accuracy.
- 🚀 **Data Augmentation**: Applies augmentation techniques to increase the variability in training data, improving accuracy to **95%**.
- 🔍 **Multi-Digit Number Recognition**: Capable of identifying multi-digit numbers by isolating and processing each digit using contour detection.
- 🎯 **Edge Enhancement**: Uses image preprocessing techniques like grayscale conversion and erosion filtering to improve digit clarity and recognition.
- 📈 **Model Evaluation**: Includes evaluation metrics and performance reporting for both ANN and CNN models, showcasing their learning curves.

## 🛠️ Project Workflow

### 1. **Image Collection**
- **Dataset**: A dataset of **20 images per digit** (0-9) was collected. These images represent handwritten Persian digits written in various styles.
- **Image Storage**: The processed and flattened images are saved in a format that can be read using **OpenCV** for easy reuse during model training.

### 2. **Preprocessing**
- **Image Resizing**: All images are resized to **28x28** pixels to maintain a uniform input size.
- **Grayscale Conversion**: Images are converted to grayscale for better clarity and to reduce computational complexity.
- **Smoothing**: The images are smoothed to reduce noise and improve model accuracy.
- **Edge Enhancement**: A **2x2 erosion filter** is applied to emphasize the edges of the digits, making them more distinguishable during training.

### 3. **Artificial Neural Network (ANN)**
   - **Model Structure**: The ANN model consists of two main layers:
     - **Dense Layer (128 Neurons, ReLU activation)**: Takes the flattened 28x28 images as input and processes them through a densely connected layer.
     - **Output Layer (10 Neurons, Softmax activation)**: Outputs the predicted class (digit) as a probability distribution.
   - **Training & Performance**: After training on the preprocessed images, the model achieves an accuracy of **80%**. The ANN serves as a simpler model for quick digit recognition tasks.

     ![](https://github.com/ParsaJahantab/Handwritten-Persian-Number-Recognition/blob/main/numbers/test4.png)

### 4. **Convolutional Neural Network (CNN)**
   - **Deeper Learning**: A **13-layer CNN** is employed for more advanced feature extraction. The layers include multiple convolutional layers, pooling layers, and dense layers to process the images more thoroughly.
   - **Performance**: The CNN improves upon the ANN, achieving a **90% success rate**.
   - **Optimization**: By adjusting the learning rate and other hyperparameters, the model is optimized for the best performance.

     ![](https://github.com/ParsaJahantab/Handwritten-Persian-Number-Recognition/blob/main/numbers/test3.png)

### 5. **Data Augmentation**
   - **Goal**: To increase the diversity of the dataset and avoid overfitting, data augmentation techniques are applied.
   - **Techniques Used**:
     - Random rotation of images.
     - Scaling and zooming.
     - Horizontal flipping.
     - Shifting pixels within the image.
   - **Result**: With data augmentation, the accuracy of both the ANN and CNN models improves to **95%**, providing a more robust recognition system.

### 6. **Multi-Digit Recognition**
   - **Contour Detection**: For recognizing numbers with multiple digits, the program uses **contours** to detect individual digits within an image.
   - **Digit Isolation**: After detecting the contours, the system isolates each digit and feeds it into the trained model (ANN or CNN) for recognition.
   - **Practical Use**: This feature allows for the recognition of multi-digit Persian numbers, expanding the utility of the program beyond single-digit classification.

     ![](https://github.com/ParsaJahantab/Handwritten-Persian-Number-Recognition/blob/main/numbers/test1.png)

## 🚀 Installation and Setup

Follow the steps below to install and run the Persian number recognition system on your local machine:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ParsaJahantab/Handwritten-Persian-Number-Recognition.git
   ```
   
2. **copy the images next to the ipynb file**:

3. **Install the dependencies**:
   Ensure you have Python 3.x installed, then install the required packages



## 📊 Model Summary

### ANN Model:
- **Input Layer**: 28x28 grayscale image (flattened).
- **Hidden Layer**: Dense layer with 128 neurons and ReLU activation.
- **Output Layer**: Dense layer with 10 neurons, using softmax activation for classification.

### CNN Model:
- **13 layers** including convolutional layers, pooling layers, and dense layers.
- **90% accuracy** with raw data and **95% accuracy** with data augmentation.

### Performance:
- **Accuracy**: 
  - ANN: 80%
  - CNN: 90%
  - CNN with Data Augmentation: 95%
