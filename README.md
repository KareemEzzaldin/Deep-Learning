# Deep-Learning
Pneumonia Detection from Chest X-Rays with Deep Learning

    Introduction
This project uses chest X-ray images to train a pneumonia detection model. The data is split into three sets:
Training: 1341 Normal, 3875 Pneumonia
Testing: 234 Normal, 390 Pneumonia
Validation: 8 Normal, 8 Pneumonia

The dataset is well imbalanced. Pneumonia cases are greater than normal cases in training and testing. This imbalance must be addressed to avoid a biased model.

    Data Preprocessing and Augmentation
Every image is resized to 224x224. To increase X-ray clarity, CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied. It does the following:Transfers image to grayscale, Improve, local contrast, and Transfers it back to RGB. Every image is also normalized by pixel value division with 255. Training images undergo extensive augmentation too:Rotations, Shifting, Zooming, Shearing, and Flipping.Validation and test images undergo only CLAHE and normalization. This ensures the model to generalize more and learn robust features from varied samples.

    Image Data Generators
Used Keras' ImageDataGenerator to feed data to the model.
Training batch size: 64
Validation batch size: 32
Test batch size: 1
Loaded images in batches and augmented while running. This keeps memory consumption low and training efficient.

    Model Design
Used ResNet152 with ImageNet pretrained weights. Removed the top layers and added custom layers:
Global Average Pooling
Dense (256 neurons, ReLU)
Dropout (0.5)
Final Dense (1 neuron, Sigmoid)
First, the ResNet layers were frozen. The custom layers were trained first alone.
Used:
Adam optimizer
Binary cross-entropy loss
Learning rate: 1e-4
This setup allows the model to learn basic patterns without interfering with pretrained weights.

    Class Imbalance Handling
Used sklearn to calculate class weights. Since Pneumonia cases dominate the dataset, this prevents the model from always predicting Pneumonia. Without this, the model would learn to cheat by always choosing the majority class.

Training
The model was fine-tuned in two stages.
Stage 1:
Trained for 30 epochs with early stopping and learning rate reduction. ResNet layers were frozen.

Stage 2:
Unfroze the last 30 layers of ResNet and trained for 10 additional epochs. Reduced the learning rate to 1e-5.

This fine-tuning improves accuracy by enabling the deeper layers to learn X-ray features.

    LIME Explanation
Applied LIME to explain a prediction. LIME disturbs the input image and observes how the model responds. Then it emphasizes what portions of the image were most crucial for that prediction. Displayed the explanation on top of the original image. Which areas the model used is displayed.


