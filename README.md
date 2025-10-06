# Cornell Rock-Paper-Scissors Image Comparison Kaggle Competition

## 🏆 6th Place Solution (out of 94 teams) - 90.03 % Accuracy

This repository contains my solution for the Cornell's CS3780 Machine Learning Kaggle Competition (Spring 2025), which focused on comparing pairs of Rock-Paper-Scissors hand gesture images to determine if the first image beats the second image in the game.

Competition Link : https://www.kaggle.com/competitions/expanded-rock-paper-scissors/leaderboard

Team Name : (fm454(Farhan Mashrur) , km942(Kiyam Merali), joblink.one)

After initially training with Kaggle's TPU, I used an A100 GPU via Google Colab Pro, which was essential due to the extensive data augmentation and model complexity required to achieve competitive results.

## 📋 Challenge Description

In this competition, we were tasked with playing rock-paper-scissors with image data. Each of the 20,000 datapoints consisted of:

- Two 24×24 grayscale images
- A binary label:
  - +1 if the first image (hand gesture) beats the second image
  - -1 otherwise

The competition evaluated submissions based on accuracy, with final standings determined by performance on a private test set (approximately half of the test data was used for the public leaderboard, and the other half for the private leaderboard that determined final rankings).

## 🔍 Solution Approach

### Dataset & Processing

I created an `EnhancedRPSDataset` class that handles:

- Loading pairs of 24×24 grayscale images from provided pickle files
- Resizing images to 224×224 for the ResNet model
- Normalizing with ImageNet mean/std values ([0.485], [0.229])
- Implementing extensive data augmentation techniques after few of the images:
  - Random rotations (±30°)
  - Affine transforms (translation ±15%, scaling 85-115%)
  - Color jitter (brightness/contrast adjustments of ±25%)
  - Horizontal flips (50% probability)
  - Perspective distortions (distortion scale of 0.25 with 30% probability)
  - Gaussian blur (kernel size 3, sigma 0.1-2.0)

The augmentation strategy was crucial for improving model generalization and accuracy to unseen images. 

### Model Architecture: ImprovedResNet34Siamese

My solution uses a Siamese network architecture based on a pre-trained ResNet-34 model with several key modifications:

1. **Modified Input Layer**: Adapted the first convolutional layer to accept single-channel grayscale images instead of RGB
2. **Attention Mechanism**: Added an attention module over feature maps to focus on discriminative regions
3. **Custom FC Layers**: Implemented fully connected layers with:
   - Batch normalization
   - Dropout (0.3-0.4)
   - ReLU activations
   - Final sigmoid output for binary classification
4. **Focal Loss**: Used to address class imbalance in the dataset

### 3-Stage Training Strategy

I implemented a progressive training strategy with three distinct phases:

#### Stage 1: Initial Training (8 epochs)

- Froze the ResNet backbone
- Trained only the attention mechanism and FC layers
- Used CosineAnnealingWarmRestarts scheduler

#### Stage 2: Intermediate Layers (up to 20 epochs)

- Unfroze deeper layers (layer3 and layer4)
- Used OneCycleLR scheduler
- Implemented early stopping with patience=6
- Applied gradient clipping

#### Stage 3: Full Fine-tuning (up to 12 epochs)

- Unfroze all parameters for fine-tuning
- Used CosineAnnealingLR scheduler
- Continued early stopping approach

### Test-Time Augmentation (TTA)

To improve prediction robustness, I implemented TTA with:

- Original images
- Horizontal flips
- Small clockwise/counterclockwise rotations (±5°)
- Aggregation with a 0.5 threshold (averaging predictions across all augmentations)

## 📊 Results

- Placed 6th out of 94 teams in the competition
- Achieved high accuracy on both validation (internal) and test sets (Kaggle leaderboard)
- Final predictions were made using the best model checkpoint based on validation performance

### Competition Format

The competition followed standard Kaggle practices:

- Public leaderboard showing performance on ~50% of the test data
- Private leaderboard (determining final rankings) on the other ~50% ( We placed 6th on both the private and public leaderboard )
- Required submission of both:
  - CSV file with predictions to Kaggle
  - Full code submission to Gradescope for academic integrity verification

## 🛠️ Requirements

- PyTorch
- torchvision
- NumPy
- pandas
- scikit-learn
- matplotlib
- tqdm
- Google Colab Pro (with A100 GPU access) or equivalent high-performance GPU

## 📁 Data Format

The competition data is provided in pickle (.pkl) files:

- `train.pkl`: Contains training image pairs and labels
- `test.pkl`: Contains test image pairs without labels

Each pickle file contains a dictionary with:

- `img1`: List of first images (24×24 grayscale)
- `img2`: List of second images (24×24 grayscale)
- `label`: List of labels (+1/-1) [only in training data]
- `id`: List of IDs for test data [only in test data]

## 📁 Repository Structure

```
.
├── model.py ##the main model used for submission that I trained on Google Collab
├── 
```

##  Key Insights & Techniques

- **Strong data augmentation**: Crucial for generalization given the limited dataset size
- **Attention mechanism**: Significantly improved the model's focus on relevant image regions for comparing gestures
- **Progressive unfreezing**: The 3-stage training approach led to better convergence and prevented catastrophic forgetting
- **Focal loss**: Helped address class imbalance issues in the dataset
- **Test-time augmentation**: Provided a notable boost to final performance, especially on edge cases
- **Model selection**: Careful validation and checkpoint saving ensured selection of the most generalizable model


## 📝 License

MIT

##  Acknowledgements

- Cornell Machine Learning course (CS5780) for organizing the challenging competition


---

_Note: This project was created as part of the Cornell Machine Learning (CS5780) Kaggle competition for Spring 2025. The code and solution approach are shared for educational purposes._
