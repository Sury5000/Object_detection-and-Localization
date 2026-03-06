# Object Detection and Localization with ConvNeXt (PyTorch)

This project explores how to extend an image classification model into an object detection system. Starting from a pretrained ConvNeXt backbone, the model is first fine-tuned for flower classification and then extended with a localization head to predict bounding boxes around objects.

The project demonstrates the workflow of building a simple detection system including dataset preparation, transfer learning, synthetic detection dataset generation, bounding box prediction, and visualization.

---

# Project Objectives

The goals of this work were:

- Fine-tune a pretrained CNN for flower classification
- Extend a classification network into a localization model
- Learn bounding box prediction using regression
- Build a synthetic object detection dataset
- Train a multi-task model for both classification and localization
- Visualize predicted bounding boxes on images

---

# Dataset

## Oxford Flowers102 Dataset

The primary dataset used in this project is **Oxford Flowers 102**.

Dataset characteristics:

- 102 flower categories  
- Large variation in flower types and image backgrounds  
- Thousands of images across training, validation, and test sets  


# Model Architecture

## Backbone Network

The backbone model used is:

**ConvNeXt Base**

Loaded from:

```python
torchvision.models.convnext_base
```

with pretrained **ImageNet weights**.

This allows the model to leverage powerful visual features learned from large-scale datasets.

---

# Transfer Learning Strategy

A staged training strategy was used to adapt the pretrained network.

## Step 1 — Replace the Classifier

The original ConvNeXt classifier outputs **1000 classes** (ImageNet).  
It was replaced with a new linear layer to predict **102 flower classes**.

```
Linear(1024 → 102)
```

---

## Step 2 — Freeze Backbone Layers

Initially all backbone parameters were frozen:

```python
param.requires_grad = False
```

Only the classifier head was trained.

Purpose:

- Preserve pretrained feature representations
- Train the classifier quickly without disturbing backbone weights

---

## Step 3 — Fine-Tune the Network

After the classifier began learning useful representations, the backbone layers were unfrozen:

```python
param.requires_grad = True
```

This allowed the entire network to adapt to the Flowers102 dataset.

---

# Training Setup

Training configuration:

- **Optimizer:** AdamW  
- **Loss Function:** CrossEntropyLoss  
- **Evaluation Metric:** Multiclass Accuracy (torchmetrics)

The training loop tracked:

- Training loss
- Training accuracy
- Validation accuracy

Multiple epochs were run to progressively improve classification performance.

---

# Extending the Model for Object Detection

Once a strong classifier was obtained, the model was extended to perform **object localization**.

A new architecture called **FlowerLocator** was implemented.

The model now predicts:

1. Flower class
2. Bounding box location

---

# FlowerLocator Architecture

The architecture shares a common feature extractor but branches into two heads.

```
Input Image
     ↓
ConvNeXt Feature Extractor
     ↓
Global Average Pooling
     ↓
Shared Feature Vector
     ↓
 ┌───────────────┬───────────────┐
 │               │
Classification    Localization
Head              Head
(102 classes)     (4 values)
```

---

# Bounding Box Prediction

Bounding boxes are predicted using the format:

```
(cx, cy, w, h)
```

Where:

- `cx` → center x coordinate
- `cy` → center y coordinate
- `w` → width of the box
- `h` → height of the box

This representation simplifies regression learning compared to corner coordinates.

---

# Synthetic Detection Dataset

The Flowers102 dataset does not provide bounding box annotations.  
To simulate a detection task, a **synthetic detection dataset** was created.

A custom dataset class called:

```
SyntheticFlowerDetection
```

was implemented.

---

## Synthetic Dataset Generation

For each image:

1. A flower image is selected.
2. A random bounding box is generated.
3. The bounding box represents the object location.
4. The image, bounding box coordinates, and class label are returned.

This allows the model to learn localization along with classification.

---

# Multi-Task Learning

The model simultaneously learns:

- **Classification** (flower type)
- **Localization** (bounding box position)

Two losses are combined during training:

### Classification Loss

```
CrossEntropyLoss
```

Measures how accurately the model predicts the flower class.

### Localization Loss

```
Mean Squared Error (MSE)
```

Measures how accurately the predicted bounding box matches the target box.

The total loss is computed as:

```
Total Loss = Classification Loss + Localization Loss
```

---

# Visualization of Predictions

After training, predicted bounding boxes are visualized on images.

Visualization steps:

1. Select test images
2. Predict bounding box coordinates
3. Convert normalized coordinates to pixel values
4. Draw bounding boxes on images
5. Display predictions

This allows visual inspection of how well the model learned localization.

---

# Key Learnings

Through this project I learned:

- How to adapt pretrained models using transfer learning
- How classification networks can be extended for detection tasks
- How to implement multi-task learning with multiple outputs
- How bounding box regression works in object detection
- How synthetic datasets can simulate detection problems
- How to visualize model predictions for qualitative evaluation

---

# Conclusion

This project demonstrates a complete progression from **image classification to object detection** using a pretrained ConvNeXt backbone.

The final model integrates:

- Transfer learning
- Multi-task learning
- Bounding box regression
- Synthetic detection dataset generation
- Visualization of predictions

This work provides a strong practical foundation for understanding how modern object detection systems extend from convolutional neural network backbones.
