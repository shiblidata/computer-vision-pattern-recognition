# 🍎 Fruit Image Classification Using Custom CNN

A multi-class fruit image classification project developed using a **Custom Convolutional Neural Network (CNN)** with **PyTorch**. The model was trained and evaluated using selected classes from the **Fruits-360 dataset**.

This project was developed as part of the **Computer Vision and Pattern Recognition (CVPR)** course assignment.

---

## 📌 Project Objective

The main objective of this project is to design, train, and evaluate a custom CNN model capable of classifying fruit images into multiple categories.

The project covers the complete deep learning workflow, including:

- Dataset preparation
- Image preprocessing
- Train-validation splitting
- Custom CNN architecture design
- Model training and validation
- Regularization using Dropout
- Model evaluation
- Performance visualization
- Model weight saving

---

## 📂 Dataset

**Dataset Name:** Fruits-360  
**Source:** Kaggle  
**Dataset Link:** https://www.kaggle.com/datasets/moltean/fruits

For this project, **7 fruit classes** were selected:

1. Apple Red 1
2. Banana 1
3. Cherry 1
4. Lychee 1
5. Orange 1
6. Strawberry 1
7. Tomato 1

### Dataset Distribution

| Dataset | Number of Images |
|---|---:|
| Original Training Set | 3,673 |
| Training Set (80%) | 2,938 |
| Validation Set (20%) | 735 |
| Test Set | 1,230 |
| Number of Classes | 7 |

The original training dataset was divided into **80% training** and **20% validation** data. A separate test dataset was used for final evaluation.

---

## 🖼️ Image Preprocessing

All input images were preprocessed before being passed to the CNN.

The preprocessing pipeline included:

- Resizing images to **100 × 100 pixels**
- Converting images to PyTorch tensors
- Normalizing RGB channels

The input shape of each image was:

```text
3 × 100 × 100
```

Where:

- `3` represents the RGB channels
- `100 × 100` represents the image resolution

---

## 🧠 Custom CNN Architecture

A custom CNN architecture was implemented from scratch using PyTorch.

The architecture consists of three convolutional blocks followed by fully connected layers.

```text
Input Image (3 × 100 × 100)
        |
        v
Conv2D (3 → 32)
        |
      ReLU
        |
    MaxPooling
        |
        v
Conv2D (32 → 64)
        |
      ReLU
        |
    MaxPooling
        |
        v
Conv2D (64 → 128)
        |
      ReLU
        |
    MaxPooling
        |
        v
     Flatten
        |
        v
Fully Connected (18432 → 256)
        |
      ReLU
        |
  Dropout (0.5)
        |
        v
Fully Connected (256 → 7)
        |
        v
  Class Prediction
```

### Total Trainable Parameters

**4,813,895**

Most of the trainable parameters come from the first fully connected layer, which connects 18,432 extracted features to 256 neurons.

---

## ⚙️ Hyperparameters

| Hyperparameter | Value |
|---|---|
| Image Size | 100 × 100 |
| Number of Classes | 7 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Learning Rate Type | Fixed |
| Loss Function | CrossEntropyLoss |
| Activation Function | ReLU |
| Dropout Rate | 0.5 |
| Epochs | 2 |
| Framework | PyTorch |
| Training Device | CUDA GPU |

---

## 🛡️ Regularization

**Dropout regularization** was used to reduce the risk of overfitting.

The model contains:

```python
nn.Dropout(0.5)
```

A dropout rate of `0.5` randomly disables a portion of neurons during training. This helps reduce dependency on individual neurons and encourages the model to learn more generalizable features.

---

## 🚀 Model Training

The model was trained using the **Adam optimizer** with a fixed learning rate of `0.001`.

**CrossEntropyLoss** was used as the loss function because the project is a multi-class classification problem.

### Training Results

| Epoch | Training Accuracy | Validation Accuracy |
|---:|---:|---:|
| 1 | 99.32% | 100.00% |
| 2 | 99.86% | 100.00% |

The best-performing model weights were saved during training.

---

## 📊 Test Performance

After training, the best model was evaluated on a separate test dataset containing **1,230 images**.

The final results were:

| Metric | Result |
|---|---:|
| **Test Accuracy** | **95.04%** |
| **Precision** | **96.41%** |
| **Recall** | **95.04%** |
| **F1 Score** | **94.87%** |

The results demonstrate that the custom CNN can effectively classify unseen fruit images across the selected seven categories.

---

## 📈 Model Evaluation

The model was evaluated using multiple performance metrics rather than relying only on accuracy.

The evaluation included:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix
- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss

These metrics provide a more comprehensive understanding of the classification performance of the model.

---

## 💾 Model Saving

The best-performing model weights were saved in PyTorch `.pth` format.

```python
torch.save(model.state_dict(), "best_model.pth")
```

The saved model weights can later be loaded using:

```python
model.load_state_dict(
    torch.load("best_model.pth", map_location=device)
)

model.eval()
```

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab
- CUDA GPU

---

## 📁 Project Structure

```text
CVPR-Fruit-CNN/
│
├── CNN_23-51036-1.ipynb
├── best_model.pth
└── README.md
```

> **Note:** The Fruits-360 dataset is not included in this repository. It can be downloaded directly from Kaggle using the dataset link provided above.

---

## ▶️ How to Run the Project

1. Download the Fruits-360 dataset from Kaggle.
2. Prepare the selected seven fruit classes.
3. Upload the dataset to Google Drive.
4. Open the Jupyter Notebook in Google Colab.
5. Mount Google Drive.
6. Update the training and testing dataset paths if necessary.
7. Run the notebook cells sequentially.
8. Train the custom CNN model.
9. Evaluate the trained model on the test dataset.
10. Generate the classification metrics and visualizations.
11. Save the best-performing model weights.

---

## 🔍 Results and Discussion

The custom CNN achieved a **95.04% test accuracy**, demonstrating strong classification performance on unseen fruit images.

The final training accuracy reached **99.86%**, while the validation accuracy reached **100.00%**.

Although validation accuracy was very high, the separate test dataset produced a slightly lower accuracy of **95.04%**. Evaluating the model on this independent test set provides a more realistic indication of its ability to generalize to unseen images.

The model also achieved a **96.41% precision**, **95.04% recall**, and **94.87% F1-score**.

A confusion matrix and classification report were used to further analyze the classification performance across individual fruit classes.

---

## 🔮 Future Improvements

Several improvements can be explored in future work:

- Apply additional data augmentation techniques
- Train the model for more epochs
- Perform hyperparameter tuning
- Increase dataset diversity
- Experiment with deeper custom CNN architectures
- Compare the custom CNN with pretrained architectures
- Apply transfer learning using models such as ResNet or EfficientNet
- Evaluate the model on more challenging real-world fruit images

---

##  Conclusion

This project demonstrates the development of a **Custom Convolutional Neural Network for multi-class fruit image classification using PyTorch**.

Seven fruit categories from the Fruits-360 dataset were used to train and evaluate the model. The custom CNN achieved a **95.04% test accuracy**, along with strong precision, recall, and F1-score.

The project provided practical experience with the complete deep learning workflow, including dataset preparation, image preprocessing, CNN architecture design, model training, regularization, validation, testing, performance evaluation, visualization, and model weight saving.

---

## 👤 Author

**Md. Mehedi Hasan Shibli**  
Department of Computer Science  
American International University-Bangladesh (AIUB)

---

## 🙏 Acknowledgement

This project was completed as part of the **Computer Vision and Pattern Recognition (CVPR)** course assignment.

The **Fruits-360 dataset** used in this project is publicly available on Kaggle.
