Amjad Moath Abdulhalim Ali - 202070317
Sam Saleh Al-Abdi - 202170340

---

# Tomato Ripeness Detection

This project involves detecting tomatoes and classifying their ripeness using a combination of YOLOv5 for tomato detection and a custom-trained neural network for ripeness classification. The system is implemented using TensorFlow and PyTorch.

## Project Structure

- Last_tensor.py: Script for training the ripeness classification model using TensorFlow.
- Last_demo1.py: Script for detecting tomatoes using YOLOv5 and classifying their ripeness using the trained model.

## Requirements

- Python 3.x
- TensorFlow 2.x
- PyTorch
- OpenCV
- NumPy
- Torchvision (for YOLOv5)

You can install the required libraries using the following command:
```bash
pip install tensorflow torch opencv-python numpy
```

## Dataset

The dataset consists of images of ripe and unripe tomatoes. The images are stored in the following structure:

```
dataset/
├── Images/
    ├── Riped tomato_*.jpeg
    ├── Unriped tomato_*.jpeg
```

### Preprocessing

- Images are resized to 224x224 pixels and normalized to the [0, 1] range before training.
- The dataset is split into training and testing sets using an 80-20 split.

## Model Training (Last_tensor.py)

This script trains a Convolutional Neural Network (CNN) to classify tomato ripeness. The model uses the following techniques:

- Early Stopping: Prevents overfitting by stopping training if the validation loss does not improve after 5 epochs.
- Learning Rate Reduction: Reduces the learning rate when the validation loss plateaus.
- Model Checkpointing: Saves the best-performing model during training based on validation accuracy.

### Steps to Train the Model:
1. Place your dataset in the `dataset` directory.
2. Run the `Last_tensor.py` script:
   ```bash
   python Last_tensor.py
   ```

The model will be saved as `best_tomato_model.keras` after training.

## Object Detection and Ripeness Classification (Last_demo1.py)

This script uses a pre-trained YOLOv5 model to detect tomatoes in an image and then uses the trained ripeness classification model to classify each detected tomato as Ripe or Unripe.

### Steps to Detect and Classify Tomatoes:
1. Ensure that the trained model `best_tomato_model.keras` is in the same directory as the script.
2. Run the `Last_demo1.py` script, specifying the image to be analyzed:
   ```bash
   python Last_demo1.py
   ```
3. The script will display the image with bounding boxes around detected tomatoes, and it will classify each tomato as either Ripe or Unripe with a confidence score.

## Example Output

- Bounding Box: A green box is drawn around each detected tomato.
- Ripeness Label: The ripeness status (Ripe/Unripe) is displayed above the bounding box with a confidence percentage.

## Notes

- YOLOv5 Model: The script uses the `yolov5m` model by default for better accuracy. You can choose between `yolov5m`, `yolov5l`, or `yolov5x` depending on your hardware capabilities and accuracy needs.
- Image Resize: The input image is resized to 800 pixels wide while maintaining the aspect ratio.

## Acknowledgments

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [TensorFlow](https://www.tensorflow.org/)

---
