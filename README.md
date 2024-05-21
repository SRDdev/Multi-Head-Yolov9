# Multi-Head YOLOv9 for Detection and Instance Segmentation

![YOLO V9](https://github.com/SRDdev/Multi-Head-Yolov9/assets/84516626/5087545b-33a1-46a1-a050-301737753b85)

## Overview
This repository contains the implementation of a multi-head YOLOv9 model for clothes detection and instance segmentation. The model is trained on the DeepFashion dataset and evaluated using MSCOCO evaluation metrics. It predicts bounding boxes, instance segmentation masks, category labels, and confidence scores simultaneously.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training](#training)
5. [Visualization](#visualization)
6. [Performance Evaluation](#performance-evaluation)
7. [Documentation](#documentation)
8. [Requirements](#requirements)
9. [Usage](#usage)
10. [Credits](#credits)
11. [License](#license)

## Introduction
This project aims to develop a robust solution for clothes detection and instance segmentation using YOLOv9 architecture. By predicting bounding boxes and instance segmentation masks simultaneously, the model enhances the understanding of clothing items in images, facilitating various applications such as fashion e-commerce, visual search, and virtual try-on.

## Dataset
The DeepFashion dataset is utilized for training, validation, and testing. It contains a diverse collection of images with annotations for clothing items. We preprocess the dataset to create a smaller sample with 500 images, ensuring a balanced distribution of classes across train, validation, and test sets.

## Model Architecture
We extend the YOLOv9-c model architecture to accommodate an additional head for instance segmentation. This modification enables the model to predict bounding boxes, instance segmentation masks, category labels, and confidence scores simultaneously, enhancing its capabilities for clothing detection and segmentation tasks.

## Training
The model is trained using transfer learning with pre-trained weights from MS COCO dataset. A modular training script is provided, allowing easy configuration of hyperparameters via a YAML config file. The training process supports both GPU and CPU execution for flexibility.

## Visualization
We provide Jupyter Notebook scripts for visualizing the detection bounding boxes and instance segmentation masks generated by the trained model. These visualizations aid in understanding the model's performance and its ability to accurately identify clothing items in images.

## Performance Evaluation
Performance metrics for both detection and instance segmentation are computed on the validation and test sets using MSCOCO evaluation metrics. We analyze these metrics to assess the model's effectiveness and discuss potential improvements to enhance its performance further.

## Documentation
The codebase is extensively documented using PyLint to ensure readability and maintainability. A detailed document explaining the solution approach, implementation details, and usage instructions is provided in the repository.

## Requirements
- Python 3.x
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- PyYAML

## Usage
1. Clone the repository:
   ```
   git clone https://github.com/SRDdev/Yolov9.git
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Configure hyperparameters in `config.yaml`.
4. Train the model:
   ```
   python trainer.ipynb
   ```
5. Visualize results:
   ```
   jupyter notebook visualize.ipynb
   ```

## Credits
- YOLOv9 implementation: [WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
- DeepFashion dataset: [DeepFashion](https://github.com/switchablenorms/DeepFashion2)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
