# README.md

## Hyperparameter Tuning in Convolutional Neural Networks for Image Classification

### Experiment 2

**Course:** Deep Learning  
**Instructor:** Dr. Lian Defu


---

### Abstract

This repository contains the code and report for Experiment 2 titled "Hyperparameter Tuning in Convolutional Neural Networks for Image Classification." The primary objective of this experiment is to optimize hyperparameters in convolutional neural networks (CNNs) for accurate image classification on the Tiny ImageNet dataset. The focus is on exploring the impact of network architecture, learning rate, and other key parameters on the model's performance.

### 1. Introduction

Convolutional Neural Networks (CNNs) have demonstrated outstanding performance in image classification tasks, making them widely used in computer vision applications [1]. This experiment aims to investigate the influence of hyperparameters on the classification accuracy of CNNs using the Tiny ImageNet dataset. Specifically, the experiment explores the effects of network depth, learning rate, and other relevant parameters on the model's ability to accurately classify images into 200 different classes.

### 2. Methodology

The codebase for Experiment 2 is organized into three main components: `main.py`, `model.py`, and `train_model.py`.

- **main.py:** This script serves as the entry point for the experiment. It initializes the CNN model, trains the model with specified hyperparameters, and evaluates the model on the test set.

- **model.py:** This file contains the definition of the CNN architecture, including residual blocks, convolutional layers, and the overall network structure.

- **train_model.py:** This script handles the training process, including data preparation, hyperparameter tuning, and performance evaluation.

### 3. Experiments

The experiments in this study involve training the CNN model with variations in hyperparameters. Key hyperparameters include the network depth, learning rate, and other architecture-related choices. The primary metrics used for evaluation are training loss, training accuracy, validation loss, validation accuracy, and test accuracy.

### 4. Conclusion

The experiments aim to draw insights into the optimal hyperparameter settings for CNNs on image classification tasks. Key conclusions include:

- The impact of varying network depth on classification accuracy.
- The influence of learning rate on training convergence and overall accuracy.
- The role of residual blocks and convolutional layers in capturing complex features.

### 5. Acknowledgement

I want to acknowledge the invaluable assistance of my senior colleagues, friends, and the TA of this course who guided me to understand the experiment phase of this project. Your support was instrumental in enabling me to finish this report.

### 6. References

[1] M. Krichen, “Convolutional Neural Networks: A survey,” Computers, vol. 12, no. 8, p. 151, Jul. 
2023. doi:10.3390/computers12080151.
[2] L. Alzubaidi et al., “Review of Deep Learning: Concepts, CNN Architectures, challenges, 
applications, Future Directions,” Journal of Big Data, vol. 8, no. 1, Mar. 2021. doi:10.1186/s40537-
021-00444-8. 

---

### Code for Experiment 2

The code for Experiment 2 is organized into three main scripts: `main.py`, `model.py`, and `train_model.py`.

#### Requirements

Before running the code, ensure you have the necessary libraries installed:

```bash
    Python 3
    PyTorch
    torchvision
    NumPy
    OpenCV
    tqdm
    matplotlib
```

#### Execution

To run the experiment, follow these steps:

1. Update the `data_path` variable in `model.py` to point to the correct directory on your machine.

2. Execute the `main.py` script in your preferred Python environment.

3. Adjust hyperparameters in `main.py` as needed.

### How to Reference this Repository

If you find this work useful or utilize the provided code for your research or projects, kindly reference this repository in your work. You can use the following entry:

```latex

[x] Emmanuel Ugwu, "Hyperparameter Tuning in Convolutional Neural Networks for Image Classification," 2023. GitHub Repository. [Online]. Available: [GitHub](https://github.com/UEmmanuel5/deep-learning-ustc-2023/tree/master/Exp2).

'x' above is the number this reference would be placed in your reference section.
```

Thank you for acknowledging and referencing this work!