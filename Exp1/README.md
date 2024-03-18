# README.md

## Hyperparameter Tuning in Feedforward Neural Networks for Accurate Sine Function Approximation

### Experiment 1
 
**Course:** Deep Learning  
**Instructor:** Dr. Lian Defu

---

### Abstract

This repository contains the code and report for Experiment 1 titled "Hyperparameter Tuning in Feedforward Neural Networks for Accurate Sine Function Approximation." The primary objective of this experiment is to explore the capabilities of feedforward neural networks (FNN) in accurately approximating the sine function within the interval [0, 2π]. The focus is on investigating the intricate relationship between the network's depth and learning rate while keeping width and activation functions constant.

### 1. Introduction

Neural networks are powerful tools for approximating complex functions, making predictions, and solving a wide range of problems [1]. This experiment explores the capabilities of a feedforward neural network (FNN) in approximating the function y = sin(x), where x is confined to the interval [0, 2π).

The motivation for this experiment lies in the fundamental ability of neural networks to capture complex patterns and learn intricate relationships between input and output data [1]. The sine function serves as a suitable platform for evaluating a neural network's regression capabilities. The central hypothesis posits that meticulous selection and tuning of hyperparameters can significantly enhance the neural network's performance in approximating the sine function.

### 2. Methodology

The codebase for Experiment 1 has been provided by the course instructor. The primary focus is on scrutinizing and adjusting specific hyperparameters, namely network depth and learning rate, as specified in the assignment instructions. The code comprises the following key components:

- **Data Preparation:** Random x values within the range [0, 2π) are sampled, and corresponding y values are calculated using the sine function. Three distinct datasets (training, validation, and testing) are generated.

- **Model Examination:** The existing neural network model architecture is thoroughly examined. The model includes input and output layers, hidden layers, and predefined activation functions.

- **Training Process:** The training phase involves optimizing the network's parameters through gradient descent. The loss function is computed, and parameters are updated during training.

- **Hyperparameter Adjustment:** The primary focus is on adjusting network depth and learning rate while keeping width and activation functions constant.

- **Performance Testing:** The trained model's performance is evaluated on a separate testing dataset using the Mean Square Error (MSE) as the primary metric.

### 3. Experiments

All experiments were conducted on Google Colab due to its suitable environment and library support. The dataset for experiments 1 to 15 is available in the Excel file (DataSet_Of_Experiments.xlsx). Key observations and discussions from the experiments include:

- **Depth Experiment:** Experiment 1 with a depth of 5 and lr = 0.1 achieved remarkable results, indicating that increasing depth does not necessarily lead to better outcomes.

- **Learning Rate Experiment:** Experiment 3, with a depth of 5, lr = 0.001, produced the lowest test set MSE, suggesting that smaller learning rates result in more precise approximations.

- **Width and Activation Functions:** Width and activation functions (width = 17, ReLU activation) remained constant, allowing the isolation of depth and learning rate effects.

### 4. Conclusion

This extensive series of experiments provides valuable insights into hyperparameter tuning in feedforward neural networks for sine function approximation. Key conclusions include:

- A smaller learning rate, such as lr = 0.001, significantly improves sine function approximation precision but comes at the cost of increased training time.
- A moderate learning rate like lr = 0.01 also performs well while being computationally efficient.
- The relationship between depth, learning rate, and model performance is complex, indicating a trade-off between training time and accuracy.

### 5. Acknowledgement

The author acknowledges the invaluable assistance of senior colleagues who guided him during this project.

### 6. References

[1] H. Taherdoost, "Deep learning and neural networks: Decision-making implications," Symmetry, vol. 15, no. 9, p. 1723, 2023. doi:10.3390/sym15091723.

---

### Code for Experiment 1

The code for Experiment 1 is available in the `main.py` script. It includes the necessary libraries, the definition of the neural network class, a training function, and the execution of the experiment with specific hyperparameters.

#### Requirements

Before running the code, ensure you at least have the following libraries installed:

```bash
pip install numpy==1.23.5
pip install matplotlib==3.7.1
pip install torch==2.1.0
```

#### Execution

To run the code, execute the script in your preferred Python environment. The script provides detailed comments for each section, making it easy to understand and reproduce the experiment.

Feel free to explore and modify the code for further experimentation or adaptation to different tasks. If you encounter any issues or have questions, please refer to the above provided brief report or reach out for assistance.

---

**Note:** Please try your hands with the code and brief explanation above. Also ensure to use the reference below to reference this repository.

---

### How to Reference this Repository

If you find this work useful or utilize the provided code for your research or projects, I kindly request that you reference this repository in your work. You can use the following entry:

```latex
[x] Emmanuel Ugwu, "Hyperparameter Tuning in Feedforward Neural Networks for Accurate Sine Function Approximation", 2023. GitHub Repository. [Online]. Available: [GitHub](https://github.com/UEmmanuel5/deep-learning-ustc-2023/tree/master/Exp1).

'x' above is the number this reference would be place in the your reference section.
```

Thank you for acknowledging and referencing this work!

--- 
