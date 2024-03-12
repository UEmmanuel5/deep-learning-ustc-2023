# README.md

## Sentiment Analysis with Attention Mechanism

### Experiment 3

**Course:** Deep Learning  
**Instructor:** Dr. Lian Defu

---

### Abstract

This repository contains the code and report for Experiment 3, focusing on sentiment analysis using a deep learning model with an attention mechanism. The experiment utilizes the Yelp reviews dataset to train a model that can accurately classify reviews as positive or negative. The primary objectives include implementing attention mechanisms in recurrent neural networks (RNNs) for sentiment analysis and exploring the impact of hyperparameters on model performance.

### 1. Introduction

Sentiment analysis plays a crucial role in understanding user opinions and feedback. This experiment leverages deep learning techniques, specifically recurrent neural networks with attention mechanisms, to perform sentiment analysis on Yelp reviews. By examining the influence of hyperparameters such as network architecture, learning rate, and dropout rate, the goal is to achieve improved sentiment classification accuracy.

### 2. Methodology

The codebase for Experiment 3 is organized into a Jupyter Notebook (`MainRNN.ipynb`) and leverages various Python scripts. Key components include data preprocessing, model architecture definition, attention mechanism implementation, and model training and evaluation.

### 3. Experiments

The experiments involve training the sentiment analysis model with variations in hyperparameters. Key hyperparameters include the attention mechanism's parameters, LSTM cell size, and dropout rate. Evaluation metrics include accuracy, precision, recall, and the area under the ROC curve (AUC).

### 4. Conclusion

Insights gained from the experiments include:

- The impact of attention mechanisms on capturing contextual information for sentiment analysis.
- The influence of LSTM cell size on the model's ability to learn complex patterns in text data.
- The significance of proper tuning for hyperparameters like dropout rate in improving model generalization.

### 5. Acknowledgement

I express my gratitude to the TA of this course for providing guidance and insights throughout this experiment. Special thanks to my peers who contributed to discussions and problem-solving during the experiment phase.

### 6. References

[1] B. P. R. Guda, M. Srivastava, and D. Karkhanis, “Sentiment analysis: Predicting Yelp scores,” arXiv.org, [https://arxiv.org/abs/2201.07999](https://arxiv.org/abs/2201.07999) (accessed Dec. 15, 2023).  
[2] J. E. White, “Hybrid church,” Google Books, [https://books.google.de/books?hl=en&amp;lr=&amp;id=c7l3EAAAQBAJ&amp;oi=fnd&amp;pg=PR7&amp;dq=In%2Bthe%2Bcontemporary%2C%2Bvisuallydriven%2Blandscape%2B%2B%2Bthe%2Bquest%2Bfor%2Bimmersive%2Bexperiences%2B%2B%2Bhas%2Bbecome%2Ba%2Bubiquitous%2Bendeavor%2B&amp;ots=V0ul5IJkiX&amp;sig=aai0hAIJWKHmidZrudVyNNTT-tg&amp;redir_esc=y#v=onepage&amp;q&amp;f=false (accessed Nov. 2, 2023).  
[3] J. Kim, “Sentiment analysis in Keras using attention mechanism on Yelp reviews dataset,” Medium, [https://medium.com/@jeewonkim1028/sentiment-analysis-in-kerasusing-attention-mechanism-on-yelp-reviews-dataset-322bd7333b8b](https://medium.com/@jeewonkim1028/sentiment-analysis-in-keras-using-attention-mechanism-on-yelp-reviews-dataset-322bd7333b8b) (accessed Dec. 15, 2023).

---

### Code for Experiment 3

The code for Experiment 3 is available in the Jupyter Notebook `MainRNN.ipynb`.

#### Requirements

Before running the notebook, ensure you have the necessary libraries installed:

```bash
Python 3
pandas
numpy
nltk
scikit-learn
tensorflow
matplotlib
seaborn
```

#### Execution

1. Open the `MainRNN.ipynb` notebook in a Jupyter environment.

2. Execute the cells sequentially to run data preprocessing, model training, and evaluation.

3. Adjust hyperparameters as needed for further exploration.

### How to Reference this Repository

If you find this work useful or utilize the provided code for your research or projects, kindly reference this repository in your work. You can use the following entry:

```latex
[x] Emmanuel Ugwu, "Sentiment Analysis with Attention Mechanism, 2024. GitHub Repository," 2023. GitHub Repository. [Online]. Available: [GitHub](https://github.com/UEmmanuel5/deep-learning-ustc-2023/tree/master/Exp3).

'x' above is the number this reference would be place in the your reference section.

```

Thank you for acknowledging and referencing this work!