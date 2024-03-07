# %% [markdown]
# # **Essay Analysis and Classification**

# %% [markdown]
# # **Overview:**
# This project involves a comprehensive analysis and classification of a dataset containing essays. The goal is to gain insights into the content patterns, structure, and characteristics of the essays. Two distinct parts of the code contribute to achieving this objective:

# %% [markdown]
# # **Part 1: Exploratory Data Analysis and Clustering (code1/script1)**
# In this section, we perform an exploratory data analysis (EDA) and clustering analysis on the essays dataset. The main steps include:

# %% [markdown]
# ## Purpose:
# This script1 is designed to perform an exploratory data analysis (EDA) and clustering analysis on a dataset of essays. The main goals are to analyze the text data, identify patterns, and perform K-Means clustering to group essays based on their content.
# 

# %% [markdown]
# ## Libraries Used:
# The following Python libraries are imported for analysis:
# - `numpy` and `pandas` for data manipulation
# - `matplotlib` and `seaborn` for data visualization
# - `os` for interacting with the operating system
# - Various modules from `sklearn` for machine learning tasks
# - `nltk` for natural language processing
# - `re` for regular expressions
# - `IPython.display` for displaying Markdown in Jupyter Notebooks

# %%
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from collections import Counter
from IPython.display import Markdown, display


# %% [markdown]
# **Fixed random seed**

# %%
# Set a fixed random seed for reproducibility
seed_value = 10
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

# %% [markdown]
# # 1. **Define Functions:**
# * '**calculate_text_metrics_simple**': Calculates basic text metrics such as word count, unique word count, sentence count, and average word length.
# * '**plot_most_common_words**': Plots the most common words in a given text series.
# * Other utility functions for thematic coherence, counting digit styles, and text preprocessing.

# %%
# Define functions for analysis
def calculate_text_metrics_simple(text):
    words = text.split()
    sentences = text.split('.')
    word_count = len(words)
    unique_word_count = len(set(words))
    sentence_count = len(sentences)
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    return word_count, unique_word_count, sentence_count, avg_word_length

def plot_most_common_words(text_series, num_words=30, title="Most Common Words"):
    all_text = ' '.join(text_series).lower()
    words = all_text.split()
    word_freq = Counter(words)
    common_words = word_freq.most_common(num_words)
    
    # Plot the most common words
    plt.figure(figsize=(15, 6))
    sns.barplot(x=[word for word, freq in common_words], y=[freq for word, freq in common_words])
    plt.title(title)
    plt.xticks(rotation=45)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.show()


# %%
def calculate_thematic_coherence(essays, prompts):
    vectorizer = TfidfVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform(essays + prompts)

    lsa = TruncatedSVD(n_components=1)
    dtm_lsa = lsa.fit_transform(dtm)
    dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)

    essay_vectors = dtm_lsa[:len(essays)]
    prompt_vectors = dtm_lsa[len(essays):]

    coherence_scores = cosine_similarity(essay_vectors, prompt_vectors)
    return coherence_scores.mean()

def count_digit_styles(text, numeral_regex, word_number_regex):
    numeral_matches = len(re.findall(numeral_regex, text))
    word_number_matches = len(re.findall(word_number_regex, text))
    return numeral_matches, word_number_matches

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)


# %%
def plot_pca_for_prompt(prompt_id, df, tfidf_vectorizer):
    prompt_df = df[df['prompt_id'] == prompt_id]
    tfidf_matrix = tfidf_vectorizer.transform(prompt_df['preprocessed_text'])
    
    pca = PCA(n_components=2, random_state=42)
    reduced_data = pca.fit_transform(tfidf_matrix.toarray())
    
    pca_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
    pca_df['generated'] = prompt_df['generated'].values
    
    plt.figure(figsize=(14, 7))
    scatter_plot = sns.scatterplot(
        x='PC1', y='PC2', hue='generated', data=pca_df, 
        palette=custom_palette, alpha=0.7
    )
    plt.title(f'PCA of Essays for Prompt {prompt_id}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    handles, labels = scatter_plot.get_legend_handles_labels()
    plt.legend(handles, ['Student', 'LLM Generated'], title='Essay Type')
    
    plt.show()


# %% [markdown]
# # 2. **Load Data:**
#    - Load training essays, test essays, and training prompts from CSV files.

# %%
# Load data
train_essays_path = 'Dataset/train_essays.csv'
test_essays_path = 'Dataset/test_essays.csv'
train_prompts_path = 'Dataset/train_prompts.csv'

train_essays_df = pd.read_csv(train_essays_path)
test_essays_df = pd.read_csv(test_essays_path)
train_prompts_df = pd.read_csv(train_prompts_path)


# %%
# Display a few rows of the training essays
display(Markdown("### Training Essays"))
display(train_essays_df.head())

# Display a few rows of the test essays
display(Markdown("### Test Essays"))
display(test_essays_df.head())

# Display a few rows of the training prompts
display(Markdown("### Training Prompts"))
display(train_prompts_df.head())


# %% [markdown]
# # 3. **EDA:**
#    - Check for missing values in datasets and analyze the distribution of the 'generated' column in training essays.
#    - Analyze essay length distribution and compare lengths between student-written and LLM-generated essays.
#    - Explore common words in student-written and LLM-generated essays.

# %%
# Exploratory Data Analysis (EDA)

# Check for missing values in the datasets
missing_values_train = train_essays_df.isnull().sum()
print("Missing Values in Training Essays:")
print(missing_values_train)

missing_values_test = test_essays_df.isnull().sum()
print("\nMissing Values in Test Essays:")
print(missing_values_test)

missing_values_prompts = train_prompts_df.isnull().sum()
print("\nMissing Values in Training Prompts:")
print(missing_values_prompts)

# Understand the balance of categories
print("\nDistribution of 'generated' column in training essays:")
print(train_essays_df['generated'].value_counts())


# %%
# EDA - Text Length Analysis

# Calculate the length of each essay and compare the distributions
train_essays_df['essay_length'] = train_essays_df['text'].apply(len)

sns.set(style="whitegrid")
plt.figure(figsize=(14, 7))

# Distribution of essay lengths for student essays
sns.histplot(train_essays_df[train_essays_df['generated'] == 0]['essay_length'], color="skyblue", label='Student Essays', kde=True)

# Distribution of essay lengths for LLM generated essays
sns.histplot(train_essays_df[train_essays_df['generated'] == 1]['essay_length'], color="red", label='LLM Generated Essays', kde=True)

plt.title('Distribution of Essay Lengths')
plt.xlabel('Essay Length (Number of Characters)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

plt.figure(figsize=(12, 8))
sns.boxplot(x='generated', y='essay_length', data=train_essays_df)
plt.title('Comparison of Essay Lengths by Source')
plt.xlabel('Essay Source')
plt.ylabel('Essay Length')
plt.xticks([0, 1], ['Student-written', 'LLM-generated'])
plt.show()


# %%
# EDA - Common words

plot_most_common_words(train_essays_df[train_essays_df['generated'] == 0]['text'], title="Most Common Words in Student Essays")
plot_most_common_words(train_essays_df[train_essays_df['generated'] == 1]['text'], title="Most Common Words in LLM Generated Essays")


# %% [markdown]
# # 4. **Text Preprocessing:**
#    - Preprocess text data by removing special characters, numbers, stopwords, and stemming.
# 

# %%
# Preprocess text data
train_essays_df['preprocessed_text'] = train_essays_df['text'].apply(preprocess_text)

# Calculate text metrics
train_essays_df['word_count'], train_essays_df['unique_word_count'], train_essays_df['sentence_count'], train_essays_df['avg_word_length'] = zip(*train_essays_df['preprocessed_text'].apply(calculate_text_metrics_simple))

# Display the calculated text metrics
display(Markdown("### Calculated Text Metrics"))
display(train_essays_df[['word_count', 'unique_word_count', 'sentence_count', 'avg_word_length']].head())


# %% [markdown]
# # 5.  **Text Metric Analysis:**
#    - Analyze boxplots for word count, unique word count, sentence count, and average word length.
# 

# %%
# EDA - Text Metric Analysis

plt.figure(figsize=(15, 10))

# Boxplot for word count
plt.subplot(2, 2, 1)
sns.boxplot(x='generated', y='word_count', data=train_essays_df)
plt.title('Comparison of Word Count by Source')
plt.xlabel('Essay Source')
plt.ylabel('Word Count')
plt.xticks([0, 1], ['Student-written', 'LLM-generated'])

# Boxplot for unique word count
plt.subplot(2, 2, 2)
sns.boxplot(x='generated', y='unique_word_count', data=train_essays_df)
plt.title('Comparison of Unique Word Count by Source')
plt.xlabel('Essay Source')
plt.ylabel('Unique Word Count')
plt.xticks([0, 1], ['Student-written', 'LLM-generated'])

# Boxplot for sentence count
plt.subplot(2, 2, 3)
sns.boxplot(x='generated', y='sentence_count', data=train_essays_df)
plt.title('Comparison of Sentence Count by Source')
plt.xlabel('Essay Source')
plt.ylabel('Sentence Count')
plt.xticks([0, 1], ['Student-written', 'LLM-generated'])

# Boxplot for average word length
plt.subplot(2, 2, 4)
sns.boxplot(x='generated', y='avg_word_length', data=train_essays_df)
plt.title('Comparison of Average Word Length by Source')
plt.xlabel('Essay Source')
plt.ylabel('Average Word Length')
plt.xticks([0, 1], ['Student-written', 'LLM-generated'])

plt.tight_layout()
plt.show()


# %% [markdown]
# # 6. **Clustering Analysis:**
#    - Use TF-IDF vectorization to convert text data into numerical features.
#    - Determine the optimal number of clusters using silhouette scores.
#    - Apply K-Means clustering and visualize results using PCA.

# %%
# Clustering Analysis

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(train_essays_df['preprocessed_text'])

# Determine the optimal number of clusters using the silhouette score
silhouette_scores = []
possible_k_values = range(2, 11)

for k in possible_k_values:
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    silhouette_avg = silhouette_score(tfidf_matrix, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(possible_k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score for Different Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()


# %%
# Import necessary libraries
from sklearn.decomposition import PCA

# Function to visualize clusters using PCA
def visualize_clusters_pca(tfidf_matrix, cluster_labels, num_clusters, title="PCA Visualization"):
    pca = PCA(n_components=2, random_state=42)
    reduced_data = pca.fit_transform(tfidf_matrix.toarray())
    
    pca_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
    pca_df['cluster'] = cluster_labels
    
    plt.figure(figsize=(14, 7))
    scatter_plot = sns.scatterplot(
        x='PC1', y='PC2', hue='cluster', data=pca_df, 
        palette=sns.color_palette("husl", num_clusters), alpha=0.7
    )
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    handles, labels = scatter_plot.get_legend_handles_labels()
    plt.legend(handles, [f'Cluster {i}' for i in range(num_clusters)], title='Cluster')
    
    plt.show()


# %%
# Choose the optimal number of clusters based on the plot (e.g., k=4)
optimal_num_clusters = 4

# Apply K-Means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_num_clusters, n_init='auto', random_state=42)
train_essays_df['cluster'] = kmeans.fit_predict(tfidf_matrix)

# Visualize clustering results using PCA
visualize_clusters_pca(tfidf_matrix, train_essays_df['cluster'], optimal_num_clusters)

# %% [markdown]
# # 7. **Prompt-Specific Analysis:**
#    - Perform prompt-specific analysis, applying clustering and visualizing results for each prompt.

# %%
# Prompt-Specific Analysis

custom_palette = sns.color_palette("husl", optimal_num_clusters)
for prompt_id in train_prompts_df['prompt_id']:
    prompt_specific_df = train_essays_df[train_essays_df['prompt_id'] == prompt_id].copy()  # Make a copy to avoid SettingWithCopyWarning
    tfidf_matrix_prompt = tfidf_vectorizer.transform(prompt_specific_df['preprocessed_text'])
    
    # Apply K-Means clustering for each prompt
    kmeans_prompt = KMeans(n_clusters=optimal_num_clusters, n_init='auto', random_state=42)
    train_essays_df.loc[train_essays_df['prompt_id'] == prompt_id, 'cluster'] = kmeans_prompt.fit_predict(tfidf_matrix_prompt)
    
    # Visualize clustering results for each prompt using PCA
    plot_pca_for_prompt(prompt_id, prompt_specific_df, tfidf_vectorizer)


# %% [markdown]
# # 8. **Generate Submission File:**
#    - Generate a random submission file for testing purposes.

# %%
# # Generate Submission File
# submission_df = pd.DataFrame({
#     'essay_id': test_essays_df.index,  # Assuming the index is the essay_id
#     'generated': np.random.randint(2, size=len(test_essays_df))
# })

# # Save the submission file
# submission_df.to_csv('/kaggle/working/submission.csv', index=False)


# %% [markdown]
# # **Part 2: BERT for Essay Classification (code2)**
# The second part of the project focuses on using the BERT (Bidirectional Encoder Representations from Transformers) model for essay classification into two categories: student-written and AI-generated. The key steps involved are:
# 

# %% [markdown]
# ## Purpose:
# This script utilizes the BERT (Bidirectional Encoder Representations from Transformers) model for the classification of essays into two categories: student-written and AI-generated. The main objectives include preprocessing text data, tokenizing and encoding for BERT, training the model, and generating predictions for test data.

# %% [markdown]
# ## Libraries Used:
# - `numpy` and `pandas` for data manipulation
# - `matplotlib` and `seaborn` for data visualization
# - `nltk` for natural language processing
# - `sklearn` for machine learning tasks
# - `transformers` library for BERT model implementation
# - `torch` for PyTorch-based deep learning

# %%
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split


# %% [markdown]
# # 1. **Text Preprocessing:**
#    - Download stopwords from NLTK and clean the text data by removing punctuations, tokenizing, converting to lowercase, and removing stop words.
# 

# %%
import nltk
nltk.download('stopwords')

# %%
# Text Preprocessing
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuations
    words = text.split()  # Tokenize
    words = [word.lower() for word in words if word.isalpha()]  # Lowercase and remove non-alphabetic words
    words = [word for word in words if word not in stop_words]  # Remove stop words
    return ' '.join(words)

train_essays_df['clean_text'] = train_essays_df['text'].apply(clean_text)

# Apply text preprocessing
train_essays_df['preprocessed_text'] = train_essays_df['text'].apply(preprocess_text)


# %%
# Apply text preprocessing
train_essays_df['preprocessed_text'] = train_essays_df['text'].apply(preprocess_text)


# %% [markdown]
# # 2. **Split Data:**
#    - Split the preprocessed text data into training and validation sets using `train_test_split`.
# 

# %%
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_essays_df['preprocessed_text'], train_essays_df['generated'], test_size=0.2, random_state=42)


# %% [markdown]
# # 3. **Tokenization and Encoding for BERT:**
#    - Use the BERT tokenizer to tokenize and encode the text data, converting it into input tensors suitable for the BERT model.
# 

# %%
# Tokenization and Encoding for BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, padding=True, truncation=True, max_length=128)
encoded_train = tokenizer(X_train.tolist(), padding=True, truncation=True, return_tensors='pt')
encoded_val = tokenizer(X_val.tolist(), padding=True, truncation=True, return_tensors='pt')


# %% [markdown]
# # 4. **Create TensorDatasets:**
#    - Convert labels to tensors and create `TensorDataset` objects for training and validation sets.
# 

# %%
# Convert labels to tensors
train_labels = torch.tensor(y_train.values)
val_labels = torch.tensor(y_val.values)


# %%
# Create TensorDatasets
train_dataset = TensorDataset(encoded_train['input_ids'], encoded_train['attention_mask'], train_labels)
val_dataset = TensorDataset(encoded_val['input_ids'], encoded_val['attention_mask'], val_labels)


# %% [markdown]
# # 5. **DataLoader for Efficient Processing:**
#    - Create `DataLoader` objects to efficiently process batches of data during training and validation.
# 

# %%
# DataLoader for efficient processing
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


# %% [markdown]
# # 6. **Define BERT Model:**
#    - Use the BERT model for sequence classification from the `transformers` library, specifying the number of labels as 2 (for binary classification).
# 

# %%
# Define the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# %% [markdown]
# # 7. **Define Optimizer and Learning Rate Scheduler:**
#    - Define the AdamW optimizer and set up learning rate scheduling.
# 

# %%
# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
epochs = 10


# %% [markdown]
# # 8. **Training Loop:**
#    - Train the BERT model on the training data using a training loop. Utilize gradient clipping to avoid exploding gradients.
# 

# %%
# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping to avoid exploding gradients
        optimizer.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss:.2f}")


# %% [markdown]
# # 9. **Validation Loop:**
#    - Evaluate the trained model on the validation set and calculate validation accuracy.
# 

# %%
# Validation loop
model.eval()
val_preds = []
val_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        val_labels.extend(labels.cpu().numpy())


# %%
# Calculate validation accuracy
val_accuracy = accuracy_score(val_labels, val_preds)
print(f"Validation Accuracy: {val_accuracy:.2f}")


# %% [markdown]
# # 10. **Test Data Processing:**
#     - Tokenize and encode the test data for predictions using the trained model.
# 

# %%
# Test data processing
test_inputs = tokenizer(test_essays_df['text'].tolist(), padding=True, truncation=True, return_tensors='pt')

# Move input tensor to the same device as the model
test_inputs = {key: value.to(device) for key, value in test_inputs.items()}


# %% [markdown]
# # 11. **Generate Predictions:**
#     - Generate predictions using the trained model and softmax activation.
# 

# %%
# Generate predictions using your trained model
with torch.no_grad():
    outputs = model(**test_inputs)
    logits = outputs.logits


# %%
# Assuming the first column of logits corresponds to the negative class (non-AI-generated)
# and the second column corresponds to the positive class (AI-generated)
predictions = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Move predictions back to CPU


# %% [markdown]
# # 12. **Create Submission File:**
#     - Create a submission DataFrame with essay IDs and corresponding predictions and save it as a CSV file.
# 

# %%
# Create a submission DataFrame with essay IDs and corresponding predictions
submission_df = pd.DataFrame({
    'essay_id': test_essays_df.index,  # Assuming the index is the essay_id
    'generated': predictions
})
# Save the submission file
submission_df.to_csv('Results/submission.csv', index=False)


# %% [markdown]
# ## Usage
# Ensure the required libraries are installed, provide the necessary paths for data files, and execute the script to perform the described analyses. Adjust parameters and functions as needed for specific requirements.

# %%
# !pip freeze > requirements.txt


