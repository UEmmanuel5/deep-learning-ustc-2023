# %%
# Importing necessary libraries
import os
import pathlib
import random
import string
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import data as tf_data
from tensorflow import strings as tf_strings

# Set the seed for reproducibility
seed = 10
random.seed(seed)
np.random.seed(seed)


# Provide the path to the locally downloaded dataset
local_data_path = "news-commentary-v15.en-zh.tsv"

# Parsing the data and filtering out lines without data
with open(local_data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    eng, zh = line.split("\t")
    # Check if both English and Chinese sentences have content
    if eng.strip() and zh.strip():
        zh = "[start] " + zh + " [end]"
        text_pairs.append((eng, zh))

# Limiting the total data to 200,000 pairs
random.shuffle(text_pairs)
text_pairs = text_pairs[:200000]

# Splitting the sentence pairs into training, validation, and test sets
num_val_samples = int(0.1 * len(text_pairs))
num_train_samples = int(0.8 * len(text_pairs))
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"Original number of pairs: {len(lines)}")
print(f"Number of pairs after filtering: {len(text_pairs)}")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")


# %%
# Vectorizing the text data
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

vocab_size = 15000
sequence_length = 20
batch_size = 64

def custom_standardization(input_string):
    lowercase = tf_strings.lower(input_string)
    return tf_strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

eng_vectorization = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
zh_vectorization = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)
train_eng_texts = [pair[0] for pair in train_pairs]
train_zh_texts = [pair[1] for pair in train_pairs]
eng_vectorization.adapt(train_eng_texts)
zh_vectorization.adapt(train_zh_texts)

# Formatting datasets
def format_dataset(eng, zh):
    eng = eng_vectorization(eng)
    zh = zh_vectorization(zh)
    return (
        {
            "encoder_inputs": eng,
            "decoder_inputs": zh[:, :-1],
        },
        zh[:, 1:],
    )

def make_dataset(pairs):
    eng_texts, zh_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    zh_texts = list(zh_texts)
    dataset = tf_data.Dataset.from_tensor_slices((eng_texts, zh_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.cache().shuffle(2048).prefetch(16)

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)
test_ds = make_dataset(test_pairs)

# %%
# Building the Transformer model
embed_dim = 256
latent_dim = 2048
num_heads = 8

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dense_dim, activation="relu"),  #change from relu to tanh
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()    #successively add the below layers
        # self.layernorm_3 = tf.keras.layers.LayerNormalization()  # Additional layer 1
        # self.layernorm_4 = tf.keras.layers.LayerNormalization()  # Additional layer 2
        # self.layernorm_5 = tf.keras.layers.LayerNormalization()  # Additional layer 3
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, None, :], dtype="int32")
        else:
            padding_mask = None

        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
            }
        )
        return config

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        else:
            return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(latent_dim, activation="relu"), #change from relu to tanh
                tf.keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()    #successively add the below layers
        # self.layernorm_4 = tf.keras.layers.LayerNormalization()  # Additional layer 1
        # self.layernorm_5 = tf.keras.layers.LayerNormalization()  # Additional layer 2
        # self.layernorm_6 = tf.keras.layers.LayerNormalization()  # Additional layer 3
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, None, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)
        else:
            padding_mask = None

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, None]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.convert_to_tensor([1, 1])],
            axis=0,
        )
        return tf.tile(mask, mult)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "latent_dim": self.latent_dim,
                "num_heads": self.num_heads,
            }
        )
        return config

# Assemble the end-to-end model
encoder_inputs = tf.keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
encoder = tf.keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = tf.keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = tf.keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
x = tf.keras.layers.Dropout(0.5)(x)
decoder_outputs = tf.keras.layers.Dense(vocab_size, activation="softmax")(x)
decoder = tf.keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = tf.keras.Model(
    [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
)

# Define the testing dataset
test_ds = make_dataset(test_pairs)

# Training the model
epochs = 30  # This should be at least 30 for convergence

transformer.summary()
# Compile the model
transformer.compile(
    optimizer="adam",  # Use Adam optimizer as recommended
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)




print("Training started.")
# Train the model
history = transformer.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
)
print("Training completed.")


# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Testing the model
test_loss, test_accuracy = transformer.evaluate(test_ds)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Save the model
transformer.save("name of model")

# %%
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

# Function to convert integer sequences back to text
def sequence_to_text(sequence, vectorization_layer):
    inv_vocab = {i: word for word, i in enumerate(vectorization_layer.get_vocabulary())}
    return " ".join([inv_vocab.get(i, '[UNK]') for i in sequence if i > 0])



# Function to calculate BLEU scores
def calculate_bleu_scores(model, dataset, vectorization_layer):
    references = []  # Actual translations
    candidates = []  # Model's translations

    for eng_batch, zh_batch_true in dataset:
        # Predict using the model
        zh_batch_pred = model.predict(eng_batch)

        # Convert integer sequences back to text
        zh_batch_true_text = [sequence_to_text(seq, vectorization_layer) for seq in zh_batch_true.numpy()]
        zh_batch_pred_text = [sequence_to_text(seq, vectorization_layer) for seq in zh_batch_pred.argmax(axis=-1)]

        references.extend(zh_batch_true_text)
        candidates.extend(zh_batch_pred_text)

    # Calculate BLEU scores
    bleu_scores = corpus_bleu([[ref.split()] for ref in references], [cand.split() for cand in candidates])

    return bleu_scores

# Calculate BLEU scores for training, validation, and testing sets
train_bleu = calculate_bleu_scores(transformer, train_ds, zh_vectorization)
val_bleu = calculate_bleu_scores(transformer, val_ds, zh_vectorization)
test_bleu = calculate_bleu_scores(transformer, test_ds, zh_vectorization)

print("BLEU Scores:")
print("Training BLEU:", train_bleu)
print("Validation BLEU:", val_bleu)
print("Testing BLEU:", test_bleu)


