import joblib
import yaml
from tensorflow.keras.models import save_model, load_model
  # Import your Django model if needed
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Data Collection
photosynthesis_answers = [
    "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with carbon dioxide and water.",
    "Photosynthesis is the process by which plants convert light energy into chemical energy.",
    "Photosynthesis is the biological process by which plants and some other organisms convert light energy into chemical energy.",
    "Photosynthesis is a vital process for the survival of plants, where they convert light energy into glucose and oxygen.",
    "Photosynthesis is how plants make food using sunlight, carbon dioxide, and water.",
    "Photosynthesis is a biological process where plants use sunlight to produce oxygen and glucose."
]
photosynthesis_scores = [5, 0, 4, 3, 2, 1]  # Corresponding scores (1: correct, 0: incorrect)

global_warming_answers = [
    "Global warming is the long-term heating of Earth’s climate system observed since the pre-industrial period (between 1850 and 1900) due to human activities, primarily fossil fuel burning, which increases greenhouse gas levels in the atmosphere.",
    "Global warming is a gradual increase in the overall temperature of the earth's atmosphere generally attributed to the greenhouse effect caused by increased levels of carbon dioxide, chlorofluorocarbons, and other pollutants.",
    "Global warming is the long-term increase in Earth's average surface temperature due to human activities, primarily the release of greenhouse gases from burning fossil fuels.",
    "Global warming is the rise in Earth's average surface temperature due to greenhouse gases like carbon dioxide and methane being released into the atmosphere.",
    "Global warming is the gradual increase in the temperature of Earth's atmosphere due to the burning of fossil fuels and other human activities.",
    "Global warming is the increase in Earth's average temperature due to the greenhouse effect, caused by increased levels of carbon dioxide, chlorofluorocarbons, and other pollutants.",
]
global_warming_scores = [5, 4, 3, 2, 1, 0]  # Corresponding scores (1: correct, 0: incorrect)

# Data Preprocessing
tokenizer = Tokenizer()
max_length = 128

def preprocess_data(answers, scores):
    tokenizer.fit_on_texts(answers)
    sequences = tokenizer.texts_to_sequences(answers)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    X = np.array(padded_sequences)
    y = np.array(scores)
    return X, y

photosynthesis_X, photosynthesis_y = preprocess_data(photosynthesis_answers, photosynthesis_scores)
global_warming_X, global_warming_y = preprocess_data(global_warming_answers, global_warming_scores)

# Model Architecture
def create_model(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Training
photosynthesis_model = create_model(len(tokenizer.word_index) + 1)
photosynthesis_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
photosynthesis_model.fit(photosynthesis_X, photosynthesis_y, epochs=10)

global_warming_model = create_model(len(tokenizer.word_index) + 1)
global_warming_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
global_warming_model.fit(global_warming_X, global_warming_y, epochs=10)
photosynthesis_model.save("photosynthesis_model.h5")  # Using TensorFlow's model saving
global_warming_model.save("global_warming_model.h5")


photosynthesis_model = load_model("photosynthesis_model.h5")


model_config = photosynthesis_model.get_config()


with open("photosynthesis_model_config.yaml", "w") as yaml_file:
    yaml.dump(model_config, yaml_file)


# with open("photosynthesis_model_config.yaml", "r") as yaml_file:
#     loaded_model_config = yaml.safe_load(yaml_file)


# loaded_model = Sequential.from_config(loaded_model_config)