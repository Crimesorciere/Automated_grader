from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from typing import List
import os
from tensorflow.keras.models import Sequential
from fastapi import FastAPI, Query
import yaml
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
# file = os.path.abspath("photosynthesis_model.h5")
# print (file)
with open("photosynthesis_model_config.yaml", "r") as yaml_file:
    loaded_model_config = yaml.load(yaml_file,Loader=yaml.FullLoader)

# Create model from config
photosynthesis_model = Sequential.from_config(loaded_model_config)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET","POST"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="."), name="static")


max_length = 128

class AnswerRequest(BaseModel):
    answer: str

class PredictionResponse(BaseModel):
    score: float


@app.get("/", response_class=HTMLResponse)
async def get_index():
    # Return the HTML file
    return open("index.html").read()


@app.get('/predict_photosynthesis/')
async def predict_photosynthesis(answer: str = Query(..., title="Answer")):
    # Load the model inside the route function
    # photosynthesis_model = load_model("D:\\Aritifcial Intelligence\\practice_projects\\app\\app\\photosynthesis_model.h5")
    tokenizer = tf.keras.preprocessing.text.Tokenizer()

    sequence = tokenizer.texts_to_sequences([answer])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = photosynthesis_model.predict(padded_sequence)
    print(float(prediction[0][0]))
    # Return the prediction as part of the response
    return {"answer": answer, "score": float(prediction[0][0])}


# @app.post("/predict_global_warming/")
# async def predict_global_warming(request: AnswerRequest):
#     answer = request.answer
    
#     # Load the model inside the route function
#     # global_warming_model = load_model("D:\\Aritifcial Intelligence\\practice_projects\\app\\app\\global_warming_model.h5")
#     tokenizer = tf.keras.preprocessing.text.Tokenizer()

#     sequence = tokenizer.texts_to_sequences([answer])
#     padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
#     prediction = global_warming_model.predict(padded_sequence)
#     return {"score": float(prediction[0][0])}
