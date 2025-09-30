import gradio as gr
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import model_from_json

# === LOAD MODEL & TOOLS ===

# Load model JSON

with open("CNN_model.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("CNN_model_weights.weights.h5")

# 
# with open("CNN_model.json", "r") as json_file:
#     loaded_model_json = json_file.read()

# model = model_from_json(loaded_model_json)
# model.load_weights("CNN_model_weights.weights.h5")

# Load scaler
with open("scaler2.pickle", "rb") as f:
    scaler = pickle.load(f)

# Load label encoder
with open("encoder2.pickle", "rb") as f:
    encoder = pickle.load(f)

# === FEATURE EXTRACTION ===

def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)
def rmse(data,frame_length=2048,hop_length=512):
    # Changed 'data' to 'y=data' as per likely librosa API change
    rmse=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse)
def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    # Also update mfcc call for consistency
    mfcc=librosa.feature.mfcc(y=data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
    result=np.array([])

    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result


# === FEATURE RESHAPE === #

target_length = 2376

def get_predict_feat(path):
    d, s_rate= librosa.load(path, duration=2.5, offset=0.6)
    res=extract_features(d)
    result=np.array(res)
    # print(f"Original feature shape: {result.shape}") # Print original shape for debugging

    # Pad the result array to the target length
    if result.shape[0] < target_length:
        padding = target_length - result.shape[0]
        result = np.pad(result, (0, padding), 'constant')
    elif result.shape[0] > target_length:
        # Truncate if the feature vector is longer than the target length
        result = result[:target_length]

    # print(f"Padded/Truncated feature shape: {result.shape}") # Print new shape

    result=np.reshape(result,newshape=(1,target_length))
    i_result = scaler.transform(result)
    final_result=np.expand_dims(i_result, axis=2)

    return final_result

# === PREDICTION FUNCTION ===

def prediction(path1):
    res=get_predict_feat(path1)
    predictions=model.predict(res)
    y_pred = encoder.inverse_transform(predictions)
    # print(y_pred[0][0])
    return y_pred[0][0]

# === GRADIO INTERFACE ===

interface = gr.Interface(
    fn=prediction,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Speech Emotion Recognition",
    description="Upload file suara untuk prediksi emosi"
)

interface.launch()

# === TEST === #

# prediction('OAF_bath_happy.wav')