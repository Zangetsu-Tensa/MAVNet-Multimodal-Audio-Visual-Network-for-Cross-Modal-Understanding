from preprocessing import load_audio,load_image
import numpy as np
def predict(audio_path, image_path, model):
    audio_input = np.expand_dims(load_audio(audio_path), axis=0)
    visual_input = np.expand_dims(load_image(image_path), axis=0)
    pred = model.predict([audio_input, visual_input])
    return np.argmax(pred)
