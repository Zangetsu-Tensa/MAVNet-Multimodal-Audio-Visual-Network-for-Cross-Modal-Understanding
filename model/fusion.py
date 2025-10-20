import tensorflow as tf
from tensorflow.keras import layers, models, applications
import librosa
import numpy as np
import cv2
import os






def build_fusion_model(num_classes):
    audio_model = build_audio_model()
    visual_model = build_visual_model()

    audio_input = audio_model.input
    visual_input = visual_model.input

    audio_feat = audio_model.output
    visual_feat = visual_model.output

    merged = layers.concatenate([audio_feat, visual_feat])
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=[audio_input, visual_input], outputs=output)
    return model