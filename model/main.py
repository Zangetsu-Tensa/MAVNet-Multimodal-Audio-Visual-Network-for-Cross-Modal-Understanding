from Model_definetions import build_fusion_model
from Data import image_classification_data, Audio_classification_data
from inference import predict



if __name__ == '__main__':
    model = build_fusion_model(num_classes=4) #2^2=4
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    audio_sample = 'data/audio/dog_bark.wav'
    image_sample = 'data/video/dog_frame.jpg'

    pred_class = predict(audio_sample, image_sample, model)
    print(f"Predicted Class: {pred_class}")
