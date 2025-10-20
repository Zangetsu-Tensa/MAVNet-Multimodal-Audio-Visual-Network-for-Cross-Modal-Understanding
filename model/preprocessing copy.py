


import tensorflow as tf

def load_audio(filepath, label):
    audio = tf.io.read_file(filepath)
    audio, _ = tf.audio.decode_wav(audio)
    audio = tf.squeeze(audio, axis=-1)  # [samples]

    stft = tf.signal.stft(audio, frame_length=512, frame_step=256)
    spectrogram = tf.abs(stft)

    # Resize and normalize
    spectrogram = tf.expand_dims(spectrogram, axis=-1)  # [freq, time, 1]
    spectrogram = tf.image.resize(spectrogram, [128, 128])
    # spectrogram = tf.math.log(spectrogram + 1e-6)  # log-mel like
    
    return spectrogram, label

def load_image(filepath, label):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label
