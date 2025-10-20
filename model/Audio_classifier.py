
import matplotlib.pylab as plt
import seaborn as sns
from Model_definetions import build_audio_model
from Data import Audio_classification_data
from itertools import cycle
import matplotlib.pyplot as plt
from tensorflow.keras import layers , models
sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

model = build_audio_model()
x = model.output
x = layers.Dense(2, activation='softmax')(x)
model = models.Model(inputs=model.input, outputs=x)

# model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
train_ds , val_ds = Audio_classification_data()
hist = model.fit(train_ds, epochs=20, validation_data=val_ds)




plt.plot(hist.history['accuracy'], label='train acc')
plt.plot(hist.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training History')
plt.show()

model.save("saved_models/Audio_classifier_model")
