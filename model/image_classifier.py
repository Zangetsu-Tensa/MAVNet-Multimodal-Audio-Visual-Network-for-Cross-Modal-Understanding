
import matplotlib.pylab as plt
import seaborn as sns
from Model_definetions import build_visual_model
from Data import image_classification_data
from itertools import cycle
from tensorflow.keras import layers , models
sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

model = build_visual_model()
x = model.output
x = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs=model.input, outputs=x)

# model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
train_ds , val_ds = image_classification_data()
hist = model.fit(train_ds, epochs=20, validation_data=val_ds)




plt.plot(hist.history['accuracy'], label='train acc')
plt.plot(hist.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training History')
plt.show()

model.save("saved_models/Image_classifier_model")
