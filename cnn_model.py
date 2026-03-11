import tensorflow as tf
from tensorflow.keras import layers, models

img_size = 128
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "train_val_test/train",
    image_size=(img_size, img_size),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "train_val_test/val",
    image_size=(img_size, img_size),
    batch_size=batch_size
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "train_val_test/test",
    image_size=(img_size, img_size),
    batch_size=batch_size
)

import json

# ✅ SAVE CLASS NAMES BEFORE NORMALIZATION
class_names = train_ds.class_names
num_classes = len(class_names)

with open("class_indices.json", "w") as f:
    json.dump(class_names, f)

print("Classes saved to class_indices.json:", class_names)

# # ✅ NORMALIZATION
# normalization_layer = layers.Rescaling(1./255)

# train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
# test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# ✅ BUILD CNN MODEL
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128,128,3)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

test_loss, test_accuracy = model.evaluate(test_ds)
print("Test Accuracy:", test_accuracy)

model.save("plant_disease_cnn.h5")
model.save("plant_disease_cnn.keras")