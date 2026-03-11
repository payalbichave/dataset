import tensorflow as tf
from tensorflow.keras import layers, models

train_dir = "train_val_test/train"
val_dir = "train_val_test/val"

# Smaller image size for faster training
img_size = (64,64)

batch_size = 32

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size
)
class_names = train_dataset.class_names
num_classes = len(class_names)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size
)

# Speed up data loading
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# Lightweight CNN model
model = models.Sequential([

    layers.Rescaling(1./255, input_shape=(64,64,3)),

    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train for fewer epochs (faster)
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5
)

# Test dataset
test_dir = "train_val_test/test"

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size
)

loss, accuracy = model.evaluate(test_dataset)

print("Test Accuracy:", accuracy)

model.save("plant_disease_cnn_model.h5")

print("Model saved successfully")
