import tensorflow as tf
from tensorflow.keras import layers, models

# Dataset paths
train_dir = "train_val_test/train"
val_dir = "train_val_test/val"
test_dir = "train_val_test/test"

# Image settings
img_size = (128,128)
batch_size = 32

# Load datasets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size
)

# Get class names
class_names = train_dataset.class_names
num_classes = len(class_names)

print("Number of classes:", num_classes)

# Speed up data loading
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128,128,3),
    include_top=False,
    weights="imagenet"
)

# Freeze base model (important for fast training)
base_model.trainable = False

# Build model
model = models.Sequential([
    
    layers.Rescaling(1./255, input_shape=(128,128,3)),
    
    base_model,
    
    layers.GlobalAveragePooling2D(),
    
    layers.Dense(128, activation='relu'),
    
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model (very fast)
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3
)

# Evaluate model
loss, accuracy = model.evaluate(test_dataset)

print("Test Accuracy:", accuracy)

# Save model
model.save("plant_disease_mobilenet_model.h5")

print("Model saved successfully")