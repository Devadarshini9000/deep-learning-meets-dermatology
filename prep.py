import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to the dataset folders
train_dir = "train"
test_dir = "test"
valid_dir = "valid"

# Define image size and batch size
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32

# Create ImageDataGenerators with augmentation for the training set and basic preprocessing for validation and test sets
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values to [0,1]
    rotation_range=40,  # Randomly rotate images by up to 40 degrees
    width_shift_range=0.2,  # Randomly shift images horizontally by up to 20% of the width
    height_shift_range=0.2,  # Randomly shift images vertically by up to 20% of the height
    shear_range=0.2,  # Apply random shear transformations
    zoom_range=0.2,  # Randomly zoom in or out on images
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode="nearest"  # Fill missing pixels after transformations
)

# For validation and test sets, only rescale the pixel values
valid_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load images and apply augmentation for the training set
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"  # Use categorical labels for multi-class classification
)

# Load images for the validation set
valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Load images for the test set
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False  # Do not shuffle test data to keep predictions in order
)

# Display sample augmented images
import matplotlib.pyplot as plt

# Get a batch of augmented images
x_batch, y_batch = next(train_generator)

# Plot a few augmented images
plt.figure(figsize=(10, 10))
for i in range(9):  # Display 9 images
    plt.subplot(3, 3, i+1)
    plt.imshow(x_batch[i])
    plt.axis('off')
plt.suptitle("Augmented Images", fontsize=16)
plt.show()
