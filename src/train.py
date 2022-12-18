from pathlib import Path
import tensorflow as tf
import yaml

# Set the paths to the train and validation directories
base_dir = Path(__file__).parent.parent
data_dir = base_dir / "data"
params = yaml.safe_load(open("params.yaml"))["train"]

# Create an ImageDataGenerator object for the train set
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=45,  # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    zoom_range=0.2,  # Randomly zoom in and out of images
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode="nearest",  # Fill in missing pixels with nearest neighbor
)

# Generate training data from the train directory
train_generator = data_gen.flow_from_directory(
    data_dir / "prepared" / "train",  # Target directory
    target_size=(params['image_width'], params['image_height']),  # Resize images
    batch_size=params['batch_size'],  # Set batch size
    class_mode="categorical",  # Use categorical labels
)

validation_generator = data_gen.flow_from_directory(
    data_dir / "prepared" / "validation",  # Target directory
    target_size=(params['image_width'], params['image_height']),  # Resize images
    batch_size=params['batch_size'],  # Set batch size
    class_mode="categorical",  # Use categorical labels
)


def get_model():
    """Define the model to be fit"""
    # Define a CNN model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu",
                                   input_shape=(
                                       params['image_width'], params['image_height'], 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            # tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu"),
            # tf.keras.layers.MaxPooling2D(2, 2),
            # tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation="relu"),
            # tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(43, activation="softmax"),
        ]
    )

    # Compile the model
    model.compile(
        # Use categorical cross-entropy loss
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
        metrics=["accuracy"],  # Calculate accuracy
    )

    return model


def main():
    # Get the model
    model = get_model()
    # Create a path to save the model
    model_path = base_dir / "models"
    model_path.mkdir(parents=True, exist_ok=True)

    # Define callbacks
    callbacks = [tf.keras.callbacks.ModelCheckpoint(model_path / "model.keras",
                                                    monitor="val_accuracy",
                                                    save_best_only=True),
                 tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5)]
    # Fit the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=params['n_epochs'],
        validation_data=validation_generator,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
