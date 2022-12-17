import shutil
from pathlib import Path

base_dir = Path(__file__).parent.parent
raw_train_dir = base_dir / "data" / "raw" / "train"

# Copy 10% of train images to validation directory
for directory in raw_train_dir.iterdir():
    # Collect image paths in each class of train directory
    image_paths = list(directory.glob("*.png"))
    # Choose the last 10% of images
    validation_images = image_paths[-int(len(image_paths) * 0.1):]

    for image_path in validation_images:
        # Create the destination path for the image
        validation_path = str(image_path).replace("train", "validation")
        Path(validation_path).mkdir(parents=True, exist_ok=True)
        print(validation_path)
        # Move the image to the new path
        shutil.move(image_path, validation_path)
