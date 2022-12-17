import shutil
from pathlib import Path

base_dir = Path(__file__).parent.parent
raw_train_dir = base_dir / "data" / "raw" / "train"
raw_validation_dir = base_dir / "data" / "raw" / "validation"

# Copy 10% of train images to validation directory
for directory in raw_train_dir.iterdir():
    validation_mirror_path = str(directory).replace("train", "validation")
    validation_mirror_path = Path(validation_mirror_path)
    validation_mirror_path.mkdir(parents=True, exist_ok=True)
    # Collect image paths in each class of train directory
    image_paths = list(directory.glob("*.png"))
    # Choose the last 10% of images
    validation_images = image_paths[-int(len(image_paths) * 0.1):]

    # Copy images to validation directory
    for image_path in validation_images:
        shutil.move(image_path, validation_mirror_path)

# # Reverse the above operation
# for directory in raw_validation_dir.iterdir():
#     train_mirror_path = str(directory).replace("validation", "train")
#     train_mirror_path = Path(train_mirror_path)
#
#     # Collect image paths in each class of validation directory
#     image_paths = list(directory.glob("*.png"))
#
#     # Copy images to train directory
#     for image_path in image_paths:
#         shutil.move(image_path, train_mirror_path)
