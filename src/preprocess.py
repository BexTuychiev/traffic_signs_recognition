from joblib import Parallel, delayed
from skimage.io import imread, imsave
from skimage.transform import resize
from pathlib import Path
from tqdm import tqdm
import warnings

DATA_DIR = Path("data")
train_dir = DATA_DIR / "raw" / "train"


def resize_image(image_path, target_size):
    """
    Resize image to target size.
    """
    image = imread(image_path) / 255.0
    image = resize(image, target_size, anti_aliasing=True)

    # Create the path to the image in the prepared directory
    target_path = str(image_path).replace("raw", "prepared")
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)

    imsave(target_path, image, )


if __name__ == "__main__":
    image_paths = []

    for directory in train_dir.iterdir():
        image_paths.extend(list(directory.glob("*.png")))

    Parallel(n_jobs=10, backend="multiprocessing")(
        delayed(resize_image)(path, (150, 150)) for path in tqdm(image_paths)
    )
