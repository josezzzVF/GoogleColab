from skimage.io import imread
from skimage.transform import resize
from skimage.exposure import equalize_adapthist

def preprocess_image(image, target_size=(128, 128)):
    image = resize(image, target_size, mode='reflect', anti_aliasing=True)
    image = equalize_adapthist(image)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    return image

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    image = imread(image_path, as_gray=True)
    image_original = resize(image, target_size, mode='reflect', anti_aliasing=True)
    image_preprocessed = preprocess_image(image, target_size)

    return image_preprocessed, image_original
