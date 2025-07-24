import os
import shutil

import cv2
from PIL import Image


def transform_images(img_path: str):
    image = cv2.imread(img_path)
    new_img = cv2.resize(image, (1280, 720))
    return new_img


def denoise_image(img, method='fastNlMeansDenoisingColored', **kwargs):
    """
    Denoises an image using various denoising methods.

    Args:
        image_path (str): Path to the image file.
        method (str): Denoising method to use. Options:
                      'fastNlMeansDenoisingColored' (default), 'fastNlMeansDenoising',
                      'blur', 'medianBlur', 'gaussianBlur'
        **kwargs:  Keyword arguments specific to each denoising method.  For example,
                   for fastNlMeansDenoisingColored, you might pass h=10, hColor=10, templateWindowSize=7, searchWindowSize=21.

    Returns:
        numpy.ndarray: The denoised image.
    """

    if method == 'fastNlMeansDenoisingColored':
        # Best for colored images with complex noise.  Relatively slow.
        denoised_img = cv2.fastNlMeansDenoisingColored(img, None, **kwargs)

    elif method == 'fastNlMeansDenoising':
        # Good for grayscale images.  Relatively slow.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised_img = cv2.fastNlMeansDenoising(gray, None, **kwargs)  # h=10, templateWindowSize=7, searchWindowSize=21
        denoised_img = cv2.cvtColor(denoised_img, cv2.COLOR_GRAY2BGR)  # Convert back to BGR

    elif method == 'blur':
        # Simple averaging blur.  Fast, but often not very effective for denoising.
        denoised_img = cv2.blur(img, (5, 5), **kwargs)  # ksize=(5,5)

    elif method == 'medianBlur':
        # Good for salt-and-pepper noise.  Fast.
        denoised_img = cv2.medianBlur(img, 5, **kwargs)  # ksize=5 (must be odd)

    elif method == 'gaussianBlur':
        # Applies a Gaussian blur.  Fast.
        denoised_img = cv2.GaussianBlur(img, (5, 5), 0)  # ksize=(5,5), sigmaX=0

    else:
        raise ValueError(f"Invalid denoising method: {method}")

    return denoised_img


base_dir = '/Users/mohsenamoon/Desktop/IOT/my paper draft/reconstructed_256_lpips_psnr_loss_simple'

counter = 0
for img in os.listdir(f'{base_dir}/epoch_0'):
    converted_img = transform_images(f'{base_dir}/epoch_0/{img}')
    # converted_img = denoise_image(converted_img, 'gaussianBlur')
    cv2.imwrite(f"{base_dir}/recreated/{img.replace('.bmp', '')}", converted_img)
    # shutil.move(f'../new_dataset/512_training/labels/{lb}', f'../new_dataset/512_training/labels/copy_{lb}')
    counter += 1
    print(counter)
