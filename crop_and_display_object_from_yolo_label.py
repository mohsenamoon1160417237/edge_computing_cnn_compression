import os

import cv2
from PIL import Image

base_dir = os.getcwd()


# img = cv2.imread(f"{base_dir}/compress.bmp")
# print(img.shape)


# Dimensions of the bmp images are (720, 1280, 3)
# class_id, x_center, y_center, width, height
# x_center(float): x-coordinate of the center of the bounding box,normalized to the image width which is between 0 and 1
# calculation: (x_min + x_max) / (2 * image_width)
# where x_min is the x_coordinate of the top-left of the bounding box.
# x_max is the x_coordinate of the bottom-right of the bounding box.
# image_width is the width of the whole image in pixels.
# y_center (float): the y-coordinate of the center of the bounding box, normalized to the image height. it is between
# 0 and 1.
# calculation: y_center = (y_min + y_max) / (2 * image_height)
# width (float): the width of the bounding box, normalized to the image width. it is between 0 and 1.
# calculation: width = (x_max - x_min) / image_width
# calculation: height = (y_max - y_min) / image_height
# Make sure that the floating numbers have sufficient accuracy with at least 5 decimal points.

def recreate_new_label(x_center, y_center, width, height):
    scale_x = 512 / 1280
    scale_y = 512 / 720
    scale = scale_x
    offset_x = (512 - 1280 * scale) / 2
    offset_y = (512 - 720 * scale) / 2
    x_center_new = (x_center * 1280 * scale + offset_x) / 512
    y_center_new = (y_center * 720 * scale + offset_y) / 512
    width_new = (width * 1280 * scale) / 512
    height_new = (height * 720 * scale) / 512
    return x_center_new, y_center_new, width_new, height_new


def crop_and_display_object_from_yolo_label(image_path, x_center, y_center, width, height):
    """
    Crops an object from an image based on YOLO label values (x_center, y_center, width, height)
    and displays the cropped object.

    Args:
        image_path (str): Path to the image file.
        x_center (float): Normalized x-coordinate of the center of the bounding box (0 to 1).
        y_center (float): Normalized y-coordinate of the center of the bounding box (0 to 1).
        width (float): Normalized width of the bounding box (0 to 1).
        height (float): Normalized height of the bounding box (0 to 1).
    """
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size

        # Calculate bounding box coordinates in pixel values
        x_min = int((x_center - width / 2) * img_width)
        y_min = int((y_center - height / 2) * img_height)
        x_max = int((x_center + width / 2) * img_width)
        y_max = int((y_center + height / 2) * img_height)

        # Check if bounding box coordinates are valid
        if not (0 <= x_min < x_max <= img_width and 0 <= y_min < y_max <= img_height):
            raise ValueError("Invalid bounding box coordinates.")

        # Crop the image
        cropped_img = img.crop((x_min, y_min, x_max, y_max))

        # Display the cropped image
        cropped_img.show()  # This opens the image in your default image viewer

        # Optionally, you can also save the cropped image
        # cropped_img.save("cropped_object.jpg")

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Example Usage:  (Replace with your actual file path and YOLO label values)
# image_path = f"{base_dir}/00001776.bmp"
image_path = f"{base_dir}/reconstructed_train_valitaion_256/recreated/reconstructed_00001776.jpg"
img = cv2.imread(image_path)
print(img.shape)
# x_center_new, y_center_new, width_new, height_new = recreate_new_label(0.904688, 0.564583, 0.028125, 0.034722199999999995)
crop_and_display_object_from_yolo_label(image_path, 0.382422, 0.6381939999999999, 0.0820312, 0.101389)

# crop_and_display_object_from_yolo_label(image_path, x_center_new, y_center_new, width_new, height_new)
