from PIL import Image
import os

base_dir = os.getcwd()


def compress_bmp_to_jpeg(bmp_path, jpeg_path, quality=80):
    """
    Compresses a .bmp image to a .jpeg image with a specified quality.

    Args:
        bmp_path: Path to the input .bmp image.
        jpeg_path: Path to save the output .jpeg image.
        quality: JPEG quality level (0-100, higher is better quality and larger file size).
                 A value of 80 is generally a good balance.
    """
    try:
        # Open the BMP image using Pillow
        img = Image.open(bmp_path)

        # Check if the image is already a JPEG
        if img.format == "JPEG":
            print(f"Error: The input image '{bmp_path}' is already a JPEG. No conversion needed.")
            return

        # Convert and save as JPEG with the specified quality
        img = img.convert('RGB')  # Ensure the image is in RGB format (JPEG doesn't support some other formats)
        img.save(jpeg_path, "JPEG", quality=quality, optimize=True)  # optimize=True reduces file size
        print(f"Successfully compressed '{bmp_path}' to '{jpeg_path}' with quality {quality}.")

    except FileNotFoundError:
        print(f"Error: File not found: {bmp_path}")
    except Exception as e:
        print(f"Error: An error occurred during compression: {e}")


# Example usage:
if __name__ == "__main__":
    bmp_file = f"{base_dir}/compress.bmp"  # Replace with your BMP file
    jpeg_file = f"{base_dir}/cmopress.jpg"  # Replace with your desired JPEG output path

    # Create a dummy bmp file for the demonstration if it doesn't exist
    if not os.path.exists(bmp_file):
        print("Creating dummy 'input.bmp' for demonstration...")
        dummy_image = Image.new("RGB", (256, 256), color="red")
        dummy_image.save(bmp_file)
        print(f"Dummy BMP file '{bmp_file}' created.")

    compress_bmp_to_jpeg(bmp_file, jpeg_file, quality=80)
