import os
import numpy as np
from PIL import Image


def create_chips(input_image, label_image, out_dir, stride=256, window_size=256,
                 geo_tagged=True):
    """
    Creates image chips of given windows_size from large image.
    Output image filenames are stored with index.

    :param input_image: Input image file path.
    :param label_image: The label image file path.
    :param out_dir: Output dir for saving small size images called chips.
    :param stride: The window step size.
    :param window_size: The output image size. Default is 256.
    :param geo_tagged: Weather to create image with geotagged information if present in the source
                       input image. Default is False.
    :return:
    """

    if geo_tagged:
        import rasterio as rs
        image = rs.open(input_image)
        label = rs.open(label_image)
        imgarr = image.read()
        invarr = label.read(1)
    else:
        image = Image.open(input_image)
        label = Image.open(label_image)
        imgarr = np.array(image).transpose(2, 0, 1)  # HWC -> #CHW
        invarr = np.array(label)

    print("Processing:")
    print("Image file:", input_image)
    print("Label file:", label_image)
    print("Shape of training data  is:", imgarr.shape)
    print("Shape of label is:", invarr.shape)

    images_out_dir = os.path.join(out_dir, "images")
    labels_out_dir = os.path.join(out_dir, "labels")

    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    image_filename, image_ext = os.path.splitext(os.path.split(input_image)[1])
    label_filename, label_ext = os.path.splitext(os.path.split(label_image)[1])

    index = 0

    for i in np.arange(0, imgarr.shape[1], stride):
        for j in np.arange(0, imgarr.shape[2], stride):
            img = imgarr[:, i:i + window_size, j:j + window_size]
            lbl = invarr[i:i + window_size, j:j + window_size]

            if img.shape[1] != window_size or img.shape[2] != window_size:
                continue

            img_out_file = f"{image_filename}_{index}{image_ext}"
            lbl_out_file = f"{label_filename}_{index}{label_ext}"

            if geo_tagged:
                x, y = (j * image.transform[0] + image.transform[2]), (image.transform[5] + i * image.transform[4])
                transform = [image.transform[0], 0, x, 0, image.transform[4], y]

                with rs.open(os.path.join(images_out_dir, img_out_file), "w", driver='GTiff', count=imgarr.shape[0],
                             dtype=imgarr.dtype, width=window_size, height=window_size, transform=transform,
                             crs=image.crs) as raschip:
                    raschip.write(img)

                with rs.open(os.path.join(labels_out_dir, lbl_out_file), "w", driver='GTiff', count=1,
                             dtype=invarr.dtype,
                             width=window_size, height=window_size, transform=transform, crs=image.crs) as lblchip:
                    lblchip.write(lbl, 1)
            else:
                Image.fromarray(np.array(img).transpose(1, 2, 0)).save(os.path.join(images_out_dir, img_out_file))
                palette = label.getpalette()
                if palette is not None:
                    lblchip = Image.fromarray(np.array(lbl), "P")
                    lblchip.putpalette(palette)
                else:
                    lblchip = Image.fromarray(np.array(lbl))

                lblchip.save(os.path.join(labels_out_dir, lbl_out_file))

            index = index + 1

    print(f"Saved {index} images.")
    print("Processed!")
