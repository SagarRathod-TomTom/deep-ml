import os
import numpy as np
import rasterio as rs


def create_chips(input_image, label_image, out_dir, stride=256, window_size=256):
    image = rs.open(input_image)
    label = rs.open(label_image)
    imgarr = image.read()
    invarr = label.read(1)

    print("Processing:")
    print(f"Image file:{input_image}")
    print(f"Label file:{input_image}")
    print("Shape of training data  is: ", imgarr.shape)
    print("Shape of label is: ", invarr.shape)

    images_out_dir = os.path.join(out_dir, "images")
    labels_out_dir = os.path.join(out_dir, "labels")

    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    image_filename = os.path.splitext(os.path.split(input_image)[1])[0]
    label_filename = os.path.splitext(os.path.split(label_image)[1])[0]

    index = 0
    for i in np.arange(0, imgarr.shape[1], stride):
        for j in np.arange(0, imgarr.shape[2], stride):
            img = imgarr[:, i:i+window_size, j:j+window_size]
            lbl = invarr[i:i+window_size, j:j+window_size]

            x, y = (j*image.transform[0]+image.transform[2]), (image.transform[5]+i*image.transform[4])
            transform = [image.transform[0], 0, x, 0, image.transform[4], y]
            index += 1

            img_out_file = f"{image_filename}_{index}.tif"
            lbl_out_file = f"{label_filename}_{index}.tif"

            with rs.open(os.path.join(images_out_dir, img_out_file), "w", driver='GTiff', count=imgarr.shape[0], dtype=imgarr.dtype,
                         width=window_size, height=window_size, transform=transform, crs=image.crs) as raschip:
                raschip.write(img)

            with rs.open(os.path.join(labels_out_dir, lbl_out_file), "w", driver='GTiff', count=1, dtype=invarr.dtype,
                         width=window_size, height=window_size, transform=transform, crs=image.crs) as lblchip:
                lblchip.write(lbl, 1)

    print("Processed!")
