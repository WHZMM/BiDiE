# UTF-8

# This is the 4-st step

# In this step, texture information is extracted from the input image.
# This will be used in subsequent steps to form the texture part of the face sketch.
# # Store the texture images in the folder "texture"

# There is a hyperparameter here that you can modify according to your needs.

import glob
from skimage.morphology import disk
import skimage.filters.rank as sfr
import cv2
import tqdm


if __name__ == "__main__":
    """ Input and output folders """
    folder_in = "./gen_sketch/photo"
    folder_out = "./gen_sketch/texture"
    """ Input and output folders """

    """ the hyperparameter """
    param_1 = 20  # default = 20
    """ the hyperparameter """

    print("Get input file list")
    photo_paths = sorted(glob.glob(folder_in + "/*.*"))
    print(f"{len(photo_paths)} images found")
    
    # start process
    for photo_path in tqdm.tqdm(photo_paths):
        file_name = (photo_path.rsplit("/", 1)[-1]).rsplit(".", 1)[0]
        outpath = folder_out + "/" + file_name + ".png"

        photo = cv2.imread(photo_path)
        grey = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)  # to gary image
        invert = cv2.bitwise_not(grey)  # invert
        # blur_img = cv2.GaussianBlur(invert, (param_1, param_1), 0)  # Blur can be used. But using minimum filter is better according to my tests.
        photo_min = sfr.minimum(invert, disk(param_1*0.1))

        # color dodge:
        inverse_blur = cv2.bitwise_not(photo_min)
        sketch_img = cv2.divide(grey, inverse_blur, scale=255.0)
        
        # save file
        cv2.imwrite(outpath, sketch_img)
    
    print("Done.")

