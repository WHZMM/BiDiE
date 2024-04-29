# UTF-8

# This is the 1-st step

# We first need to obtain the segmentation result of the human image by U^2-Net.
# This result will be used to remove the background of the image in subsequent steps.

# We used the open source code and pre-trained model of the paper "U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection".
# The web link is: https://github.com/xuebinqin/U-2-Net
# They provide an image segmentation model for removing background from human images.
# In the resulting mask images, the human area is white and the background area is black.
# Store the mask images in the folder "bg_mask"

# You can also use other models to obtain masks.
