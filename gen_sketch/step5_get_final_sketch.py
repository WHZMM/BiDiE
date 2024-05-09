# UTF-8

# This is the 5-st step, the last step

# In the previous steps, we have obtained: segmentation results, outlines, textures.
# We can easily get a grayscale image of input images. (brightness)

# In this step, we need to mix the 3 types of information: outline, texture, and brightness (based on the segmentation results)
# Store the final sketch in the folder "final_sketch"

# There are many hyperparameters in this step, please adjust them freely as needed.

# If our algorithm inspires you, please cite our paper (*^_^*)


import glob
from PIL import Image, ImageOps, ImageFilter, ImageChops, ImageEnhance
import tqdm


if __name__ == "__main__":
    """ Input and output folders """
    folder_face_parsing = "./gen_sketch/face_parsing"
    folder_bg_mask = "./gen_sketch/bg_mask"  # back ground mask
    folder_photo = "./gen_sketch/photo"
    folder_outline = "./gen_sketch/outline"
    folder_texture = "./gen_sketch/texture"
    folder_final_sketch = "./gen_sketch/final_sketch"
    """ Input and output folders """

    """ the hyperparameter """
    param_1 = 0.5  # (default = 0.5) param related to the light and dark brightness of the skin area
    param_2 = 1.3  # (default = 1.3) param related to the light and dark brightness of the skin area
    param_3 = 0.35  # (default = 0.35) param related to the brightness of the outline
    param_4 = 0.8  # (default = 0.8) param related to the smoothness of the outline and other areas
    """ the hyperparameter """

    print("Get input file list ...   ", end="")
    paths_photo = sorted(glob.glob(folder_photo + "/*.*"))
    paths_face_parsing = sorted(glob.glob(folder_face_parsing + "/*.*"))
    paths_bg_mask = sorted(glob.glob(folder_bg_mask + "/*.*"))
    paths_outline = sorted(glob.glob(folder_outline + "/*.*"))
    paths_texture = sorted(glob.glob(folder_texture + "/*.*"))

    assert len(paths_photo) == len(paths_face_parsing) == len(paths_bg_mask) == len(paths_outline) == len(paths_texture)
    print(f"{len(paths_photo)} groups of images found.")
    
    # start combine the sketches !
    for idx in tqdm.tqdm(range(len(paths_photo))):
        # get paths
        path_photo = paths_photo[idx]
        path_face_parsing = paths_face_parsing[idx]
        path_bg_mask = paths_bg_mask[idx]
        path_outline = paths_outline[idx]
        path_texture = paths_texture[idx]
        file_name = (path_photo.rsplit("/", 1)[-1]).rsplit(".", 1)[0]
        path_final_sketch = folder_final_sketch + "/" + file_name + ".png"

        # read
        img_rgb = Image.open(path_photo)
        img_size = img_rgb.size  # Use original image size
        img_gray = img_rgb.convert("L")
        img_parsing = Image.open(path_face_parsing).convert("L").resize(img_size)
        img_bgmask = Image.open(path_bg_mask).convert("L").resize(img_size)
        img_outline = Image.open(path_outline).convert("L").resize(img_size)
        img_texture = Image.open(path_texture).convert("L").resize(img_size)

        # new
        img_white = Image.new(mode="L", size=img_size, color=(255,))  # gen white img
        img_black = Image.new(mode="L", size=img_size, color=(0,))  # gen white img

        # Get hair mask image (based on the face-parsing results)
        enhancer_parsing_hair = ImageEnhance.Brightness(img_parsing)  # input parising result: hair = 250
        img_hairmask = enhancer_parsing_hair.enhance(1.02)  # hair from 250 to 255
        enhancer_parsing_hair = ImageEnhance.Brightness(ImageOps.invert(img_hairmask))  # hair = 0. others > zero
        img_hairmask = enhancer_parsing_hair.enhance(20)  # this is hair mask! (hair = 0. others = 255)

        # Get skin mask image (based on the face-parsing results)
        enhancer_parsing_bg = ImageEnhance.Brightness(img_parsing)
        img_head_mask = enhancer_parsing_bg.enhance(20)  # skin and hair = 255, others = 0
        img_skinmask = ImageChops.darker(img_head_mask, img_hairmask)  # skin = 255, others = 0

        # Remove background from outline/texture/brightness (based on BackGround mask)
        img_gray_bgmasked = Image.composite(img_white, img_gray, ImageOps.invert(img_bgmask))
        img_outline_bgmasked = Image.composite(img_white, img_outline, ImageOps.invert(img_bgmask))
        img_texture_bgmasked = Image.composite(img_white, img_texture, ImageOps.invert(img_bgmask))

        # Adjust brightness according to skin area, so that people with different skin colors can all have reasonable sketches
        img_skin_contrast = ImageOps.equalize(img_gray_bgmasked, mask=img_skinmask)  # Histogram equalization based on skin area
        img_skin_contrast = Image.blend(img_skin_contrast, img_white, param_1)
        enhancer_skin = ImageEnhance.Brightness(img_skin_contrast)
        img_skin_contrast = enhancer_skin.enhance(param_2) # Makes the skin area moderately bright and preserves the contrast between light and dark

        # Adjust outline from black-white to gray-white
        img_outline_bgmasked = Image.blend(img_outline_bgmasked, img_white, param_3)  # Outline from black-white to gray-white

        # Blend outline, texture, and brightness
        img_outline_bgmasked = img_outline_bgmasked.filter(ImageFilter.GaussianBlur(param_4))  # Blur, so that blending can proceed smoothly
        img_result_wo_texture = ImageChops.darker(img_skin_contrast, img_outline_bgmasked)  # Blend outline and brightness (You can modify it if needed)
        img_result = ImageChops.multiply(img_result_wo_texture, img_texture_bgmasked).convert("L")  # Blend outline-brightness and texture (You can modify it if needed)

        # save final sketch
        img_result.save(path_final_sketch)
    
    print("Done.")
