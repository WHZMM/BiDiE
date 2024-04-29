# UTF-8

# This is the 2-nd step

# We need to obtain the face-parsing result of the human images.
# The result be used to blend the image in subsequent steps.

# We use Facer (https://github.com/FacePerceiver/facer) for face-parsing.
# The face image is divided into 11 types of areas, and their grayscale values are i*25 (i=0,1,2, ... ,10)
# For the values of 11 types of areas, please refer to the sample images in: BiDiE/gen_sketch/face_parsing/
# Store the face-parsing images in the folder "face_parsing"

# You can also use other models to obtain face_parsing results.
