from os import listdir
from os.path import isfile, join

import cv2


def resize_from_folder(input_folder, output_folder, new_dimensions):
    """Function that reads all the files in the input folder, resize them to match the specified (width, height)
    and finally store them in the output folder
        :param input_folder: string indicating the path of the input folder
        :param output_folder: string indicating the path of the output folder
        :param new_dimensions: tuple indicating the desired width and height in pixel of the resized images
    """

    # !Attention! all the files in the folder will be considered
    onlyfiles = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]
    
    for filename in onlyfiles:
        full_path_input = join(input_folder, filename)
        img = cv2.imread(full_path_input, cv2.IMREAD_UNCHANGED)

        resized = cv2.resize(img, new_dimensions, interpolation = cv2.INTER_AREA)

        full_path_output = join(output_folder, filename)
        if not cv2.imwrite(full_path_output, resized):
            print("[ERROR] Impossible to save resized image {}".format(full_path_output))

