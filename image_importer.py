import requests
import cv2
import numpy as np
import os


def import_images_from_web(urls, normalize_float_format=True, resize=None|tuple):
    '''
        Parameters:
        urls (list): The set of images to import.
        resize (None|tuple): A desired size for the image set.

        Returns:
        np.array: An array where each element is an image in the opencv mat format.
    '''
    images = []
    for url in urls:
        image = np.asarray(bytearray(requests.get(url).content))
        raw_data = cv2.imdecode(image, flags=cv2.IMREAD_COLOR)
        if (resize):
            raw_data = cv2.resize(raw_data, resize)

        if normalize_float_format:
            raw_data = raw_data.astype(dtype=np.float32)
            raw_data /= 255.0
        images.append(raw_data)

    return np.array(images)



def import_images_from_folder(folder_directory, normalize_float_format=True, resize=None|tuple):
    '''
        Parameters:
        folder_directory (string): The path leading to the set of images to import.
        resize (None|tuple): A desired size for the image set.

        Returns:
        np.array: An array where each element is an image in the opencv mat format.
    '''
    images = []

    file_names = os.listdir(folder_directory)

    for file in file_names:
        file_path = folder_directory + '\\'+ file
        raw_data = cv2.imread(file_path, flags=cv2.IMREAD_COLOR)
        if (resize):
            raw_data = cv2.resize(raw_data, resize)

        if normalize_float_format:
            raw_data = raw_data.astype(dtype=np.float32)
            raw_data /= 255.0
        images.append(raw_data)

    return np.array(images)
