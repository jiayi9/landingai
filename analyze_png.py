import numpy as np
import os
import cv2
def list_files_recur(path, fmt=".bmp"):
    """List files recursively in a folder with the specified extension name"""
    file_paths = []
    file_names = []
    path = str(path)
    for r, d, f in os.walk(path):
        for file_name in f:
            if fmt in file_name or fmt.lower() in file_name:
                file_paths.append(os.path.join(r, file_name))
                file_names.append(file_name)
    return [file_paths, file_names]


files = list_files_recur(r"C:\Users\kjhm285\Downloads\test_jiayi_2_download-166618297675.tar\test_jiayi_2_download-166618297675", "png")[0]

n = 0

for file in files:
    img = cv2.imread(file)
    img[img == 6] = 0
    img[img == 7] = 0

    if np.alltrue(img==0):
        n = n+1

len(files)

len(files) - n
