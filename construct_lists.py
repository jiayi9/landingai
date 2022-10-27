import os
import json
from pathlib import Path


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


def load_json(path):
    """Load a json file"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    return data


def generate_list(home_folder, split):
    folder = os.path.join(home_folder, split, "Images")
    image_list = list_files_recur(folder, "bmp")[1]
    image_name_list = [Path(x).stem[50:] for x in image_list]
    return {split: image_name_list}


def save_json(target_folder, filename, contents):
    save_path = os.path.join(target_folder, filename)
    with open(save_path, 'w', encoding='utf-8') as json_file:
        json.dump(contents, json_file, ensure_ascii=False, indent=4)
        print("write json file success!")


home_folder = r"C:\Temp\Dataset_add_ok_train30_dev4-166668888979"

dataset_info = dict(
    generate_list(home_folder, 'train'),
    **generate_list(home_folder, 'dev'),
    **generate_list(home_folder, 'test')
)

for dataset in ['train', 'dev', 'test']:
    print(len(dataset_info[dataset]))

save_json(home_folder, "dataset_info.json", dataset_info)