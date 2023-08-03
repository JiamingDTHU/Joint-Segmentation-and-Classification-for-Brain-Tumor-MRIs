import os
import random
import shutil

random.seed(22)

# get the name of files in dataset
if __name__ == "__main__":
    current_dir = os.getcwd()
    if "Joint-Segmentation-and-Classification-for-Brain-Tumor-MRIs" in current_dir.split("/"):
        if current_dir.split("/")[-1] == "Joint-Segmentation-and-Classification-for-Brain-Tumor-MRIs":  
            dataset_dir = os.path.join(current_dir, "dataset")
        elif current_dir.split("/")[-1] == "scripts":
            dataset_dir = os.path.join("/", *current_dir.split("/")[:-1], "dataset")
        elif current_dir.split("/")[-1] == "dataset":
            dataset_dir = current_dir
    else:
        raise FileNotFoundError("check your work directory")
    subdirs = ['train', 'test', 'valid']
    filename_list = []
    for subdir in subdirs:
        os.makedirs(os.path.join(dataset_dir, subdir), exist_ok=True)
        filename_list.extend([os.path.join(dataset_dir, subdir, file) for file in os.listdir(os.path.join(dataset_dir, subdir)) if file.endswith('.mat')])

    random.shuffle(filename_list)
    
    # shuffling and separate data file name list
    total_files = len(filename_list)
    train_ratio, test_ratio, valid_ratio = 0.8, 0.1, 0.1
    train_files = filename_list[:int(total_files * train_ratio)]
    test_files = filename_list[int(total_files * train_ratio):int(total_files * (train_ratio + test_ratio))]
    valid_files = filename_list[int(total_files * (train_ratio + test_ratio)):]

    # move corresponding file to its dir
    for file in train_files:
        shutil.move(file, os.path.join(dataset_dir, 'train', file.split("/")[-1]))

    for file in test_files:
        shutil.move(file, os.path.join(dataset_dir, 'test', file.split("/")[-1]))

    for file in valid_files:
        shutil.move(file, os.path.join(dataset_dir, 'valid', file.split("/")[-1]))

    print("dataset reshuffling done. train : val : test = 8 : 1 : 1")
