import os
import shutil
import random

def create_dataset(path_to_source_directory, 
                   path_to_output_directory, 
                   train_percent, test_percent=0.2, val_percent=0.1, 
                   shuffle=True):
    if not (0 <= train_percent <= 1 and 0 <= test_percent <= 1 and 0 <= val_percent <= 1):
        raise ValueError("Percentages must be between 0 and 1.")
    if not abs(train_percent + test_percent + val_percent - 1) < 1e-6:
        raise ValueError("Percentages must sum to 1.")
    # 1. Removing all folders/ files from output directory.
    if not os.path.exists(path_to_output_directory):
        print(f"Directory {path_to_output_directory} does not exist.")
        return
    for item in os.listdir(path_to_output_directory):
        item_path = os.path.join(path_to_output_directory, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path) 
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Failed to delete {item_path}. Reason: {e}")
            
    # 2. Creating train, val and test directories.
    os.makedirs(f"{path_to_output_directory}/images")
    os.makedirs(f"{path_to_output_directory}/images/train")
    os.makedirs(f"{path_to_output_directory}/images/val")
    os.makedirs(f"{path_to_output_directory}/images/test")
    
    os.makedirs(f"{path_to_output_directory}/labels")
    os.makedirs(f"{path_to_output_directory}/labels/train")
    os.makedirs(f"{path_to_output_directory}/labels/val")
    os.makedirs(f"{path_to_output_directory}/labels/test")
    
    # 3. Saving path to images and labels from directory.
    path_to_source_images = f"{path_to_source_directory}/images"
    target_train_images = f"{path_to_output_directory}/images/train"
    target_val_images = f"{path_to_output_directory}/images/val"
    target_test_images = f"{path_to_output_directory}/images/test"
    
    path_to_source_labels = f"{path_to_source_directory}/labels"
    target_train_labels = f"{path_to_output_directory}/labels/train"
    target_val_labels = f"{path_to_output_directory}/labels/val"
    target_test_labels = f"{path_to_output_directory}/labels/test"
    
    # 4. Saving all paths to training images.
    paths_to_images = []
    path_to_image = path_to_source_images
    for category in os.listdir(path_to_image):
        path_to_category = f"{path_to_image}/{category}"
        for image in os.listdir(path_to_category):
            images = os.listdir(path_to_category)
            paths_to_images.append(f"{path_to_category}/{image}")
    if shuffle:
        random.shuffle(paths_to_images)
    n = len(paths_to_images)
    train_end = int(train_percent * n)
    test_end = train_end + int(test_percent * n)
    train_set = paths_to_images[:train_end]
    test_set = paths_to_images[train_end:test_end]
    val_set = paths_to_images[test_end:]
    
    print(f"Length of train set: {len(train_set)}")
    print(f"Length of val set: {len(val_set)}")
    print(f"Length of test set: {len(test_set)}")
    
    # 5. Moving images to specific categories (training, val, test).
    for source_image_path in train_set:
        source_image_file = str(source_image_path).split("/")[-1]
        source_label_file = str(source_image_file).split(".")[0]
        source_label_file = f"{source_label_file}.txt"
        category = source_label_file.split("_")[0]
        target_image_path = f"{target_train_images}/{source_image_file}"
        target_label_path = f"{target_train_labels}/{source_label_file}"
        source_label_path = f"{path_to_source_labels}/{category}/{source_label_file}"
        shutil.copy(source_image_path, target_image_path)
        shutil.copy(source_label_path, target_label_path)
        
    for source_image_path in val_set:
        source_image_file = str(source_image_path).split("/")[-1]
        source_label_file = str(source_image_file).split(".")[0]
        source_label_file = f"{source_label_file}.txt"
        category = source_label_file.split("_")[0]
        target_image_path = f"{target_val_images}/{source_image_file}"
        target_label_path = f"{target_val_labels}/{source_label_file}"
        source_label_path = f"{path_to_source_labels}/{category}/{source_label_file}"
        shutil.copy(source_image_path, target_image_path)
        shutil.copy(source_label_path, target_label_path)
        
    for source_image_path in test_set:
        source_image_file = str(source_image_path).split("/")[-1]
        source_label_file = str(source_image_file).split(".")[0]
        source_label_file = f"{source_label_file}.txt"
        category = source_label_file.split("_")[0]
        target_image_path = f"{target_test_images}/{source_image_file}"
        target_label_path = f"{target_test_labels}/{source_label_file}"
        source_label_path = f"{path_to_source_labels}/{category}/{source_label_file}"
        shutil.copy(source_image_path, target_image_path)
        shutil.copy(source_label_path, target_label_path)

if __name__ == "__main__":
    create_dataset(
        path_to_source_directory="./SMOTE_balanced_flags",
        path_to_output_directory="./YOLO_SMOTE_balanced_flags",
        train_percent=0.7, test_percent=0.2, val_percent=0.1, shuffle=True
    )