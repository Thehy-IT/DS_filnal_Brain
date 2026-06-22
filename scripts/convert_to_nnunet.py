import os
import json
import shutil
import nibabel as nib

source_dataset_directory = "data/raw/Task01_BrainTumour"
target_nnunet_directory = "data/processed/nnUNet_raw/Dataset001_BrainTumour"

training_images_directory = os.path.join(target_nnunet_directory, "imagesTr")
training_labels_directory = os.path.join(target_nnunet_directory, "labelsTr")

os.makedirs(training_images_directory, exist_ok=True)
os.makedirs(training_labels_directory, exist_ok=True)

with open(os.path.join(source_dataset_directory, "dataset.json"), "r") as file_handle:
    dataset_metadata = json.load(file_handle)

total_training_cases = dataset_metadata["numTraining"]

for training_case in dataset_metadata["training"]:
    image_relative_path = training_case["image"]
    label_relative_path = training_case["label"]
    
    case_identifier = os.path.basename(image_relative_path).split(".")[0]
    
    absolute_image_path = os.path.join(source_dataset_directory, image_relative_path)
    absolute_label_path = os.path.join(source_dataset_directory, label_relative_path)
    
    four_dimensional_image = nib.load(absolute_image_path)
    image_data_array = four_dimensional_image.get_fdata()
    image_spatial_affine = four_dimensional_image.affine
    image_header_metadata = four_dimensional_image.header
    
    for channel_index in range(4):
        three_dimensional_slice = image_data_array[..., channel_index]
        single_channel_nifti = nib.Nifti1Image(three_dimensional_slice, image_spatial_affine, image_header_metadata)
        destination_image_file = os.path.join(training_images_directory, f"{case_identifier}_{channel_index:04d}.nii.gz")
        nib.save(single_channel_nifti, destination_image_file)
        
    destination_label_file = os.path.join(training_labels_directory, f"{case_identifier}.nii.gz")
    shutil.copy(absolute_label_path, destination_label_file)

nnunet_configuration_metadata = {
    "channel_names": {
        "0": "FLAIR",
        "1": "T1",
        "2": "T1c",
        "3": "T2"
    },
    "labels": {
        "background": 0,
        "necrotic": 1,
        "edema": 2,
        "enhancing": 3
    },
    "numTraining": total_training_cases,
    "file_ending": ".nii.gz"
}

with open(os.path.join(target_nnunet_directory, "dataset.json"), "w") as file_handle:
    json.dump(nnunet_configuration_metadata, file_handle, indent=4)
