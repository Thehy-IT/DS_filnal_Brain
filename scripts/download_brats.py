import os
from monai.apps import download_and_extract

dataset_source_url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
destination_directory = "data/raw"

if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

archive_file_path = os.path.join(destination_directory, "Task01_BrainTumour.tar")

download_and_extract(url=dataset_source_url, filepath=archive_file_path, output_dir=destination_directory)
