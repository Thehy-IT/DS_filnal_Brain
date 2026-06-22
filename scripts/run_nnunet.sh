export nnUNet_raw="data/processed/nnUNet_raw"
export nnUNet_preprocessed="data/processed/nnUNet_preprocessed"
export nnUNet_results="data/processed/nnUNet_results"

mkdir -p "$nnUNet_preprocessed"
mkdir -p "$nnUNet_results"

nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
nnUNetv2_train 1 3d_fullres 0
