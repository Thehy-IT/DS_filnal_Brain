from monai.networks.nets import SwinUNETR

def initialize_swin_unetr_model(input_channels_count=3, target_classes_count=4, spatial_patch_dimensions=(224, 224)):
    return SwinUNETR(
        spatial_dims=2,
        in_channels=input_channels_count,
        out_channels=target_classes_count,
        feature_size=48,
        use_checkpoint=True
    )

