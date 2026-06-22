from monai.networks.nets import UNet

def initialize_2d_unet_model(input_channels_count=3, target_classes_count=4):
    return UNet(
        spatial_dims=2,
        in_channels=input_channels_count,
        out_channels=target_classes_count,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    )
