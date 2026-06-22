from monai.networks.nets import DenseNet121

def initialize_2d_classifier_model(input_channels_count=3, target_classes_count=4):
    return DenseNet121(
        spatial_dims=2,
        in_channels=input_channels_count,
        out_channels=target_classes_count
    )
