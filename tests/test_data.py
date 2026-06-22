import numpy as np
from src.data.transforms import SkullStripping2Dd, IntensityNormalization2Dd

def test_intensity_normalization():
    normalization_transform = IntensityNormalization2Dd(keys=["image"])
    dummy_data_dictionary = {"image": np.array([[[1.0, 2.0], [3.0, 4.0]]])}
    processed_data = normalization_transform(dummy_data_dictionary)
    normalized_pixel_values = processed_data["image"]
    
    assert np.allclose(np.mean(normalized_pixel_values), 0.0)
    assert np.allclose(np.std(normalized_pixel_values), 1.0)

def test_skull_stripping():
    skull_stripping_transform = SkullStripping2Dd(keys=["image"])
    dummy_pixel_volume = np.zeros((1, 10, 10))
    dummy_pixel_volume[0, 3:7, 3:7] = 10.0
    dummy_data_dictionary = {"image": dummy_pixel_volume}
    
    processed_data = skull_stripping_transform(dummy_data_dictionary)
    stripped_pixel_values = processed_data["image"]
    
    assert np.all(stripped_pixel_values[0, 0:2, :] == 0.0)
    assert np.any(stripped_pixel_values > 0.0)
