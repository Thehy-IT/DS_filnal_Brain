import shap

class VolumetricShapExplainer:
    def __init__(self, neural_network, background_reference_volumes):
        self.neural_network = neural_network
        self.shap_explainer = shap.DeepExplainer(
            model=self.neural_network,
            data=background_reference_volumes
        )

    def calculate_shap_attributions(self, target_volume_tensor):
        self.neural_network.eval()
        shap_attribution_values = self.shap_explainer.shap_values(target_volume_tensor)
        return shap_attribution_values
