import torch

class AttentionMapExtractor:
    def __init__(self, neural_network):
        self.neural_network = neural_network
        self.extracted_attention_tensors = None

    def capture_attention_matrix(self, module_instance, input_tensors, output_tensors):
        self.extracted_attention_tensors = output_tensors

    def register_extraction_hook(self, network_layer_identifier):
        named_modules_dictionary = dict(self.neural_network.named_modules())
        target_network_layer = named_modules_dictionary[network_layer_identifier]
        return target_network_layer.register_forward_hook(self.capture_attention_matrix)

    def extract_self_attention(self, input_volume_tensor):
        self.neural_network.eval()
        with torch.no_grad():
            _ = self.neural_network(input_volume_tensor)
        return self.extracted_attention_tensors
