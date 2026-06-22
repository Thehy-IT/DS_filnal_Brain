import os
import requests
from pynetdicom import AE, evt, AllStoragePresentationContexts

class DicomReceiver:
    def __init__(self, output_directory="data/raw", api_endpoint="http://localhost:8000/predict"):
        self.output_directory = output_directory
        self.api_endpoint = api_endpoint
        os.makedirs(self.output_directory, exist_ok=True)
        
        self.application_entity = AE(ae_title="AI_LISTENER")
        self.application_entity.supported_contexts = AllStoragePresentationContexts

    def handle_store_event(self, event):
        dataset = event.dataset
        dataset.file_meta = event.file_meta
        
        file_name = f"{dataset.SOPInstanceUID}.dcm"
        destination_path = os.path.join(self.output_directory, file_name)
        
        dataset.save_as(destination_path, write_like_original=False)
        
        try:
            with open(destination_path, "rb") as file_handle:
                requests.post(self.api_endpoint, files={"file": file_handle})
        except requests.exceptions.RequestException:
            pass
            
        return 0x0000

    def start_receiver_service(self, host_address="0.0.0.0", port_number=104):
        handlers = [(evt.EVT_C_STORE, self.handle_store_event)]
        self.application_entity.start_server((host_address, port_number), block=True, evt_handlers=handlers)

if __name__ == "__main__":
    receiver = DicomReceiver()
    receiver.start_receiver_service()
