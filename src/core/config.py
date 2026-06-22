class MedicalSystemConfig:
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001
    MAX_EPOCHS = 20
    IMAGE_SIZE = (224, 224)
    
    DATA_TRAINING_DIRECTORY = "data/Training"
    DATA_TESTING_DIRECTORY = "data/Testing"
    DATA_REFERENCE_DIRECTORY = "data/reference"
    
    LABEL_MAPPING = {
        "glioma": 0,
        "meningioma": 1,
        "notumor": 2,
        "pituitary": 3
    }
    
    ORTHANC_PACS_HOST = "localhost"
    ORTHANC_PACS_DICOM_PORT = 4242
    ORTHANC_PACS_HTTP_PORT = 8042
