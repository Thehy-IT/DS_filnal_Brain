class MedicalSystemError(Exception):
    pass

class InvalidVolumetricFormatError(MedicalSystemError):
    pass

class DimensionMismatchError(MedicalSystemError):
    pass

class MissingSpatialMetadataError(MedicalSystemError):
    pass

class PACSConnectionError(MedicalSystemError):
    pass
