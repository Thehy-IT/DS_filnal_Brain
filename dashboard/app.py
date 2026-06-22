import streamlit as st
import numpy as np
import os
import torch
import sys
import importlib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.utils import persist_uploaded_file, extract_nifti_data_matrix, apply_morphological_correction
import dashboard.components.viewer
importlib.reload(dashboard.components.viewer)
from dashboard.components.viewer import display_medical_slice_panels
from dashboard.components.report import render_clinical_report_widget
from src.models.classifier import initialize_2d_classifier_model
from src.inference.predictor import VolumetricPredictor
from src.data.slicer import MedicalVolumeSlicer

@st.cache_data
def cached_load_3d_volume(file_path):
    volume_slicer = MedicalVolumeSlicer()
    return volume_slicer.load_3d_volume(file_path)

st.set_page_config(page_title="Hệ Thống Phân Tích U Não MRI Lai 2D/3D", layout="wide")

st.title("Hệ Thống Chẩn Đoán Hình Ảnh & Phân Vùng U Não MRI")

uploaded_files = st.sidebar.file_uploader(
    "Tải lên Ảnh MRI (NIfTI .nii/.nii.gz, DICOM .dcm hoặc JPG/PNG)",
    type=["nii", "nii.gz", "jpg", "jpeg", "png", "dcm"],
    accept_multiple_files=True
)

temporary_file_path = None
is_3d = False

if "test_nifti" in st.query_params:
    temporary_file_path = "data/synthetic_volume.nii.gz"
    is_3d = True
    st.sidebar.success("Tải tập tin thử nghiệm NIfTI thành công.")
elif uploaded_files:
    if len(uploaded_files) > 1:
        selected_file = st.sidebar.selectbox("Chọn tập tin phân tích", [f.name for f in uploaded_files])
        uploaded_file_handle = [f for f in uploaded_files if f.name == selected_file][0]
    else:
        uploaded_file_handle = uploaded_files[0]
    temporary_file_path = persist_uploaded_file(uploaded_file_handle)
    is_3d = temporary_file_path.lower().endswith((".nii", ".nii.gz"))
    st.sidebar.success("Tải tập tin thành công.")

if temporary_file_path is not None:
    
    checkpoint_path = "data/reference/best_metric_model.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = initialize_2d_classifier_model()
    predictor = VolumetricPredictor(model, checkpoint_path, device)
    slicer = MedicalVolumeSlicer()
    
    mask_opacity = st.sidebar.slider("Độ Mờ Nhãn U (Mask Opacity)", 0.0, 1.0, 0.5)
    has_unet = getattr(predictor, "unet_model", None) is not None
    cutoff_label = "Ngưỡng Phân Vùng U-Net" if has_unet else "Độ Nhạy Chẩn Đoán (Grad-CAM Cutoff)"
    probability_cutoff = st.sidebar.slider(cutoff_label, 0.1, 0.9, 0.5)
    morphology_correction_mode = st.sidebar.selectbox("Hiệu Chỉnh Hình Thái Học", ["none", "dilation", "erosion"])
    correction_iterations = st.sidebar.slider("Bán Kính Hiệu Chỉnh", 1, 5, 1)
    
    label_names = ["glioma", "meningioma", "notumor", "pituitary"]
    
    if is_3d:
        volume = cached_load_3d_volume(temporary_file_path)
        
        anatomical_plane = st.sidebar.selectbox("Mặt Cắt Giải Phẫu", ["Axial", "Coronal", "Sagittal"])
        max_slices = slicer.get_slice_count(volume, anatomical_plane)
        slice_index = st.sidebar.slider("Lát Cắt (Slice)", 0, max_slices - 1, max_slices // 2)
        
        slice_np = slicer.extract_slice(volume, anatomical_plane, slice_index)
    else:
        image_object = extract_nifti_data_matrix(temporary_file_path)
        slice_np = np.array(image_object)
        
    activation_map_resized, confidence, predicted_class_index, uncertainty_map_resized = predictor.execute_inference_on_slice(
        slice_np,
        run_uncertainty=True,
        num_passes=5
    )
    
    segmentation_mask = (activation_map_resized >= probability_cutoff).astype(np.uint8)
    if morphology_correction_mode != "none":
        segmentation_mask = apply_morphological_correction(segmentation_mask, morphology_correction_mode, correction_iterations)
        
    single_pixel_area_cm2 = 0.05 * 0.05
    estimated_tumor_area = np.sum(segmentation_mask > 0) * single_pixel_area_cm2
    
    tumor_pixel_coordinates = np.argwhere(segmentation_mask > 0)
    if len(tumor_pixel_coordinates) > 0:
        centroid_pixel_location = np.mean(tumor_pixel_coordinates, axis=0)
        centroid_physical_coordinates = np.array([
            centroid_pixel_location[0] * 0.5,
            centroid_pixel_location[1] * 0.5
        ])
    else:
        centroid_physical_coordinates = np.array([0.0, 0.0])
        
    column_left, column_right = st.columns([3, 1])
    
    with column_left:
        st.subheader("Giao Diện Trực Quan Hóa Lát Cắt Giải Phẫu 2D/3D")
        display_medical_slice_panels(
            slice_image=slice_np,
            segmentation_mask=segmentation_mask,
            gradcam_map=activation_map_resized,
            uncertainty_map=uncertainty_map_resized,
            overlay_opacity=mask_opacity,
            is_unet_active=has_unet
        )
        
    with column_right:
        st.subheader("Báo Cáo Phân Tích")
        render_clinical_report_widget(
            estimated_area_cm2=estimated_tumor_area,
            centroid_physical_coordinates=centroid_physical_coordinates,
            model_confidence_ratio=confidence,
            predicted_class_name=label_names[predicted_class_index]
        )
else:
    st.info("Vui lòng tải lên ảnh MRI giải phẫu (NIfTI, DICOM hoặc ảnh thường) để bắt đầu phân tích.")
