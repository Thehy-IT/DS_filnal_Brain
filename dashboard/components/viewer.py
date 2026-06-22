import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def display_medical_slice_panels(slice_image, segmentation_mask=None, gradcam_map=None, uncertainty_map=None, overlay_opacity=0.5, is_unet_active=False):
    column_left, column_center, column_right = st.columns(3)
    
    with column_left:
        st.write("**Ảnh Gốc & Vùng Nghi Ngờ U**")
        fig_orig, ax_orig = plt.subplots(figsize=(5, 5))
        ax_orig.imshow(slice_image, cmap="gray")
        if segmentation_mask is not None and np.sum(segmentation_mask) > 0:
            masked_overlay = np.ma.masked_where(segmentation_mask == 0, segmentation_mask)
            ax_orig.imshow(masked_overlay, cmap="Reds", alpha=overlay_opacity, vmin=0, vmax=1)
        ax_orig.axis("off")
        st.pyplot(fig_orig)
        plt.close(fig_orig)
        
    with column_center:
        if is_unet_active:
            st.write("**Bản Đồ Xác Suất U-Net**")
        else:
            st.write("**Bản Đồ Kích Hoạt Grad-CAM**")
        fig_cam, ax_cam = plt.subplots(figsize=(5, 5))
        if gradcam_map is not None and np.sum(gradcam_map) > 0:
            ax_cam.imshow(slice_image, cmap="gray")
            masked_cam = np.ma.masked_where(gradcam_map < 0.1, gradcam_map)
            ax_cam.imshow(masked_cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        else:
            ax_cam.imshow(slice_image, cmap="gray")
        ax_cam.axis("off")
        st.pyplot(fig_cam)
        plt.close(fig_cam)
        
    with column_right:
        st.write("**Bản Đồ Độ Bất Định (Uncertainty)**")
        fig_unc, ax_unc = plt.subplots(figsize=(5, 5))
        if uncertainty_map is not None and np.sum(uncertainty_map) > 0:
            ax_unc.imshow(slice_image, cmap="gray")
            masked_unc = np.ma.masked_where(uncertainty_map < 0.1, uncertainty_map)
            ax_unc.imshow(masked_unc, cmap="hot", alpha=0.5, vmin=0, vmax=1)
        else:
            ax_unc.imshow(slice_image, cmap="gray")
        ax_unc.axis("off")
        st.pyplot(fig_unc)
        plt.close(fig_unc)
