import streamlit as st
import requests
import io
import base64
import os
from PIL import Image

# Config
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")

st.set_page_config(
    page_title="BrainTumorAI - Phân tích MRI",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 BrainTumorAI: Hệ thống chẩn đoán khối u não")
st.markdown("""
Hệ thống sử dụng **Vision Transformer (ViT)** để phát hiện và phân loại khối u não từ ảnh MRI.
Đồng thời, **Explainable AI (Grad-CAM)** được tích hợp để giải thích vùng tập trung của mô hình.
""")

st.sidebar.header("Tải ảnh MRI")
uploaded_file = st.sidebar.file_uploader("Chọn một ảnh MRI (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Original Image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Ảnh MRI gốc")
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh MRI tải lên", use_column_width=True)
        
    with col2:
        st.subheader("Kết quả phân tích")
        
        # Call API button
        if st.button("Phân tích ảnh"):
            with st.spinner('Đang xử lý và dự đoán...'):
                try:
                    # Prepare file to send
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(API_URL, files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        class_name = result["class_name"]
                        confidence = result["confidence"]
                        heatmap_b64 = result["heatmap_base64"]
                        
                        st.success("Hoàn tất phân tích!")
                        
                        # Display Results
                        st.markdown(f"### 🩺 Chẩn đoán: **{class_name}**")
                        st.progress(float(confidence))
                        st.write(f"**Độ tin cậy:** {confidence * 100:.2f}%")
                        
                        if heatmap_b64:
                            st.subheader("Bản đồ giải thích (Grad-CAM)")
                            st.markdown("Vùng màu đỏ/cam là khu vực AI tập trung nhiều nhất để đưa ra quyết định.")
                            heatmap_bytes = base64.b64decode(heatmap_b64)
                            st.image(heatmap_bytes, caption="Heatmap Gradient-CAM", use_column_width=True)
                        else:
                            st.warning("Không thể sinh Heatmap cho ảnh này.")
                            
                    elif response.status_code == 500:
                        st.error("Lỗi từ server: Model chưa được huấn luyện hoặc load thất bại.")
                    else:
                        st.error(f"Lỗi {response.status_code}: {response.text}")
                
                except requests.exceptions.ConnectionError:
                    st.error("Không thể kết nối đến Backend Server. Hãy chắc chắn rằng bạn đang chạy `uvicorn app.backend.main:app`.")
else:
    st.info("Vui lòng tải lên một ảnh MRI từ sidebar để bắt đầu phân tích.")
