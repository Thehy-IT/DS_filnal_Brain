import streamlit as st

def generate_structured_clinical_report(estimated_area_cm2, centroid_physical_coordinates, model_confidence_ratio, predicted_class_name):
    report_text_content = f"""# BÁO CÁO ĐỊNH LƯỢNG LÂM SÀNG
## Phân Tích Không Gian và Phân Loại Tổn Thương Não
---
### Đánh Giá Định Lượng
- **Loại u dự đoán:** {predicted_class_name.upper()}
- **Diện tích lát cắt u ước tính:** {estimated_area_cm2:.2f} cm²
- **Tọa độ trọng tâm u (Image Space):** 
  - Trục ngang (X): {centroid_physical_coordinates[0]:.2f} mm
  - Trục đứng (Y): {centroid_physical_coordinates[1]:.2f} mm
- **Độ tin cậy chẩn đoán của AI:** {model_confidence_ratio * 100.0:.2f}%

### Phát Hiện Lâm Sàng & Ghi Chú
- **Đánh giá AI:** Phân tích cấu trúc cho thấy tổn thương giải phẫu phù hợp với nhóm {predicted_class_name}.
- **Phân bố không gian:** Các giá trị trọng tâm được đối chiếu với hệ trục gốc của lát cắt hiển thị.

### Khước Từ Trách Nhiệm Pháp Lý
*Lưu ý: Phân tích này là một công cụ hỗ trợ chẩn đoán dựa trên trí tuệ nhân tạo. Kết luận lâm sàng cuối cùng và các quyết định y khoa phải được xem xét và xác nhận bởi bác sĩ chẩn đoán hình ảnh hoặc bác sĩ ngoại thần kinh có chứng chỉ hành nghề.*
"""
    return report_text_content

def render_clinical_report_widget(estimated_area_cm2, centroid_physical_coordinates, model_confidence_ratio, predicted_class_name):
    report_document = generate_structured_clinical_report(estimated_area_cm2, centroid_physical_coordinates, model_confidence_ratio, predicted_class_name)
    st.markdown(report_document)
    st.download_button(
        label="Xuất Báo Cáo Lâm Sàng",
        data=report_document,
        file_name="bao_cao_lam_sang_u_nao.md",
        mime="text/markdown"
    )
