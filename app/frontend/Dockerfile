FROM python:3.9-slim

WORKDIR /app

# Copy requirements và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn
COPY . /app/

# Expose port của Streamlit
EXPOSE 8501

# Khởi chạy ứng dụng Streamlit
CMD ["streamlit", "run", "app/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
