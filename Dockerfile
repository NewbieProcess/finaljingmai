FROM python:3.12

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
        ffmpeg \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY Main.py ./
COPY EyeDetect.keras ./
COPY EyeAnalysis.keras ./

ENV TF_ENABLE_ONEDNN_OPTS=0 \
    TF_CPP_MIN_LOG_LEVEL=2

ENV PORT=8501
EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "Main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
