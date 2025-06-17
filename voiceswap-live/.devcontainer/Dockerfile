FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y git ffmpeg build-essential
RUN pip install --no-cache-dir torch==2.3.0+cu124 torchvision --index-url https://download.pytorch.org/whl/cu124
RUN git clone https://github.com/w-okada/voice-changer.git /opt/voice-changer && \
    cd /opt/voice-changer && ./setup.sh
ENV PATH="/opt/voice-changer:${PATH}"
