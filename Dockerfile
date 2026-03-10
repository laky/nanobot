# Stage 1: Build whisper.cpp with ROCm/HIP
FROM rocm/dev-ubuntu-22.04:5.7 AS whisper-builder

RUN apt-get update && apt-get install -y git cmake build-essential

WORKDIR /build
RUN git clone https://github.com/ggerganov/whisper.cpp.git && \
    cd whisper.cpp && \
    cmake -B build -DGGML_HIPBLAS=ON && \
    cmake --build build --config Release -j$(nproc)

# Download medium model
RUN mkdir -p /models && \
    apt-get install -y wget && \
    wget -O /models/ggml-medium.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin

# Stage 2: Application image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install ROCm runtime (minimal) and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg2 ffmpeg \
    && wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | gpg --dearmor > /etc/apt/trusted.gpg.d/rocm.gpg \
    && echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7 jammy main" > /etc/apt/sources.list.d/rocm.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends hip-runtime-amd rocblas hipblas \
    && rm -rf /var/lib/apt/lists/*

# Copy whisper.cpp binary, libraries, and model from builder
COPY --from=whisper-builder /build/whisper.cpp/build/bin/whisper-cli /usr/local/bin/whisper-cpp
COPY --from=whisper-builder /build/whisper.cpp/build/src/libwhisper.so* /usr/local/lib/
COPY --from=whisper-builder /build/whisper.cpp/build/ggml/src/libggml*.so* /usr/local/lib/
COPY --from=whisper-builder /models/ggml-medium.bin /app/models/ggml-medium.bin
RUN ldconfig

# Install Node.js 20 for the WhatsApp bridge
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates gnupg git && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get purge -y gnupg && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build WhatsApp bridge first (independent of Python code)
# Copy package.json separately for better npm cache
COPY bridge/package.json bridge/
WORKDIR /app/bridge
RUN npm install
COPY bridge/src/ src/
COPY bridge/tsconfig.json ./
RUN npm run build
WORKDIR /app

# Install Python dependencies (cached layer)
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p nanobot && touch nanobot/__init__.py && \
    uv pip install --system --no-cache . && \
    rm -rf nanobot

# Copy Python source and install
COPY nanobot/ nanobot/
RUN uv pip install --system --no-cache .

# Create config directory
RUN mkdir -p /root/.nanobot

# Gateway default port
EXPOSE 18790

ENTRYPOINT ["nanobot"]
CMD ["status"]
