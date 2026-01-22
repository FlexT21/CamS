FROM python:3.10-slim-bullseye

# UV environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=.

# Set working directory
WORKDIR /workspace/apps/server

# Update packages and build dependencies
RUN apt-get -y update
RUN apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    python3-distutils \
    python3-setuptools \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS

# Install UV from prebuilt image
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

ENV PATH="/workspace/apps/server/.venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=./uv.lock,target=uv.lock \
    --mount=type=bind,source=./pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

# Copy package files
COPY ./pyproject.toml \
    ./uv.lock \
    ./

# Copy user data
COPY ./users/ ./users

# Define volume for user data
VOLUME [ "/workspace/apps/server/users" ]

# Copy source code
COPY ./src ./src

# Sync and compile source code
RUN --mount=type=cache,target=/root/.cache/uv \ 
    uv sync

CMD [ "python", "-m", "src.main" ]
