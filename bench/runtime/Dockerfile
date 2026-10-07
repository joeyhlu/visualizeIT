# Deferred: this file does not start or update Docker Desktop.
# Supply an immutable resolved base digest; bootstrap records pip freeze too.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 SAM2_BUILD_CUDA=0 XFORMERS_DISABLED=1 PYOPENGL_PLATFORM=egl
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgl1 libgl1-mesa-dri libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
COPY bench/runtime/requirements-quality.txt /requirements.txt
RUN python -m pip install pip==24.3.1 setuptools==75.3.0 wheel==0.44.0 && python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 && python -m pip install -r /requirements.txt
COPY bench /work/bench
WORKDIR /work
ENTRYPOINT ["python", "-B", "-m", "bench.quality_runner"]
