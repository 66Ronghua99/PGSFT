FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 grader
WORKDIR /opt/livecodebench
RUN git clone --filter=blob:none https://github.com/LiveCodeBench/LiveCodeBench source \
    && git -C source checkout 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24 \
    && rm -rf source/.git \
    && pip install --no-cache-dir numpy==1.26.4 pandas==2.3.3 tqdm==4.67.1
ENV PYTHONPATH=/opt/livecodebench/source
COPY scripts/lcb_worker.py /opt/livecodebench/lcb_worker.py
USER grader
ENTRYPOINT ["python", "/opt/livecodebench/lcb_worker.py"]
