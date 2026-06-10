# Container image for the modern web app.
#
# Build from the REPOSITORY ROOT so the model code and checkpoints are in the
# build context (the engine imports src/model_training/* and reads
# saved_models/separation_models/*):
#
#   docker build -f src/application/modern_web/Dockerfile -t snc-web .
#   docker run --rm -p 8000:8000 snc-web
#
# A trained checkpoint (separator_unet_film_multi_*.h5 with a matching
# *_classes.json) must exist under saved_models/separation_models/ at build
# time, since those files are gitignored. For a smaller image, add a
# .dockerignore that excludes data/, venv/ and notebooks/.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY src/application/modern_web/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PORT=8000
EXPOSE 8000
WORKDIR /app/src/application/modern_web
CMD ["python", "launch.py", "--host", "0.0.0.0", "--port", "8000"]
