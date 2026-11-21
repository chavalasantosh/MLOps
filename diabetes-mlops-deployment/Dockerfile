FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    BENTOML_HOME=/app/bentoml BENTOML_DO_NOT_TRACK=True
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt && python -m pip check
RUN useradd --create-home --uid 10001 appuser && chown appuser:appuser /app
COPY --chown=appuser:appuser diabetes_model.bentomodel ./
COPY --chown=appuser:appuser model_manifest.json api_fixtures.json verify_artifact.py ./
USER appuser
RUN python verify_artifact.py && bentoml models import ./diabetes_model.bentomodel && rm ./diabetes_model.bentomodel
COPY --chown=appuser:appuser service.py ./
EXPOSE 3000
CMD ["bentoml", "serve", "service:DiabetesClassifier", "--host", "0.0.0.0", "--port", "3000"]
