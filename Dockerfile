ARG FOLIOLEAF_DEPS_IMAGE=python:3.11-slim
FROM ${FOLIOLEAF_DEPS_IMAGE} AS dependencies
USER root
WORKDIR /build/folioleaf
COPY requirements.txt .
RUN python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt && python -m pip check

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 FOLIOLEAF_DATA=/var/lib/folioleaf
WORKDIR /opt/folioleaf
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY folioleaf ./folioleaf
COPY static ./static
RUN useradd -U -u 11519 folioleaf && install -d -o folioleaf -g folioleaf /var/lib/folioleaf
USER folioleaf
EXPOSE 8000
CMD ["python", "-m", "folioleaf"]
