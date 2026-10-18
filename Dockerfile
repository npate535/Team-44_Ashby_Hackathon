FROM python:3.13-slim
LABEL sepia.app=sepia

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATHQUEUE_SITE=local PATHQUEUE_ARTIFACTS_ROOT=/data \
    PATH=/opt/cursor/.local/bin:$PATH HOME=/home/app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gosu \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
# The existing evidence interpreter uses the official headless Cursor CLI.
RUN curl -fsSL https://cursor.com/install -o /tmp/install-cursor.sh \
    && HOME=/opt/cursor bash /tmp/install-cursor.sh \
    && rm /tmp/install-cursor.sh
WORKDIR /app
COPY deploy/requirements.txt /app/deploy/requirements.txt
RUN pip install --no-cache-dir -r /app/deploy/requirements.txt
COPY backend /app/backend
COPY core /app/core
COPY agent /app/agent
COPY config /app/config
COPY fixtures /app/fixtures
COPY docs/RESULTS.md /app/docs/RESULTS.md
COPY team44-triage-frontend/app /app/team44-triage-frontend/app
COPY deploy/entrypoint.sh /app/deploy/entrypoint.sh
COPY deploy/seed/ /seed/
ARG RELEASE_SHA=local
ENV PATHQUEUE_RELEASE_SHA=$RELEASE_SHA
RUN chmod 0755 /app/deploy/entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["/app/deploy/entrypoint.sh"]
