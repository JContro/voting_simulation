# ============================================================
# Voting simulation — lightweight Python scientific image
# ============================================================
FROM python:3.11-slim

# Create non-root user (uid/gid 1000 matches typical host defaults)
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --create-home appuser

WORKDIR /home/appuser/app

# Install runtime deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the package and entry point
COPY votesim/    votesim/
COPY run_demo.py run_sweep.py run_approval_sweep.py run_ordinal_demo.py .

# Output directory — mounted as a volume in docker-compose
RUN mkdir -p figures && chown -R appuser:appuser /home/appuser/app

USER appuser

ENTRYPOINT ["python"]