#!/usr/bin/env bash
# Script: Put the talent-etl notebook + code into your JupyterHub workspace on EKS
# Purpose: The stock JupyterHub starts every user with an empty, temporary workspace and no S3 access.
#          This copies the notebook and modules in, installs the job's libraries, points the notebook at
#          the cloud data lake, and gives it your temporary workshop credentials.
# Usage:   1) kubectl port-forward svc/proxy-public 8080:80   2) sign in at http://localhost:8080
#          3) data/talent-etl/deploy/jupyter_sync.sh            (waits up to 10 minutes for the sign-in)
# Re-run:  when the workshop credentials expire (S3 "ExpiredToken" in the notebook) or the workspace restarts.

set -euo pipefail

JOB_DIR="$(cd "$(dirname "$0")/.." && pwd -P)"
PROJECT_ROOT="$(cd "$JOB_DIR/../.." && pwd -P)"
cd "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"

./bin/setup-participant.sh > /dev/null
source ENVIRONMENT.config
export AWS_REGION="${AWS_REGION:-ap-south-1}"
aws eks update-kubeconfig --region "$AWS_REGION" --name "coding-workshop-${PARTICIPANT_ID}" > /dev/null

echo "== Waiting for your JupyterHub workspace (sign in at http://localhost:8080)"
POD=""
for _ in $(seq 1 60); do
    POD="$(kubectl get pods -l component=singleuser-server \
        -o jsonpath='{range .items[?(@.status.phase=="Running")]}{.metadata.name}{"\n"}{end}' | head -1)"
    [ -n "$POD" ] && break
    sleep 10
done
[ -n "$POD" ] || { echo "ERROR: no running JupyterHub workspace after 10 minutes. Sign in first."; exit 1; }
echo "  workspace: $POD"

echo "== Copying notebook and code"
kubectl exec "$POD" -- mkdir -p /home/jovyan/talent-etl/notebooks /home/jovyan/.aws \
    /home/jovyan/.ipython/profile_default/startup
for f in "$JOB_DIR"/*.py "$JOB_DIR/requirements.txt"; do
    kubectl cp "$f" "$POD:/home/jovyan/talent-etl/$(basename "$f")"
done
kubectl cp "$JOB_DIR/notebooks/talent_insights.ipynb" "$POD:/home/jovyan/talent-etl/notebooks/talent_insights.ipynb"

echo "== Pointing notebooks at the cloud data lake"
# IPython runs this at every kernel start, so the pipeline config resolves to Amazon S3, not LocalStack.
kubectl exec -i "$POD" -- sh -c 'cat > /home/jovyan/.ipython/profile_default/startup/00-talent-etl.py' << EOF
import os
os.environ.setdefault("IS_LOCAL", "false")
os.environ.setdefault("AWS_REGION", "${AWS_REGION}")
os.environ.setdefault("AWS_DEFAULT_REGION", "${AWS_REGION}")
os.environ.setdefault("APP_ID", "${PARTICIPANT_ID}")
os.environ.setdefault("LAKE_BUCKET", "coding-workshop-datalake-${PARTICIPANT_ID}")
EOF

echo "== Installing temporary workshop credentials (they expire; re-run this script to refresh)"
# Sent over stdin so the keys never appear in a process list or in this script's output.
kubectl exec -i "$POD" -- sh -c 'umask 077; cat > /home/jovyan/.aws/credentials' << EOF
[default]
aws_access_key_id = ${AWS_ACCESS_KEY_ID}
aws_secret_access_key = ${AWS_SECRET_ACCESS_KEY}
aws_session_token = ${AWS_SESSION_TOKEN}
region = ${AWS_REGION}
EOF

echo "== Installing the job's libraries in the workspace (about a minute)"
kubectl exec "$POD" -- pip install --quiet --user -r /home/jovyan/talent-etl/requirements.txt 2>&1 | tail -2 || true

echo "== Checking the workspace can read the Gold layer"
kubectl exec "$POD" -- sh -c 'cd /home/jovyan/talent-etl && python - << "PY"
import os
for k, v in {"IS_LOCAL": "false", "AWS_REGION": "'"${AWS_REGION}"'", "LAKE_BUCKET": "coding-workshop-datalake-'"${PARTICIPANT_ID}"'"}.items():
    os.environ.setdefault(k, v)
from config import load_settings
from storage import Lake
lake = Lake(load_settings())
print("  gold objects:", len(lake.list_keys("gold/")), "in", lake.uri("gold/"))
PY'

echo ""
echo "Ready: http://localhost:8080 -> talent-etl/notebooks/talent_insights.ipynb -> Kernel > Restart & Run All"
