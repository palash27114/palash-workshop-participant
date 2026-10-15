#!/usr/bin/env bash
# Script: Run the Streamlit dashboard on the workshop EKS cluster, reading Gold from the S3 data lake
# Purpose: Host the dashboard in AWS without editing infra/ or bin/. It listens on the node's loopback
#          only (never on a public address) and is reached through a kubectl tunnel.
# Usage:   data/talent-etl/deploy/dashboard_deploy.sh      then open http://localhost:8502
# Image note: the cloud image ships numexpr/bottleneck built for numpy 1.x; they are upgraded at start.
# Prereq:  cloud_setup.sh + cloud_run.sh have run (bucket, node S3 access, Gold data in S3).

set -euo pipefail

JOB_DIR="$(cd "$(dirname "$0")/.." && pwd -P)"
PROJECT_ROOT="$(cd "$JOB_DIR/../.." && pwd -P)"
cd "$PROJECT_ROOT"
export PATH="$HOME/.local/bin:$PATH"

./bin/setup-participant.sh > /dev/null
source ENVIRONMENT.config
export AWS_REGION="${AWS_REGION:-ap-south-1}"
aws eks update-kubeconfig --region "$AWS_REGION" --name "coding-workshop-${PARTICIPANT_ID}" > /dev/null

NAME="talent-dashboard"
LOCAL_PORT="${LOCAL_PORT:-8502}"

echo "== Shipping dashboard + pipeline modules"
kubectl create configmap "${NAME}-lib" --from-file="$JOB_DIR/config.py" --from-file="$JOB_DIR/storage.py" \
    --from-file="$JOB_DIR/requirements.txt" --dry-run=client -o yaml | kubectl apply -f - > /dev/null
kubectl create configmap "${NAME}-app" --from-file="$JOB_DIR/dashboard/app.py" \
    --from-file="$JOB_DIR/dashboard/gold_data.py" --dry-run=client -o yaml | kubectl apply -f - > /dev/null

echo "== Deploying ${NAME}"
kubectl apply -f - << EOF > /dev/null
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${NAME}
  labels: {app.kubernetes.io/name: ${NAME}, app.kubernetes.io/part-of: coding-workshop}
spec:
  replicas: 1
  strategy: {type: Recreate}
  selector:
    matchLabels: {app.kubernetes.io/name: ${NAME}}
  template:
    metadata:
      labels: {app.kubernetes.io/name: ${NAME}}
      annotations:
        # Changing this forces a restart so new code is picked up.
        talent-etl/deployed-at: "$(date -u +%Y%m%dT%H%M%SZ)"
    spec:
      # Node network: uses the node role's data-lake permission (IMDS hop limit 1). The app binds to the
      # node's 127.0.0.1 only, so it is reachable through kubectl port-forward and nothing else.
      hostNetwork: true
      dnsPolicy: ClusterFirstWithHostNet
      containers:
        - name: dashboard
          image: jupyter/all-spark-notebook:latest
          imagePullPolicy: IfNotPresent
          command: ["sh", "-c"]
          args:
            - |
              set -e
              pip install --no-cache-dir --quiet -r /src/lib/requirements.txt streamlit==1.66.0 altair==6.3.0 "numexpr>=2.10" "bottleneck>=1.4"
              mkdir -p /tmp/app/dashboard && cp /src/lib/*.py /tmp/app/ && cp /src/dashboard/*.py /tmp/app/dashboard/
              cd /tmp/app && exec streamlit run dashboard/app.py --server.address 127.0.0.1 \
                --server.port 8501 --server.headless true --browser.gatherUsageStats false \
                --theme.primaryColor "#2a78d6"
          env:
            - {name: PYTHONDONTWRITEBYTECODE, value: "1"}
            - {name: IS_LOCAL, value: "false"}
            - {name: APP_ID, value: "${PARTICIPANT_ID}"}
            - {name: AWS_REGION, value: "${AWS_REGION}"}
            - {name: AWS_DEFAULT_REGION, value: "${AWS_REGION}"}
            - {name: LAKE_BUCKET, value: "coding-workshop-datalake-${PARTICIPANT_ID}"}
          readinessProbe:
            exec: {command: ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"]}
            initialDelaySeconds: 20
            periodSeconds: 10
          resources:
            requests: {cpu: 250m, memory: 512Mi}
            limits: {cpu: "1", memory: 1536Mi}
          volumeMounts:
            - {name: lib, mountPath: /src/lib}
            - {name: app, mountPath: /src/dashboard}
      volumes:
        - name: lib
          configMap: {name: ${NAME}-lib}
        - name: app
          configMap: {name: ${NAME}-app}
EOF

echo "== Waiting until the dashboard is ready (pip install takes a minute or two)"
kubectl rollout status "deploy/${NAME}" --timeout=600s

echo "== Opening tunnel http://localhost:${LOCAL_PORT} -> dashboard in EKS"
pkill -f "port-forward deploy/${NAME}" 2> /dev/null || true
nohup kubectl port-forward "deploy/${NAME}" "${LOCAL_PORT}:8501" > /tmp/talent-dashboard-tunnel.log 2>&1 &
sleep 3
curl -sf "http://localhost:${LOCAL_PORT}/_stcore/health" > /dev/null \
    && echo "Ready: http://localhost:${LOCAL_PORT}" \
    || { echo "Tunnel not answering yet; see /tmp/talent-dashboard-tunnel.log"; exit 1; }
