#!/usr/bin/env bash
# Script: Run the Streamlit dashboard on the workshop EKS cluster, reading Gold from the S3 data lake
# Purpose: Host the dashboard in AWS without editing infra/ or bin/.
# Usage:   data/talent-etl/deploy/dashboard_deploy.sh            private: node loopback only, opened through a
#                                                                kubectl tunnel at http://localhost:8502
#          data/talent-etl/deploy/dashboard_deploy.sh --public   public: AWS load balancer URL, open to anyone
#                                                                (demo; set DASHBOARD_PASSWORD to add a password screen)
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
PUBLIC=false
[ "${1:-}" = "--public" ] && PUBLIC=true
BIND="127.0.0.1"
AUTH_ENV=""
if [ "$PUBLIC" = true ]; then
    BIND="0.0.0.0"
    # Open demo by default. Set DASHBOARD_PASSWORD before running to add the app's password screen.
    if [ -n "${DASHBOARD_PASSWORD:-}" ]; then
        printf '%s' "$DASHBOARD_PASSWORD" | kubectl create secret generic "${NAME}-auth" \
            --from-file=DASHBOARD_PASSWORD=/dev/stdin --dry-run=client -o yaml | kubectl apply -f - > /dev/null
        AUTH_ENV="          envFrom:
            - secretRef: {name: ${NAME}-auth}"
    fi
fi

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
      # Node network: uses the node role's data-lake permission (IMDS hop limit 1). Private mode binds to
      # the node's 127.0.0.1 (kubectl port-forward only); public mode binds 0.0.0.0 behind the load balancer.
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
              cd /tmp/app && exec streamlit run dashboard/app.py --server.address ${BIND} \
                --server.port 8501 --server.headless true --browser.gatherUsageStats false \
                --theme.primaryColor "#2a78d6"
          env:
            - {name: PYTHONDONTWRITEBYTECODE, value: "1"}
            - {name: IS_LOCAL, value: "false"}
            - {name: APP_ID, value: "${PARTICIPANT_ID}"}
            - {name: AWS_REGION, value: "${AWS_REGION}"}
            - {name: AWS_DEFAULT_REGION, value: "${AWS_REGION}"}
            - {name: LAKE_BUCKET, value: "coding-workshop-datalake-${PARTICIPANT_ID}"}
${AUTH_ENV}
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

if [ "$PUBLIC" = true ]; then
    echo "== Public load balancer"
    SUBNETS="$(aws eks describe-cluster --name "coding-workshop-${PARTICIPANT_ID}" \
        --query 'cluster.resourcesVpcConfig.subnetIds' --output text | tr '\t' ',')"
    kubectl apply -f - << SVC > /dev/null
apiVersion: v1
kind: Service
metadata:
  name: ${NAME}
  labels: {app.kubernetes.io/name: ${NAME}, app.kubernetes.io/part-of: coding-workshop}
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-subnets: "${SUBNETS}"
spec:
  type: LoadBalancer
  selector: {app.kubernetes.io/name: ${NAME}}
  ports:
    - {name: http, port: 80, targetPort: 8501, protocol: TCP}
SVC
    HOST=""
    for _ in $(seq 1 60); do
        HOST="$(kubectl get svc "$NAME" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')"
        [ -n "$HOST" ] && break
        sleep 5
    done
    [ -n "$HOST" ] || { echo "ERROR: no load balancer address yet"; kubectl describe svc "$NAME" | tail -8; exit 1; }
    echo "  waiting for http://${HOST} to answer (DNS + health checks take a few minutes)"
    for _ in $(seq 1 60); do
        curl -sf -m 5 "http://${HOST}/_stcore/health" > /dev/null && break
        sleep 10
    done
    curl -sf -m 5 "http://${HOST}/_stcore/health" > /dev/null || { echo "Not answering yet: http://${HOST}"; exit 1; }
    echo ""
    echo "Public URL: http://${HOST}"
    exit 0
fi

# Private mode: make sure no public load balancer is left over from an earlier --public deploy.
kubectl delete svc "$NAME" --ignore-not-found > /dev/null
echo "== Opening tunnel http://localhost:${LOCAL_PORT} -> dashboard in EKS"
pkill -f "port-forward deploy/${NAME}" 2> /dev/null || true
nohup kubectl port-forward "deploy/${NAME}" "${LOCAL_PORT}:8501" > /tmp/talent-dashboard-tunnel.log 2>&1 &
sleep 3
curl -sf "http://localhost:${LOCAL_PORT}/_stcore/health" > /dev/null \
    && echo "Ready: http://localhost:${LOCAL_PORT}" \
    || { echo "Tunnel not answering yet; see /tmp/talent-dashboard-tunnel.log"; exit 1; }
