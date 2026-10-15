#!/usr/bin/env bash
# Script: One-time AWS setup for running talent-etl on the workshop EKS cluster
# Purpose: Give the stock EKS environment what the stock Terraform does not (bucket, S3 access, path to Aurora)
#          without editing infra/ or bin/. Idempotent: safe to run again.
# Usage:   data/talent-etl/deploy/cloud_setup.sh
# Re-run:  after every ./bin/deploy-backend.sh, because Terraform resets Aurora's security groups (step 3).
#
# Prerequisites: EKS enabled and deployed (TF_VAR_aws_eks_enabled=true + ./bin/deploy-backend.sh).

set -euo pipefail

JOB_DIR="$(cd "$(dirname "$0")/.." && pwd -P)"
PROJECT_ROOT="$(cd "$JOB_DIR/../.." && pwd -P)"
cd "$PROJECT_ROOT"

# Fresh temporary credentials (the workshop script only writes ENVIRONMENT.config; nothing is modified).
./bin/setup-participant.sh > /dev/null
source ENVIRONMENT.config
export AWS_REGION="${AWS_REGION:-ap-south-1}"

APP_ID="$PARTICIPANT_ID"
BUCKET="coding-workshop-datalake-${APP_ID}"
CLUSTER="coding-workshop-${APP_ID}"
EKS_ROLE="coding-workshop-eks-${AWS_REGION}-${APP_ID}"
RDS_CLUSTER="coding-workshop-rds-${APP_ID}"

echo "== 1/4 Data lake bucket s3://${BUCKET}"
if aws s3api head-bucket --bucket "$BUCKET" 2> /dev/null; then
    echo "  exists"
else
    aws s3api create-bucket --bucket "$BUCKET" --region "$AWS_REGION" \
        --create-bucket-configuration "LocationConstraint=${AWS_REGION}" > /dev/null
    echo "  created"
fi
# Personal HR data: never public, always encrypted.
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
echo "  public access blocked, encryption on"

echo "== 2/4 Let EKS nodes read/write that bucket only (inline policy on ${EKS_ROLE})"
aws iam put-role-policy --role-name "$EKS_ROLE" --policy-name "coding-workshop-eks-datalake-${APP_ID}" \
    --policy-document "$(cat << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid": "ListDataLake", "Effect": "Allow", "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
     "Resource": "arn:aws:s3:::${BUCKET}"},
    {"Sid": "ReadWriteDataLake", "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
     "Resource": "arn:aws:s3:::${BUCKET}/*"}
  ]
}
EOF
)"
echo "  policy coding-workshop-eks-datalake-${APP_ID} in place"

echo "== 3/4 Network path EKS -> Aurora (add the EKS cluster security group to Aurora)"
CLUSTER_SG="$(aws eks describe-cluster --name "$CLUSTER" --query cluster.resourcesVpcConfig.clusterSecurityGroupId --output text)"
CURRENT_SGS="$(aws rds describe-db-clusters --db-cluster-identifier "$RDS_CLUSTER" \
    --query 'DBClusters[0].VpcSecurityGroups[].VpcSecurityGroupId' --output text)"
if grep -qw "$CLUSTER_SG" <<< "$CURRENT_SGS"; then
    echo "  already attached ($CLUSTER_SG)"
else
    # shellcheck disable=SC2086  # word-splitting the current group list is intended
    aws rds modify-db-cluster --db-cluster-identifier "$RDS_CLUSTER" --apply-immediately \
        --vpc-security-group-ids $CURRENT_SGS "$CLUSTER_SG" > /dev/null
    echo "  attached $CLUSTER_SG (kept: $CURRENT_SGS)"
fi

echo "== 4/4 Upload the sample dataset to the landing zone"
IS_LOCAL=false LAKE_BUCKET="$BUCKET" APP_ID="$APP_ID" \
    "$JOB_DIR/.venv/bin/python" "$JOB_DIR/seed_sources.py" --skip-postgres 2>&1 | grep -E "seed uploaded|ERROR" || true

echo ""
echo "Done. Next: data/talent-etl/deploy/cloud_run.sh"
