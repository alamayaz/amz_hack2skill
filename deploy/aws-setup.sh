#!/usr/bin/env bash
# Rezaa AI — Launch an EC2 t3.medium instance with docker-compose support.
# Prerequisites: AWS CLI configured (`aws configure`), jq installed.
set -euo pipefail

###############################################################################
# Configuration
###############################################################################
INSTANCE_TYPE="t3.medium"
AMI_NAME_PATTERN="al2023-ami-2023*-x86_64"   # Amazon Linux 2023
KEY_NAME="rezaa-key"
KEY_FILE="rezaa-key.pem"
SG_NAME="rezaa-sg"
VOLUME_SIZE=20          # GB, gp3
REGION="${AWS_DEFAULT_REGION:-ap-south-1}"

echo "==> Region: $REGION"

###############################################################################
# 1. Key pair
###############################################################################
if aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &>/dev/null; then
    echo "==> Key pair '$KEY_NAME' already exists, reusing."
else
    echo "==> Creating key pair '$KEY_NAME'..."
    aws ec2 create-key-pair \
        --key-name "$KEY_NAME" \
        --query 'KeyMaterial' \
        --output text \
        --region "$REGION" > "$KEY_FILE"
    chmod 400 "$KEY_FILE"
    echo "    Saved to $KEY_FILE"
fi

###############################################################################
# 2. Security group
###############################################################################
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' --output text --region "$REGION")

if SG_ID=$(aws ec2 describe-security-groups --group-names "$SG_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text --region "$REGION" 2>/dev/null); then
    echo "==> Security group '$SG_NAME' already exists ($SG_ID), reusing."
else
    echo "==> Creating security group '$SG_NAME'..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "Rezaa AI - SSH, HTTP, HTTPS, 8000" \
        --vpc-id "$VPC_ID" \
        --query 'GroupId' --output text --region "$REGION")

    MY_IP=$(curl -s https://checkip.amazonaws.com)/32

    # SSH from your IP only
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --region "$REGION" \
        --protocol tcp --port 22 --cidr "$MY_IP"
    # HTTP
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --region "$REGION" \
        --protocol tcp --port 80 --cidr 0.0.0.0/0
    # HTTPS
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --region "$REGION" \
        --protocol tcp --port 443 --cidr 0.0.0.0/0
    # App port
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --region "$REGION" \
        --protocol tcp --port 8000 --cidr 0.0.0.0/0

    echo "    Created $SG_ID (SSH from $MY_IP, HTTP/HTTPS/8000 from anywhere)"
fi

###############################################################################
# 3. Find latest Amazon Linux 2023 AMI
###############################################################################
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=$AMI_NAME_PATTERN" "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text --region "$REGION")
echo "==> AMI: $AMI_ID"

###############################################################################
# 4. Launch instance
###############################################################################
echo "==> Launching $INSTANCE_TYPE instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=$VOLUME_SIZE,VolumeType=gp3}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=rezaa-ai}]" \
    --query 'Instances[0].InstanceId' --output text --region "$REGION")

echo "    Instance: $INSTANCE_ID — waiting for running state..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

###############################################################################
# 5. Elastic IP
###############################################################################
echo "==> Allocating Elastic IP..."
ALLOC_ID=$(aws ec2 allocate-address --domain vpc \
    --query 'AllocationId' --output text --region "$REGION")
ELASTIC_IP=$(aws ec2 describe-addresses --allocation-ids "$ALLOC_ID" \
    --query 'Addresses[0].PublicIp' --output text --region "$REGION")

aws ec2 associate-address --instance-id "$INSTANCE_ID" --allocation-id "$ALLOC_ID" --region "$REGION" >/dev/null

###############################################################################
# Done
###############################################################################
echo ""
echo "============================================"
echo "  Rezaa AI EC2 Instance Ready"
echo "============================================"
echo "  Instance ID : $INSTANCE_ID"
echo "  Elastic IP  : $ELASTIC_IP"
echo "  Key file    : $KEY_FILE"
echo ""
echo "  SSH:"
echo "    ssh -i $KEY_FILE ec2-user@$ELASTIC_IP"
echo ""
echo "  Next: copy deploy/instance-init.sh to the server and run it:"
echo "    scp -i $KEY_FILE deploy/instance-init.sh ec2-user@$ELASTIC_IP:~/"
echo "    ssh -i $KEY_FILE ec2-user@$ELASTIC_IP 'bash ~/instance-init.sh'"
echo ""
echo "  App will be at: http://$ELASTIC_IP:8000"
echo "============================================"
