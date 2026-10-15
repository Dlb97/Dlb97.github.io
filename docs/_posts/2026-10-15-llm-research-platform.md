---
title: Building a LLM research platform with Amazon EKS, Skypilot and Flyte
date: 2026-10-15
description: A practical guide to building a platform that makes LLM training and post-training easier to run.
tags:
  - LLMs
  - Infrastructure
---

<h2 class="index-title">Index</h2>

**I. [Introduction](#introduction)**

1. Overview of how LLMs and Diffusion Models have shifted infrastructure requirements to large GPU clusters.
2. Explanation of the pain points faced by each user persona on these large GPU clusters (Researchers and Platform administrators).

**II. [Solution Overview](#solution-overview)**

1. Description on how Amazon EKS, skypilot and flyte work together to give researchers a simple way to spin up GPU clusters and move from experimentation to productive jobs at scale.
2. Architecture:
   1. Oview of solution architecture.
   2. How Amazon EKS and Karpenter provide a scalable backend for GPU workloads.
   3. How skypilot eases the creation of GPU clusters across both onprem and cloud based environments.
   4. How flyte help researchers define python based workflows.

**III. [Technical Implementation](#technical-implementation)**

1. Clone repository and deploy infrastructure as code (IAC).
2. How researchers can use predefined ML Templates to productionalize ML workloads.
3. Show how to easily spin up a GPU experimentaiton cluster with skypilot to iterate on the model development process.
4. Explain how to launch the model training job at scale using flyte.
5. Clean up of IAC.

<hr class="index-rule">

<h2 id="introduction" class="section-title">I. Intro & Motivation</h2>

The use of LLMs and Diffusion Models has shifted infrastructure requirements towards massive, distributed GPU clusters. Operating this clusters at scale introduces new operational bottlenecks and pain points across data pipelines, compute orchestration, and environment parity. Below is a breakdown of the pain points currently faced by the different personas:

### Researchers

- **Environment and tool drift:** Most companies use both on-premises and cloud environments to train and do inference with their ML models. These environments usually differ on the tools and frameworks leading to researchers wasting days on debugging rather than modeling when they want to port their jobs to another environment.
- **Research to Production Hand-Off:** Moving a model from research into production typically requires handing off experimental code to ML engineers. The heavy code adaptation, optimization, and refactoring needed to make the model production-ready adds significant friction and delays time-to-market.
- **GPU Availability:** Researchers frequently have to check if the required GPU models are available in their environment and how much compute capacity remains.
- **Distributed Scale Complexity:** When scaling up to large clusters, researchers must parallelize jobs, handle checkpointing to avoid losing progress when a GPU randomly fails, and deal with network communication bottlenecks.

### Platform Admins

- **Hybrid Data Governance Bottleneck:** Platform Admins struggle to sync and share public or team data across on-premises and cloud clusters while strictly maintaining cross-team privacy boundaries.
- **Multi-Tenant Quota Enforcement:** Enforcing fair GPU allocation across diverse architectures and environments is highly challenging for admins, yet necessary to stop single-user workloads from starving the rest of the organization.
- **Dynamic Scaling & Cost Optimization:** Admins must ensure workloads automatically scale out when needed and shrink immediately upon completion, eliminating expensive idle GPU compute waste.

This blog series explores practical solutions to these challenges. In **Part 1**, we'll build the foundation of a cloud-based research platform. This part covers infrastructure setup, developer-friendly tools for spinning up GPU clusters, and staightforward workflows for scaling code to production training. **Part 2** will focus on extending this baseline into a secure, multi-tenant platform with team-level data and resource access controls.

---

<h2 id="solution-overview" class="section-title">II. Solution Overview</h2>

The platform is built on four core pillars:

1. **Amazon Elastic Kubernetes Service (Amazon EKS) as the Control Plane:** Serves as the backend foundation, providing a standard Kubernetes distribution that ensures open-source tools can be reused across environments.
2. **SkyPilot for GPU Cluster Management:** Enables researchers to easily provision, manage, and tear down GPU clusters across any cloud provider or on-premises infrastructure using simple commands.
3. **Flyte for Workflow Orchestration:** Offers Python-native workflow definitions that integrate with Kubernetes. Combined with pre-built MLOps templates (provided in this post), Flyte reduces code adaptation and accelerates the research-to-production hand-off.
4. **Ray as the Distributed Execution Engine:** Powers both SkyPilot and Flyte under the hood. Whenever a SkyPilot cluster is spun up or a Flyte workflow is executed, a RayJob handles the underlying distributed compute.

Beyond the four core pillars, the platform relies on several key supporting components:

- **Karpenter:** Handles autoscaling by dynamically provisioning compute nodes on demand and terminating idle instances to optimize cost.
- **NVIDIA GPU Operator:** Automates GPU management, including driver installation, node labeling for GPU architectures, and health monitoring.
- **KubeRay:** Simplifies distributed compute management by automating the deployment, scaling, and lifecycle of Ray clusters on Kubernetes.
- **AWS EFS CSI Driver:** Enables persistent, shared storage by allowing Kubernetes pods to mount Amazon Elastic File System (EFS) volumes across nodes.

### How It All Comes Together: An End-to-End Walkthrough

To see these tools in action, consider a typical workflow: a researcher fine-tuning an LLM using Reinforcement Learning from Human Feedback (RLHF).

#### Phase 1: Cluster Provisioning & Autoscaling

1. **Job Request:** The researcher submits a YAML cluster definition via the sky CLI, which sends the request to the SkyPilot API server running on Kubernetes.
2. **Job Definition:** The SkyPilot API processes the request and creates a RayJob specifying the required resources.
3. **Dynamic Infrastructure Scaling:** The RayJob initializes a head pod. If resources are insufficient, the pod enters a Pending state. Karpenter detects the pending pod, provisions a matching EC2 node from its node pools, and scales out the cluster.
4. **Hardware Initialization:** Once the node is online, the NVIDIA GPU Operator automatically installs the required NVIDIA drivers and labels the node with its GPU type and capacity.
5. **Workload Scheduling:** Kubernetes schedules the RayJob head pod onto the new node. Worker pods go through this exact same automated provisioning process as the job scales out.

#### Phase 2: Experimentation & Model Iteration

6. **Rapid Prototyping:** Using the standardized MLOps template structure, the researcher writes the RLHF fine-tuning logic in PyTorch and configures the distributed strategy with Ray.
7. **Execution & Cleanup:** They run small-scale experimental jobs on the cluster. Once a promising candidate model is identified, the researcher tears down the ephemeral SkyPilot cluster to free up resources.

#### Phase 3: Production Workflow Execution

8. **Workflow Packaging:** To scale up the finalized job, the researcher populates the Flyte MLOps template with their Python classes.
9. **Production Orchestration:** Triggering the Flyte workflow spins up a new RayJob, with Karpenter seamlessly handling the underlying autoscaling once again.

---

<h2 id="technical-implementation" class="section-title">III. Technical Implementation</h2>

In this section, we will deploy the Infrastructure as Code (IaC) and walk through the MLOps template. Before running the IaC deployment, let's examine the network architecture:

![Network architecture]({{ '/assets/images/network-architecture.png' | relative_url }})

As highlighted earlier, the platform is hosted on Amazon EKS within a VPC spanning two Availability Zones (AZs) across four subnets:

- **Public Subnets:** Two subnets route internet traffic directly through an Internet Gateway and host NAT Gateways for outbound access. They also host an internet-facing Network Load Balancer (NLB), which acts as the main entry point for researchers interacting with SkyPilot and Flyte.
- **Private Subnets:** Two subnets host the Amazon EKS worker nodes, routing outbound internet traffic securely through the NAT Gateways in the public subnets.

**Traffic Ingress & Routing:** The Network Load Balancer targets a Kubernetes NGINX Ingress Controller running inside the cluster. While the NLB acts as the entry point, the NGINX Ingress handles application-level routing, directing incoming requests to their corresponding services (such as SkyPilot or Flyte) based on the request path and host headers.

**TLS & Access Configuration:** The NLB uses Server Name Indication (SNI) with two self-signed certificates managed via AWS Certificate Manager (ACM), eliminating the need to purchase a domain name (local `/etc/hosts` edits are required). To avoid the extra cost of a bastion host, the EKS cluster endpoint is configured for public and private access, secured via IAM permissions.

### 3.1 Deployment Steps

**1. Clone the Repository:** Clone the project repository and navigate into the IaC directory:

```bash
git clone <URL> ./research_platform
cd ./research_platform/Kubernetes-GPU-Cluster
```

*Note: The repository contains two directories: `/Kubernetes-GPU-Cluster` (IaC and `/llm-template` (MLOps template).*

**2. Create the SkyPilot Authentication Secret:** Generate a basic authentication secret and store it in AWS Secrets Manager. Be sure to replace `<username>`, `<password>`, and `<project_name>` with your own values:

```bash
WEB_USERNAME=<username>
WEB_PASSWORD=<password>
AUTH_STRING=$(htpasswd -nb "$WEB_USERNAME" "$WEB_PASSWORD")
aws secretsmanager create-secret \
  --name "<project_name>/skypilot-basic-auth" \
  --secret-string "$AUTH_STRING"
```

**3. Configure Terraform Variables:** Copy `envs/main/main.tfvars.example` to `envs/main/main.tfvars` and update the values. Ensure `project_name` matches the value used in Step 2:

```bash
cp envs/main/main.tfvars.example envs/main/main.tfvars
# Edit envs/main/main.tfvars with your configuration
```

**4. Initialize and Plan Infrastructure:** Initialize Terraform and generate an execution plan to preview the resources to be deployed:

```bash
terraform init
terraform plan -var-file=envs/main/main.tfvars
```

**5. Provision the Resources:** Apply the Terraform configuration to deploy the infrastructure:

```bash
terraform apply -var-file=envs/main/main.tfvars
```

**6. Retrieve the Load Balancer Details:** Fetch the DNS name for the deployed Load Balancer using its ARN (replace `<load_balancer_arn>` with your actual ARN):

```bash
aws elbv2 describe-load-balancers \
  --region us-east-1 \
  --load-balancer-arns "<load_balancer_arn>" | grep DNSName
```

Next, resolve the DNS name to retrieve the Load Balancer's public IP (replace `<DNSName>` with the output from above):

```bash
nslookup <DNSName>
```

**6. Map Local DNS Hosts:** Add two host entries to your local `/etc/hosts` file mapping to one of the resolved Load Balancer IP addresses. This routes traffic locally without needing a registered domain name:

```bash
sudo vim /etc/hosts
```

Add the following lines (replacing `<load_balancer_ip_address>` with your resolved IP):

```text
<load_balancer_ip_address> hpc_clusters.internal
<load_balancer_ip_address> workflows.internal
```

**7. Access the Web Interfaces:** Open the platform services in your browser:

- **SkyPilot UI:** https://hpc_clusters.internal
- **Flyte Console:** https://workflows.internal/console

*Note: Since the platform uses self-signed TLS certificates, your browser will display a security warning. You can safely bypass the warning to proceed to the interfaces.*

### 3.1 Working with the MLOs template

The MLOps template help researchers experiment and productionize LLM training without worrying about GPU management and scaling.

#### Repository Layout

The repository is organized under the `src/` directory as follows:

- `docker_image/` — Dockerfile used to build the container image for running Flyte workflows.
- `experimentation/` — SkyPilot configuration files to spin up interactive GPU clusters and iterate on model development.
- `model/` — The core model implementation (built with PyTorch).
- `pipeline/` — Flyte workflow definitions for end-to-end model training and evaluation.

#### Prerequisites

**1. Kubernetes Infrastructure**

Ensure you have access to a running Kubernetes cluster pre-configured with the required control plane components (SkyPilot and Flyte API servers, GPU node pools, and TLS certificates).

**2. Environment Setup (uv):** Install uv to manage Python packages:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and activate a virtual environment (Python 3.10 to 3.12 supported), then sync the project dependencies:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync
```

**3. Container Registry (Amazon ECR):** Set your target AWS region and account ID as environment variables:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=<123456789012>
```

Create the ECR repository to host your pipeline image:

```bash
aws ecr create-repository \
  --repository-name llm-pipeline \
  --region $AWS_REGION
```

Authenticate Docker with your ECR registry:

```bash
aws ecr get-login-password --region $AWS_REGION | \
docker login --username AWS --password-stdin \
$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
```

**4. Build and Push the Container Image:** Build the workflow container for x86_64 architectures and push it to ECR:

```bash
docker build --platform linux/amd64 -t \
$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/llm-pipeline:latest .
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/llm-pipeline:latest
```

**5. SkyPilot and Flyte Client Configuration:** Configure your local CLI clients to communicate with the cluster services.

**SkyPilot Configuration (`~/.sky/config.yaml`):**

```bash
cat > ~/.sky/config.yaml <<'EOF'
api_server:
  endpoint: https://hpc_clusters.internal
EOF
```

**Flyte Configuration (`~/.flyte/config.yaml`):**

Update `caCertFilePath` to point to the local path of your `combined-ca.crt` file (provisioned in Step 1).

```bash
cat > ~/.flyte/config.yaml <<'EOF'
admin:
  endpoint: dns:///workflows.internal:443
  insecure: false
  caCertFilePath: /Kubernetes-GPU-Cluster/modules/certificates/combined-ca.crt
EOF
```

**Trust Self-Signed Certificates:** Configure Python and CLI utilities (`sky`, `pyflyte`) to trust the self-signed certificates by updating your environment variables (replace `CERT_DIR` with your local path):

```bash
cat >> ~/.zshrc <<'EOF'
export CERT_DIR="/Kubernetes-GPU-Cluster/modules/certificates"
export REQUESTS_CA_BUNDLE="$CERT_DIR/combined-ca.crt"
export SSL_CERT_FILE="$CERT_DIR/combined-ca.crt"
EOF
```

```bash
source ~/.zshrc
```

#### Usage Guide

##### Phase 1: Interactive Model Development

During this phase, researchers iterate rapidly on model code using an ephemeral, dedicated GPU cluster managed by SkyPilot.

**1. Activate the Virtual Environment**

From the repository root:

```bash
source .venv/bin/activate
```

**2. Authenticate with the SkyPilot API Server**

Log in using the basic authentication credentials configured during infrastructure deployment:

```bash
sky api login -e "https://<SECRET_VALUE>@hpc_clusters.internal"
```

**3. Provision the Experimentation Cluster**

Provision a dedicated GPU cluster (defined in `hpc_cluster.yaml` as 4x NVIDIA T4 GPUs on a single node):

```bash
sky launch -c hpc -d src/experimentation/hpc_cluster.yaml
```

*Wait for SkyPilot and Karpenter to finish node provisioning.*

**4. Execute Experimental Training Runs**

Modify model code in `src/model/` and execute test runs directly on the remote cluster:

```bash
sky exec -c hpc src/experimentation/train.yaml
```

*(Note: `train.yaml` executes `src/model/model.py` across the cluster).*

##### Phase 2: Production Scale Training Pipeline

Once model code is validated, transition to Flyte for scalable, orchestrated training and evaluation.

**1. Configure GPU Resource Scaling**

Define the target GPU count in `src/pipeline/config.py`. This setting is consumed by `src/pipeline/utils.py` to configure the dynamic RayJob submitted to Kubernetes.

**2. Initialize the Flyte Project**

Create the Flyte project namespace.

*Note: Do not modify the project `--id`, as it binds directly to pre-configured Kubernetes namespaces, persistent volumes (PV), and PVC claims.*

```bash
flytectl create project \
  --id llm-pipeline \
  --name "LLM Pipeline" \
  --description "MLOps pipeline for production LLM training"
```

**3. Submit the Production Flyte Workflow**

Trigger remote workflow execution in the development domain:

```bash
pyflyte run --remote \
  --project llm-pipeline \
  --domain development \
  ./src/pipeline/pipeline.py pipeline
```

### 3.2 Deletion of Resources

When the platform is no longer needed, tear it down to avoid ongoing GPU costs:

Delete the content of the s3 bucket. Replace with your AWS Account Id:

```bash
aws s3 rm s3://<account_id>-flyte-terraform-data --recursive
aws s3 rm s3://<account_id>-flyte-terraform-data --recursive
```

```bash
terraform destroy
```

Terraform does not manage the `/etc/hosts` entries added in step 6 — remove `hpc_clusters.internal` and `workflows.internal` manually.

We covered the journey to set up the foundations for a cloud-based GPU platform on AWS EKS with SkyPilot and Flyte to executing dynamic, auto-scaling LLM training jobs.

**Want to see Part 2?** In the next post, I plan to dive into **multi-tenancy, team-level data segregation, compute isolated namespaces, and seamless SSO (Single Sign-On) integration.**

If you'd like me to write part 2, **let me know by liking this post or dropping your feedback in the comments below!** I'll be monitoring interest to decide whether to publish the next deep dive.
