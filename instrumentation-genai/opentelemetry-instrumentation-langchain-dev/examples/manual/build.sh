#!/usr/bin/env bash
# Automated Docker image build, verify, push, and deploy script
# Can be used standalone or called by Packer
#
# Usage:
#   AUTO_PUSH=true AUTO_DEPLOY=true bash build.sh
#
# Environment Variables:
#   AUTO_PUSH      - Push to Docker Hub (default: true)
#   AUTO_DEPLOY    - Deploy to Kubernetes (default: false)
#   K8S_MANIFEST   - Path to Kubernetes manifest file
#   K8S_NAMESPACE  - Kubernetes namespace for deployment

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
DOCKER_USERNAME="${DOCKER_USERNAME:-admehra621}"
IMAGE_NAME="${IMAGE_NAME:-lgchn-dev-py39-compat}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
BUILD_CONTEXT="${BUILD_CONTEXT:-../../../..}"
DOCKERFILE="instrumentation-genai/opentelemetry-instrumentation-langchain-dev/examples/manual/Dockerfile"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"
# Determine the platform for the local verification build; fall back to linux/amd64 if detection fails
HOST_PLATFORM="${HOST_PLATFORM:-$(docker info --format '{{.OSType}}/{{.Architecture}}' 2>/dev/null || echo linux/amd64)}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

FULL_IMAGE_NAME="${DOCKER_USERNAME}/${IMAGE_NAME}"

# Kubernetes deployment configuration
K8S_MANIFEST="${K8S_MANIFEST:-instrumentation-genai/opentelemetry-instrumentation-langchain-dev/examples/manual/cronjob-py39.yaml}"
K8S_NAMESPACE="${K8S_NAMESPACE:-o11y-4-ai-admehra}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo "======================================================================"
    echo "$1"
    echo "======================================================================"
}

normalize_platform() {
    case "$1" in
        linux/x86_64) echo "linux/amd64" ;;
        linux/aarch64) echo "linux/arm64" ;;
        *) echo "$1" ;;
    esac
}

# Main script
main() {
    HOST_PLATFORM="$(normalize_platform "${HOST_PLATFORM}")"

    print_header "Docker Image Build Automation"
    echo "Image: ${FULL_IMAGE_NAME}:${IMAGE_TAG}"
    echo "Build Context: ${BUILD_CONTEXT}"
    echo "Dockerfile: ${DOCKERFILE}"
    echo ""

    # Step 1: Build
    log_info "Building Docker image for local verification (${HOST_PLATFORM})..."
    cd "${BUILD_CONTEXT}" || exit 1
    
    docker buildx build \
        --platform "${HOST_PLATFORM}" \
        -f "${DOCKERFILE}" \
        -t "${FULL_IMAGE_NAME}:${IMAGE_TAG}" \
        -t "${FULL_IMAGE_NAME}:${TIMESTAMP}" \
        -t "${FULL_IMAGE_NAME}:py39" \
        --load \
        .
    
    log_info "✓ Build completed"
    echo ""

    # Step 2: Verify
    log_info "Verifying image..."
    
    echo "  Image details:"
    docker images "${FULL_IMAGE_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}" | head -4
    
    echo ""
    echo "  Checking installed packages:"
    docker run --rm "${FULL_IMAGE_NAME}:${IMAGE_TAG}" pip list | grep -E "opentelemetry|langchain" | head -10
    
    echo ""
    echo "  Testing Python imports (basic verification):"
    docker run --rm \
        --entrypoint /bin/bash \
        "${FULL_IMAGE_NAME}:${IMAGE_TAG}" \
        -c "python -c 'import sys; print(f\"Python {sys.version}\"); import opentelemetry; import langchain; print(\"✓ Core imports successful\")'"
    
    log_info "✓ Verification completed"
    log_info "Note: Full application testing requires OPENAI_API_KEY environment variable"
    echo ""

    # Step 3: Push (optional)
    if [ "${AUTO_PUSH:-true}" = "true" ]; then
        log_info "Preparing to build and push multi-architecture image (${PLATFORMS})..."
        
        log_info "Pushing images to Docker Hub..."
        
        # Check if logged in
        if ! docker info 2>/dev/null | grep -q "Username:"; then
            log_warn "Not logged into Docker Hub. Attempting login..."
            docker login || {
                log_error "Docker login failed. Set DOCKER_PASSWORD env var or run 'docker login'"
                exit 1
            }
        fi
        
        docker buildx build \
            --platform "${PLATFORMS}" \
            -f "${DOCKERFILE}" \
            -t "${FULL_IMAGE_NAME}:${IMAGE_TAG}" \
            -t "${FULL_IMAGE_NAME}:${TIMESTAMP}" \
            -t "${FULL_IMAGE_NAME}:py39" \
            --push \
            .
        
        log_info "✓ Push completed"
        echo ""
        echo "Images available at:"
        echo "  - ${FULL_IMAGE_NAME}:${IMAGE_TAG}"
        echo "  - ${FULL_IMAGE_NAME}:${TIMESTAMP}"
        echo "  - ${FULL_IMAGE_NAME}:py39"
        echo ""
        echo "View on Docker Hub: https://hub.docker.com/r/${DOCKER_USERNAME}/${IMAGE_NAME}"
    else
        log_info "Skipping push (AUTO_PUSH=false)"
    fi

    # Step 4: Deploy to Kubernetes (optional)
    if [ "${AUTO_DEPLOY:-false}" = "true" ]; then
        log_info "Deploying to Kubernetes cluster..."
        
        # Check if kubectl is available
        if ! command -v kubectl &> /dev/null; then
            log_error "kubectl not found. Please install kubectl to deploy to Kubernetes."
            exit 1
        fi
        
        # Check if the manifest file exists
        if [ ! -f "${BUILD_CONTEXT}/${K8S_MANIFEST}" ]; then
            log_error "Kubernetes manifest not found: ${K8S_MANIFEST}"
            exit 1
        fi
        
        log_info "Applying manifest: ${K8S_MANIFEST}"
        log_info "Namespace: ${K8S_NAMESPACE}"
        
        cd "${BUILD_CONTEXT}" || exit 1
        
        if kubectl apply -f "${K8S_MANIFEST}" -n "${K8S_NAMESPACE}"; then
            log_info "✓ Deployment completed successfully"
            echo ""
            echo "CronJob deployed to namespace: ${K8S_NAMESPACE}"
            echo "Check status with: kubectl get cronjob -n ${K8S_NAMESPACE}"
        else
            log_error "Deployment failed. Check kubectl connectivity and credentials."
            exit 1
        fi
        echo ""
    else
        log_info "Skipping Kubernetes deployment (AUTO_DEPLOY=false)"
    fi

    print_header "Build Complete!"
}

# Run main function
main "$@"
