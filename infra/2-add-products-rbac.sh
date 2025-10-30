#!/bin/bash
echo "======================================================================"
echo "Update RBAC Permissions for Azure AI Search Index"
echo "======================================================================"

# Exit shell immediately if command exits with non-zero status
set -e

# Check if user is logged into Azure
echo "Checking Azure login status..."
if ! az account show &> /dev/null; then
    echo "Error: You are not logged into Azure. Please run 'az login' first."
    exit 1
fi
echo "✓ Azure login confirmed."
echo ""

# Find the repository root by looking for .git directory
# Start from the script's directory and go up the directory tree
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="$SCRIPT_DIR"

# Go up the directory tree looking for .git directory (primary indicator)
while [ "$REPO_ROOT" != "/" ] && [ "$REPO_ROOT" != "" ]; do
    if [ -d "$REPO_ROOT/.git" ]; then
        break
    fi
    REPO_ROOT=$(dirname "$REPO_ROOT")
done

# If .git not found, just go up one level from scripts directory
if [ "$REPO_ROOT" == "/" ] || [ "$REPO_ROOT" == "" ]; then
    REPO_ROOT=$(dirname "$SCRIPT_DIR")
fi

# Load variables from .env file into your shell
# First try the repo root, then try current directory
if [ -f "$REPO_ROOT/.env" ]; then
    source "$REPO_ROOT/.env"
    echo "Loaded environment from: $REPO_ROOT/.env"
elif [ -f .env ]; then
    source .env
    echo "Loaded environment from: $(pwd)/.env"
else
    echo ".env file not found!"
    echo "Looked in:"
    echo "  - $REPO_ROOT/.env"
    echo "  - $(pwd)/.env"
    exit 1
fi

# -------------- Get core environment variables -------------------------
# Get subscription ID from current Azure login
AZURE_SUBSCRIPTION_ID=$(az account show --query id -o tsv)
if [ -z "$AZURE_SUBSCRIPTION_ID" ]; then
    echo "Error: Could not retrieve subscription ID from Azure login."
    exit 1
fi
echo "Using subscription: $AZURE_SUBSCRIPTION_ID"

# Check if resource group is provided in .env, otherwise prompt for it
if [ -z "$AZURE_RESOURCE_GROUP" ]; then
    echo "Error: AZURE_RESOURCE_GROUP is not set in .env file."
    exit 1
fi

echo "Using resource group: $AZURE_RESOURCE_GROUP"

# Get principal id from authenticated account
PRINCIPAL_ID=$(az ad signed-in-user show --query id -o tsv)
if [ -z "$PRINCIPAL_ID" ]; then
    echo "Error: Could not retrieve principal ID from Azure login."
    exit 1
fi
echo "Using principal ID: $PRINCIPAL_ID"

echo ""
echo "All required environment variables are set. Proceeding..."
echo ""

# -------------- Create any additional RBAC roles required -------------------------
echo "Assigning Azure RBAC roles..."
echo ""

# Search Index Data Contributor
# Grants full access to Azure Cognitive Search index data.
echo "Assigning 'Search Index Data Contributor' role..."
az role assignment create \
        --role "8ebe5a00-799e-43f5-93ac-243d3dce84a7" \
        --assignee-object-id "${PRINCIPAL_ID}" \
        --scope /subscriptions/"${AZURE_SUBSCRIPTION_ID}"/resourceGroups/"${AZURE_RESOURCE_GROUP}" \
        --assignee-principal-type 'User' \
        --only-show-errors 2>/dev/null || echo "  (Role may already be assigned)"

# Search Index Data Reader
# Grants read access to Azure Cognitive Search index data.
echo "Assigning 'Search Index Data Reader' role..."
az role assignment create \
        --role "1407120a-92aa-4202-b7e9-c0e197c71c8f" \
        --assignee-object-id "${PRINCIPAL_ID}" \
        --scope /subscriptions/"${AZURE_SUBSCRIPTION_ID}"/resourceGroups/"${AZURE_RESOURCE_GROUP}" \
        --assignee-principal-type 'User' \
        --only-show-errors 2>/dev/null || echo "  (Role may already be assigned)"

# Cognitive Services OpenAI User
# Read access to view files, models, deployments. The ability to create completion and embedding calls.
echo "Assigning 'Cognitive Services OpenAI User' role..."
az role assignment create \
        --role "5e0bd9bd-7b93-4f28-af87-19fc36ad61bd" \
        --assignee-object-id "${PRINCIPAL_ID}" \
        --scope /subscriptions/"${AZURE_SUBSCRIPTION_ID}"/resourceGroups/"${AZURE_RESOURCE_GROUP}" \
        --assignee-principal-type 'User' \
        --only-show-errors 2>/dev/null || echo "  (Role may already be assigned)"

echo ""
echo "✓ RBAC permissions updated successfully"
echo "======================================================================"

