#!/bin/bash

# ---------------------------------------------------------------------------------------------------------------------------------
# Add Models Script - Adds additional AI model deployments to existing infrastructure
# - Reads available models from customization/add-models.json
# - Allows user to select which models to deploy
# - Updates infrastructure with selected models
#
# Prerequisites:
# - 1-setup.sh must have been run successfully
# - ForBeginners repository must be cloned
# - Azure infrastructure must be deployed
# ---------------------------------------------------------------------------------------------------------------------------------

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Color formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
TARGET_DIR="$SCRIPT_DIR/ForBeginners"
AZD_SETUP_DIR="$SCRIPT_DIR/ForBeginners/.azd-setup"
MODELS_CONFIG="$SCRIPT_DIR/customization/add-models.json"

#==============================================================================
# Helper Functions
#==============================================================================

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if ForBeginners directory exists
    if [ ! -d "$TARGET_DIR" ]; then
        log_error "ForBeginners directory not found. Please run 1-setup.sh first."
        exit 1
    fi
    
    # Check if azd environment exists by navigating to .azd-setup directory
    if [ ! -d "$AZD_SETUP_DIR" ]; then
        log_error ".azd-setup directory not found in ForBeginners. Please run 1-setup.sh first."
        exit 1
    fi
    
    cd "$AZD_SETUP_DIR"
    
    # Check if azd environment exists
    local env_count=$(azd env list --output json 2>/dev/null | jq '. | length' 2>/dev/null || echo "0")
    if [ "$env_count" -eq 0 ]; then
        log_error "No azd environment found. Please run 1-setup.sh first to create and deploy the infrastructure."
        cd - > /dev/null
        exit 1
    fi
    
    cd - > /dev/null
    
    # Check if models config file exists
    if [ ! -f "$MODELS_CONFIG" ]; then
        log_error "Models configuration file not found at: $MODELS_CONFIG"
        exit 1
    fi
    
    # Check if jq is installed
    if ! command -v jq &> /dev/null; then
        log_error "jq is required but not installed. Please install jq."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

ensure_azure_auth() {
    log_info "Checking Azure authentication..."
    
    # Check if Azure CLI is authenticated
    if ! az account show &>/dev/null; then
        log_warning "Azure CLI is not authenticated."
        log_info "Attempting to authenticate using azd credentials..."
        
        cd "$AZD_SETUP_DIR"
        
        # Get the subscription ID from azd environment
        local subscription_id=$(azd env get-value AZURE_SUBSCRIPTION_ID 2>&1 || echo "")
        
        if [[ "$subscription_id" =~ ^ERROR ]] || [ -z "$subscription_id" ]; then
            log_error "Could not retrieve subscription ID from azd environment."
            log_error "Please run: az login"
            cd - > /dev/null
            exit 1
        fi
        
        cd - > /dev/null
        
        log_info "Please authenticate with Azure CLI..."
        if ! az login; then
            log_error "Azure CLI authentication failed."
            exit 1
        fi
        
        # Set the subscription
        log_info "Setting Azure subscription: $subscription_id"
        az account set --subscription "$subscription_id"
    fi
    
    log_success "Azure authentication verified"
}

patch_bicep_files() {
    log_info "Patching Bicep files to support additional model deployments..."
    
    local main_bicep="$AZD_SETUP_DIR/infra/main.bicep"
    local main_params="$AZD_SETUP_DIR/infra/main.parameters.json"
    
    # Check if already patched
    if grep -q "additionalModelDeployments" "$main_bicep" 2>/dev/null; then
        log_info "Bicep files already patched"
        return 0
    fi
    
    # Backup original files
    cp "$main_bicep" "$main_bicep.backup" 2>/dev/null || true
    cp "$main_params" "$main_params.backup" 2>/dev/null || true
    
    log_info "Patching main.bicep..."
    
    # Patch 1: Add parameter after azureTracingGenAIContentRecordingEnabled
    awk '
    /param azureTracingGenAIContentRecordingEnabled bool/ {
        print
        print ""
        print "@description('\''Additional model deployments to add beyond the base agent and embedding models'\'')"
        print "param additionalModelDeployments array = []"
        next
    }
    {print}
    ' "$main_bicep" > "$main_bicep.tmp" && mv "$main_bicep.tmp" "$main_bicep"
    
    # Patch 2: Update aiDeployments concat to include additionalModelDeployments
    awk '
    /var aiDeployments = concat\(/ {
        in_concat = 1
        print "var aiDeployments = concat("
        next
    }
    in_concat && /aiChatModel/ {
        print "  aiChatModel,"
        next
    }
    in_concat && /useSearchService/ {
        print "  useSearchService ? aiEmbeddingModel : [],"
        print "  additionalModelDeployments)"
        in_concat = 0
        next
    }
    !in_concat {print}
    ' "$main_bicep" > "$main_bicep.tmp" && mv "$main_bicep.tmp" "$main_bicep"
    
    log_success "Patched main.bicep"
    
    log_info "Patching main.parameters.json..."
    
    # Patch main.parameters.json - Remove the environment variable reference
    # We'll pass the parameter directly via command line instead
    local temp_file=$(mktemp)
    if jq 'del(.parameters.additionalModelDeployments)' "$main_params" > "$temp_file" 2>/dev/null; then
        mv "$temp_file" "$main_params"
        log_success "Patched main.parameters.json"
    else
        rm -f "$temp_file"
        log_info "additionalModelDeployments not in parameters file (expected on first run)"
    fi
    
    log_success "Bicep files patched successfully"
}

get_current_deployments() {
    cd "$AZD_SETUP_DIR"
    
    # Get the current environment name (we know it exists from prerequisites check)
    local current_env=$(azd env list --output json 2>/dev/null | jq -r '.[0].Name' 2>/dev/null || echo "")
    
    # Select the environment
    azd env select "$current_env" 2>/dev/null
    
    # Get currently deployed models from environment variables
    local agent_deployment=$(azd env get-value AZURE_AI_AGENT_DEPLOYMENT_NAME 2>/dev/null || echo "")
    local embed_deployment=$(azd env get-value AZURE_AI_EMBED_DEPLOYMENT_NAME 2>/dev/null || echo "")
    
    CURRENT_DEPLOYMENTS=()
    [ -n "$agent_deployment" ] && CURRENT_DEPLOYMENTS+=("$agent_deployment")
    [ -n "$embed_deployment" ] && CURRENT_DEPLOYMENTS+=("$embed_deployment")
    
    # Also check for additional models already deployed
    local additional_models=$(azd env get-value ADDITIONAL_MODEL_DEPLOYMENTS 2>/dev/null || echo "[]")
    if [ "$additional_models" != "[]" ] && [ -n "$additional_models" ]; then
        # Parse the JSON array and extract deployment names
        local additional_count=$(echo "$additional_models" | jq '. | length' 2>/dev/null || echo "0")
        for ((i=0; i<$additional_count; i++)); do
            local dep_name=$(echo "$additional_models" | jq -r ".[$i].name" 2>/dev/null || echo "")
            [ -n "$dep_name" ] && CURRENT_DEPLOYMENTS+=("$dep_name")
        done
    fi
    
    cd - > /dev/null
}

display_deployed_models() {
    if [ ${#CURRENT_DEPLOYMENTS[@]} -eq 0 ]; then
        return
    fi
    
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Currently Deployed Models${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    
    for deployment in "${CURRENT_DEPLOYMENTS[@]}"; do
        # Only display if deployment name is not empty
        if [ -n "$deployment" ] && [ "$deployment" != "null" ]; then
            echo -e "${GREEN}✓ $deployment${NC}"
        fi
    done
    
    echo ""
}

display_available_models() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Available Models to Deploy${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    # Read models from JSON file
    local model_count=$(jq '. | length' "$MODELS_CONFIG")
    local available_index=1
    
    # Create a mapping between display numbers and actual indices
    AVAILABLE_INDICES=()
    
    for ((i=0; i<$model_count; i++)); do
        local name=$(jq -r ".[$i].name" "$MODELS_CONFIG")
        local model_name=$(jq -r ".[$i].model.name" "$MODELS_CONFIG")
        local version=$(jq -r ".[$i].model.version" "$MODELS_CONFIG")
        local format=$(jq -r ".[$i].model.format" "$MODELS_CONFIG")
        local sku=$(jq -r ".[$i].sku.name" "$MODELS_CONFIG")
        local capacity=$(jq -r ".[$i].sku.capacity" "$MODELS_CONFIG")
        
        # Check if already deployed
        local is_deployed=false
        for deployed in "${CURRENT_DEPLOYMENTS[@]}"; do
            if [ "$deployed" == "$name" ]; then
                is_deployed=true
                break
            fi
        done
        
        # Only show models that are NOT deployed
        if [ "$is_deployed" = false ]; then
            echo -e "${YELLOW}${available_index}. ${NC}$name"
            echo -e "   Model: $model_name (v$version)"
            echo -e "   Format: $format | SKU: $sku | Capacity: $capacity"
            echo ""
            
            AVAILABLE_INDICES+=($i)
            available_index=$((available_index + 1))
        fi
    done
    
    if [ ${#AVAILABLE_INDICES[@]} -eq 0 ]; then
        echo -e "${YELLOW}No additional models available - all models are already deployed!${NC}"
        echo ""
        exit 0
    fi
}

select_models() {
    echo -e "${YELLOW}Select models to deploy (enter numbers separated by spaces, e.g., '1 3 5'):${NC}"
    echo -e "${YELLOW}Or type 'all' to deploy all available models, or 'cancel' to exit:${NC}"
    read -p "> " selection
    
    if [ "$selection" == "cancel" ]; then
        log_info "Operation cancelled by user"
        exit 0
    fi
    
    SELECTED_INDICES=()
    
    if [ "$selection" == "all" ]; then
        # Select all available (non-deployed) models
        for index in "${AVAILABLE_INDICES[@]}"; do
            SELECTED_INDICES+=($index)
        done
    else
        # Parse selected numbers and map to actual indices
        for num in $selection; do
            local array_pos=$((num-1))
            
            if [ $array_pos -ge 0 ] && [ $array_pos -lt ${#AVAILABLE_INDICES[@]} ]; then
                local actual_index=${AVAILABLE_INDICES[$array_pos]}
                SELECTED_INDICES+=($actual_index)
            else
                log_warning "Invalid selection: $num (skipped)"
            fi
        done
    fi
    
    if [ ${#SELECTED_INDICES[@]} -eq 0 ]; then
        log_error "No valid models selected"
        exit 1
    fi
}

build_deployment_array() {
    echo ""
    log_info "Building deployment configuration..."
    
    cd "$AZD_SETUP_DIR"
    
    # Get existing additional model deployments
    local existing_deployments=$(azd env get-value ADDITIONAL_MODEL_DEPLOYMENTS 2>&1 || echo "[]")
    
    # Check if the result is an error message or empty/invalid JSON
    if [[ "$existing_deployments" =~ ^ERROR ]] || [ -z "$existing_deployments" ] || [ "$existing_deployments" = "[]" ]; then
        DEPLOYMENT_JSON="["
        local first=true
    else
        # Validate that we have valid JSON
        if ! echo "$existing_deployments" | jq empty 2>/dev/null; then
            DEPLOYMENT_JSON="["
            local first=true
        else
            # Remove the closing bracket from existing deployments to append new ones
            DEPLOYMENT_JSON="${existing_deployments%]}"
            # Don't add comma yet - will add it before first new item
            local first=false
        fi
    fi
    
    # Add new selected models
    for index in "${SELECTED_INDICES[@]}"; do
        local name=$(jq -r ".[$index].name" "$MODELS_CONFIG")
        local model_name=$(jq -r ".[$index].model.name" "$MODELS_CONFIG")
        local model_version=$(jq -r ".[$index].model.version" "$MODELS_CONFIG")
        local model_format=$(jq -r ".[$index].model.format" "$MODELS_CONFIG")
        local sku_name=$(jq -r ".[$index].sku.name" "$MODELS_CONFIG")
        local capacity=$(jq -r ".[$index].sku.capacity" "$MODELS_CONFIG")
        
        # Add comma separator if not first entry
        if [ "$first" = false ]; then
            DEPLOYMENT_JSON+=","
        fi
        first=false
        
        # Build JSON object for this deployment
        DEPLOYMENT_JSON+="{\"name\":\"$name\",\"model\":{\"format\":\"$model_format\",\"name\":\"$model_name\",\"version\":\"$model_version\"},\"sku\":{\"name\":\"$sku_name\",\"capacity\":$capacity}}"
        
        log_info "  Adding: $name ($model_name v$model_version)"
    done
    
    DEPLOYMENT_JSON+="]"
    
    # Validate the JSON
    if ! echo "$DEPLOYMENT_JSON" | jq empty 2>/dev/null; then
        log_error "Generated invalid JSON configuration"
        log_error "JSON: $DEPLOYMENT_JSON"
        exit 1
    fi
    
    log_success "Deployment configuration built"
    cd - > /dev/null
}

deploy_additional_models() {
    echo ""
    log_info "Deploying additional models..."
    
    cd "$AZD_SETUP_DIR"
    
    # Get the current environment name
    local current_env=$(azd env list --output json 2>/dev/null | jq -r '.[0].Name' 2>/dev/null || echo "")
    
    if [ -z "$current_env" ] || [ "$current_env" == "null" ]; then
        log_error "No azd environment found. Please run 1-setup.sh first."
        cd - > /dev/null
        exit 1
    fi
    
    log_info "Using existing environment: $current_env"
    azd env select "$current_env"
    
    # Get required values from azd environment
    local resource_group=$(azd env get-value AZURE_RESOURCE_GROUP 2>/dev/null || echo "")
    local resource_id=$(azd env get-value AZURE_EXISTING_AIPROJECT_RESOURCE_ID 2>/dev/null || echo "")
    
    # Extract the AI Services account name from the resource ID
    # Format: /subscriptions/.../resourceGroups/.../providers/Microsoft.CognitiveServices/accounts/ACCOUNT_NAME/...
    local ai_account_name=""
    if [ -n "$resource_id" ]; then
        ai_account_name=$(echo "$resource_id" | grep -oP 'accounts/\K[^/]+' || echo "")
    fi
    
    if [ -z "$resource_group" ] || [ -z "$ai_account_name" ]; then
        log_error "Could not retrieve required information from environment"
        log_error "Resource Group: $resource_group"
        log_error "AI Account: $ai_account_name"
        log_error "Resource ID: $resource_id"
        cd - > /dev/null
        exit 1
    fi
    
    log_info "Resource Group: $resource_group"
    log_info "AI Account: $ai_account_name"
    
    # Create parameters file for the deployment
    local params_file=$(mktemp)
    cat > "$params_file" << EOF
{
  "\$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentParameters.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "accountName": {
      "value": "$ai_account_name"
    },
    "modelDeployments": {
      "value": $DEPLOYMENT_JSON
    }
  }
}
EOF
    
    log_info "Deployment parameters:"
    cat "$params_file" | jq '.'
    
    log_warning "This will deploy the selected models to your Azure AI Services account."
    log_info "Running: az deployment group create"
    echo ""
    
    # Deploy using Azure CLI directly
    local template_file="$SCRIPT_DIR/customization/add-models.bicep"
    
    if az deployment group create \
        --resource-group "$resource_group" \
        --template-file "$template_file" \
        --parameters "@$params_file" \
        --name "additional-models-$(date +%Y%m%d-%H%M%S)"; then
        
        log_success "Additional models deployed successfully!"
        
        # Update the environment variable to track what we've deployed
        local backup_deployments=$(azd env get-value ADDITIONAL_MODEL_DEPLOYMENTS 2>/dev/null || echo "[]")
        azd env set ADDITIONAL_MODEL_DEPLOYMENTS "$DEPLOYMENT_JSON"
        log_success "Environment variable updated with deployment configuration"
    else
        log_error "Failed to deploy additional models"
        rm -f "$params_file"
        cd - > /dev/null
        exit 1
    fi
    
    rm -f "$params_file"
    cd - > /dev/null
}

display_summary() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Deployment Summary${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    for index in "${SELECTED_INDICES[@]}"; do
        local name=$(jq -r ".[$index].name" "$MODELS_CONFIG")
        echo -e "${GREEN}✓ $name deployed${NC}"
    done
    
    echo -e "${GREEN}========================================${NC}"
}

#==============================================================================
# Main Execution
#==============================================================================

main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Add Additional Model Deployments${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    check_prerequisites
    ensure_azure_auth
    # No need to patch Bicep files - using standalone template
    log_info "Using standalone Bicep template: customization/add-models.bicep"
    get_current_deployments
    display_deployed_models
    display_available_models
    select_models
    build_deployment_array
    deploy_additional_models
    display_summary
    
    echo ""
    log_success "Model deployment completed successfully!"
}

# Run main function
main
