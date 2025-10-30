#!/bin/bash

# ---------------------------------------------------------------------------------------------------------------------------------
# Setup Script - Provisions infrastructure for the labs
# - Clones the ForBeginners repository (if not already present)
# - Creates and configures AZD environment
# - Deploys Azure infrastructure using azd up
#
# Note: This script is for testing purposes only.
#       In-venue labs use pre-provisioned Azure subscriptions with resources already deployed.
# ---------------------------------------------------------------------------------------------------------------------------------

set -e  # Exit on error during setup

# Color formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
REPO_URL="https://github.com/microsoft/ForBeginners"
DEFAULT_BRANCH="aitour26-demos"
TARGET_DIR="./ForBeginners"
AZD_SETUP_DIR="./ForBeginners/.azd-setup"

#==============================================================================
# Helper Functions
#==============================================================================

clone_forbeginners_repo() {
    if [ -d "$TARGET_DIR" ]; then
        echo -e "${YELLOW}ForBeginners directory already exists. Skipping clone.${NC}"
        return 0
    fi
    
    # Prompt for branch name
    echo -e "${YELLOW}Cloning ForBeginners repository...${NC}"
    read -p "Enter branch name [${DEFAULT_BRANCH}]: " branch_input
    local branch=${branch_input:-$DEFAULT_BRANCH}
    
    echo -e "${YELLOW}Cloning branch: ${branch}${NC}"
    if git clone -b "$branch" --single-branch "$REPO_URL" "$TARGET_DIR"; then
        echo -e "${GREEN}✓ Repository cloned successfully from branch: ${branch}${NC}"
    else
        echo -e "${RED}✗ Failed to clone repository from branch: ${branch}${NC}"
        exit 1
    fi
}

setup_azd_environment() {
    echo -e "${YELLOW}Setting up AZD environment...${NC}"
    
    # Check if an environment already exists
    local existing_env=$(azd env list --output json 2>/dev/null | jq -r '.[0].Name' 2>/dev/null || echo "")
    
    # Validate we have a real environment (not empty and not "null")
    if [ -n "$existing_env" ] && [ "$existing_env" != "null" ]; then
        echo -e "${YELLOW}Found existing AZD environment: ${existing_env}${NC}"
        read -p "Use existing environment? (yes/no): " use_existing
        
        if [ "$use_existing" != "yes" ]; then
            create_new_environment
        fi
    else
        echo -e "${YELLOW}No existing AZD environment found. Creating new one...${NC}"
        create_new_environment
    fi
}

create_new_environment() {
    echo -e "${YELLOW}Creating new AZD environment...${NC}"
    
    # Prompt for environment details
    read -p "Enter environment name: " env_name
    read -p "Enter Azure region [swedencentral]: " region
    region=${region:-swedencentral}
    read -p "Enter subscription ID (optional): " subscription_id
    
    # Create environment
    if [ -n "$subscription_id" ]; then
        azd env new "$env_name" --location "$region" --subscription "$subscription_id"
    else
        azd env new "$env_name" --location "$region"
    fi
    
    echo -e "${GREEN}✓ Environment created: ${env_name}${NC}"
}

configure_environment_variables() {
    echo -e "${YELLOW}Configuring environment variables...${NC}"
    
    # Set required environment variables for the deployment
    azd env set USE_APPLICATION_INSIGHTS true
    azd env set ENABLE_AZURE_MONITOR_TRACING true
    azd env set AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED true
    
    echo -e "${GREEN}✓ Base environment variables configured${NC}"
}

configure_azure_ai_model() {
    echo ""
    echo -e "${YELLOW}======================================${NC}"
    echo -e "${YELLOW}  Azure AI Model Configuration${NC}"
    echo -e "${YELLOW}======================================${NC}"
    echo ""
    
    # Path to the models configuration file (relative to AZD_SETUP_DIR: ./ForBeginners/.azd-setup)
    # The file is in infra/customization/, so we need to go up two levels from .azd-setup
    local models_file="../../customization/add-models.json"
    
    if [ ! -f "$models_file" ]; then
        echo -e "${RED}⚠️  Models configuration file not found: $models_file${NC}"
        echo -e "${YELLOW}Using default model configuration...${NC}"
        use_default_model=true
    else
        echo -e "${YELLOW}Available AI models:${NC}"
        echo ""
        
        # Extract and display chat models (exclude embedding models)
        local models=$(jq -r '.[] | select(.model.name | test("gpt|o3|o4")) | "\(.model.name) (version: \(.model.version), capacity: \(.sku.capacity))"' "$models_file")
        echo "$models" | nl -w2 -s'. '
        echo ""
        echo -e "${YELLOW}Default: gpt-4.1 (version: 2025-04-14, capacity: 50)${NC}"
        echo ""
        
        read -p "Do you want to select a different model? (yes/no) [no]: " select_custom
        select_custom=${select_custom:-no}
        
        if [ "$select_custom" == "no" ]; then
            use_default_model=true
        else
            use_default_model=false
            echo ""
            read -p "Enter model name from the list above: " selected_model
            
            # Validate and extract model details
            local model_info=$(jq -r --arg model "$selected_model" '.[] | select(.model.name == $model) | {name: .model.name, version: .model.version, capacity: .sku.capacity}' "$models_file")
            
            if [ -z "$model_info" ] || [ "$model_info" == "null" ]; then
                echo -e "${RED}⚠️  Model '$selected_model' not found. Using default (gpt-4.1)${NC}"
                use_default_model=true
            else
                AGENT_MODEL_NAME=$(echo "$model_info" | jq -r '.name')
                AGENT_MODEL_VERSION=$(echo "$model_info" | jq -r '.version')
                AGENT_DEPLOYMENT_CAPACITY=$(echo "$model_info" | jq -r '.capacity')
                AGENT_DEPLOYMENT_NAME="$AGENT_MODEL_NAME"
                echo -e "${GREEN}✓ Selected: $AGENT_MODEL_NAME${NC}"
            fi
        fi
    fi
    
    # Set defaults if needed
    if [ "$use_default_model" == true ]; then
        AGENT_MODEL_NAME="gpt-4.1"
        AGENT_MODEL_VERSION="2025-04-14"
        AGENT_DEPLOYMENT_CAPACITY="50"
        AGENT_DEPLOYMENT_NAME="gpt-4.1"
        echo -e "${GREEN}✓ Using default: $AGENT_MODEL_NAME${NC}"
    fi
    
    # Prompt for capacity override
    echo ""
    read -p "Enter deployment capacity [$AGENT_DEPLOYMENT_CAPACITY]: " capacity_input
    AGENT_DEPLOYMENT_CAPACITY=${capacity_input:-$AGENT_DEPLOYMENT_CAPACITY}
    
    # Set environment variables for Azure AI Model
    azd env set AZURE_AI_AGENT_DEPLOYMENT_CAPACITY "$AGENT_DEPLOYMENT_CAPACITY"
    azd env set AZURE_AI_AGENT_DEPLOYMENT_NAME "$AGENT_DEPLOYMENT_NAME"
    azd env set AZURE_AI_AGENT_MODEL_NAME "$AGENT_MODEL_NAME"
    azd env set AZURE_AI_AGENT_MODEL_VERSION "$AGENT_MODEL_VERSION"
    
    echo ""
    echo -e "${GREEN}✓ Azure AI Model configured${NC}"
    echo -e "${GREEN}  - Model: $AGENT_MODEL_NAME${NC}"
    echo -e "${GREEN}  - Version: $AGENT_MODEL_VERSION${NC}"
    echo -e "${GREEN}  - Deployment Name: $AGENT_DEPLOYMENT_NAME${NC}"
    echo -e "${GREEN}  - Capacity: $AGENT_DEPLOYMENT_CAPACITY${NC}"
    echo ""
}

configure_azure_ai_search() {
    echo ""
    echo -e "${YELLOW}======================================${NC}"
    echo -e "${YELLOW}  Azure AI Search Configuration${NC}"
    echo -e "${YELLOW}======================================${NC}"
    echo ""
    echo -e "${YELLOW}Azure AI Search enables RAG (Retrieval Augmented Generation) by searching${NC}"
    echo -e "${YELLOW}through indexed documents to provide context for better AI responses.${NC}"
    echo ""
    echo -e "${RED}⚠️  IMPORTANT: Search must be configured BEFORE initial deployment!${NC}"
    echo ""
    
    read -p "Do you want to activate Azure AI Search? (yes/no) [no]: " enable_search
    enable_search=${enable_search:-no}
    
    if [ "$enable_search" == "yes" ]; then
        echo ""
        echo -e "${YELLOW}Configuring Azure AI Search with defaults:${NC}"
        echo -e "  - Index Name: zava-products"
        echo -e "  - Embedding Model: text-embedding-3-large"
        echo -e "  - Model Version: 1"
        echo -e "  - SKU: Standard"
        echo -e "  - Capacity: 50"
        echo ""
        
        read -p "Use these defaults? (yes/no) [yes]: " use_defaults
        use_defaults=${use_defaults:-yes}
        
        if [ "$use_defaults" == "yes" ]; then
            SEARCH_INDEX_NAME="zava-products"
            EMBED_MODEL_NAME="text-embedding-3-large"
            EMBED_MODEL_VERSION="1"
            EMBED_MODEL_FORMAT="OpenAI"
            EMBED_DEPLOYMENT_NAME="text-embedding-3-large"
            EMBED_SKU_NAME="Standard"
            EMBED_CAPACITY="50"
        else
            read -p "Enter search index name [zava-products]: " index_input
            SEARCH_INDEX_NAME=${index_input:-zava-products}
            
            read -p "Enter embedding model name [text-embedding-3-large]: " model_input
            EMBED_MODEL_NAME=${model_input:-text-embedding-3-large}
            
            read -p "Enter model version [1]: " version_input
            EMBED_MODEL_VERSION=${version_input:-1}
            
            read -p "Enter model format [OpenAI]: " format_input
            EMBED_MODEL_FORMAT=${format_input:-OpenAI}
            
            read -p "Enter deployment name [text-embedding-3-large]: " deploy_input
            EMBED_DEPLOYMENT_NAME=${deploy_input:-text-embedding-3-large}
            
            read -p "Enter SKU name [Standard]: " sku_input
            EMBED_SKU_NAME=${sku_input:-Standard}
            
            read -p "Enter capacity [50]: " capacity_input
            EMBED_CAPACITY=${capacity_input:-50}
        fi
        
        # Set environment variables for Azure AI Search
        azd env set USE_AZURE_AI_SEARCH_SERVICE true
        azd env set AZURE_AI_SEARCH_INDEX_NAME "$SEARCH_INDEX_NAME"
        azd env set AZURE_AI_EMBED_DEPLOYMENT_NAME "$EMBED_DEPLOYMENT_NAME"
        azd env set AZURE_AI_EMBED_MODEL_NAME "$EMBED_MODEL_NAME"
        azd env set AZURE_AI_EMBED_MODEL_VERSION "$EMBED_MODEL_VERSION"
        azd env set AZURE_AI_EMBED_MODEL_FORMAT "$EMBED_MODEL_FORMAT"
        azd env set AZURE_AI_EMBED_DEPLOYMENT_SKU "$EMBED_SKU_NAME"
        azd env set AZURE_AI_EMBED_DEPLOYMENT_CAPACITY "$EMBED_CAPACITY"
        
        echo ""
        echo -e "${GREEN}✓ Azure AI Search configured${NC}"
        echo -e "${GREEN}  - Search Service: Enabled${NC}"
        echo -e "${GREEN}  - Index Name: $SEARCH_INDEX_NAME${NC}"
        echo -e "${GREEN}  - Embedding Model: $EMBED_MODEL_NAME (version: $EMBED_MODEL_VERSION)${NC}"
        echo ""
    else
        echo -e "${YELLOW}Azure AI Search will not be enabled${NC}"
        azd env set USE_AZURE_AI_SEARCH_SERVICE false
    fi
}

deploy_infrastructure() {
    echo -e "${YELLOW}======================================${NC}"
    echo -e "${YELLOW}Ready to deploy Azure infrastructure${NC}"
    echo -e "${YELLOW}======================================${NC}"
    
    read -p "Proceed with deployment? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo -e "${YELLOW}Deployment cancelled${NC}"
        exit 0
    fi
    
    echo -e "${YELLOW}Running azd up...${NC}"
    if azd up --no-prompt; then
        echo -e "${GREEN}✓ Infrastructure deployed successfully${NC}"
    else
        echo -e "${RED}✗ Deployment failed${NC}"
        exit 1
    fi
}

#==============================================================================
# Main Execution
#==============================================================================

echo -e "${YELLOW}Starting setup process...${NC}"

# Clone the ForBeginners repository
clone_forbeginners_repo

# Navigate to the AZD setup directory
if [ ! -d "$AZD_SETUP_DIR" ]; then
    echo -e "${RED}✗ AZD setup directory not found: ${AZD_SETUP_DIR}${NC}"
    exit 1
fi

cd "$AZD_SETUP_DIR"
echo -e "${GREEN}Changed to AZD setup directory${NC}"

# Setup AZD environment
setup_azd_environment

# Configure environment variables
configure_environment_variables

# Configure Azure AI Model
configure_azure_ai_model

# Configure Azure AI Search (optional)
configure_azure_ai_search

# Deploy infrastructure
deploy_infrastructure

# Summary
echo ""
echo -e "${YELLOW}======================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${YELLOW}======================================${NC}"
echo -e "${GREEN}✓ Repository cloned${NC}"
echo -e "${GREEN}✓ Environment configured${NC}"
echo -e "${GREEN}✓ Infrastructure deployed${NC}"

# Check if search was enabled
search_enabled=$(azd env get-value USE_AZURE_AI_SEARCH_SERVICE 2>/dev/null || echo "false")
if [ "$search_enabled" == "true" ]; then
    search_index=$(azd env get-value AZURE_AI_SEARCH_INDEX_NAME 2>/dev/null || echo "N/A")
    echo -e "${GREEN}✓ Azure AI Search enabled (Index: $search_index)${NC}"
else
    echo -e "${YELLOW}ℹ️  Azure AI Search not enabled${NC}"
fi

echo -e "${YELLOW}======================================${NC}"
