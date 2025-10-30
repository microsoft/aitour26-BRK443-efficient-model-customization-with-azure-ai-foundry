#!/bin/bash
# ---------------------------------------------------------------------------------------------------------------------------------
# This script is for testing purposes only.
# It reverses the setup process in the 1-setup.sh script and cleans up resources.
# This is NOT required for in-venue labs (where resources are pre-provisioned and torn down automatically).
#
# This script will
# - Check if you have a ForBeginners clone and an active AZD environment
# - Tears down Azure infrastructure (if it exists)
# - Deletes the ForBeginners directory clone
# ---------------------------------------------------------------------------------------------------------------------------------

set +e  # Continue cleanup even if individual steps fail

# Color formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# State tracking
ORIGINAL_DIR=$(pwd)
INFRA_TORN_DOWN=false
DIRECTORY_DELETED=false

#==============================================================================
# Helper Functions
#==============================================================================

get_azd_environment() {
    # Try to get the default environment first, then fall back to the first available
    local env=$(azd env list --output json 2>/dev/null | jq -r '.[] | select(.IsDefault == true) | .Name' 2>/dev/null || echo "")
    
    if [ -z "$env" ]; then
        env=$(azd env list --output json 2>/dev/null | jq -r '.[0].Name' 2>/dev/null || echo "")
    fi
    
    echo "$env"
}

check_deployment_exists() {
    local env=$1
    local env_file=".azure/${env}/.env"
    
    # Check if .env file exists
    if [ ! -f "$env_file" ]; then
        echo "false"
        return
    fi
    
    # Check for deployment indicators (resource group or location are set after deployment)
    if grep -q "AZURE_RESOURCE_GROUP=" "$env_file" || grep -q "AZURE_LOCATION=" "$env_file"; then
        echo "true"
        return
    fi
    
    # Additional check: look for any service endpoints or URLs (indicates actual resources)
    if grep -qE "SERVICE_|_ENDPOINT|_URL|_ID=" "$env_file" | grep -v "SUBSCRIPTION_ID\|TENANT_ID"; then
        echo "true"
        return
    fi
    
    echo "false"
}

teardown_azure_infrastructure() {
    echo -e "${YELLOW}Checking for AZD environments...${NC}"
    
    # Validate AZD project exists
    if ! azd env list &>/dev/null; then
        echo -e "${YELLOW}No AZD environments found. Infrastructure may have been torn down previously.${NC}"
        return
    fi
    
    # Get environment name
    local env=$(get_azd_environment)
    
    # Validate we have a real environment (not empty or "null")
    if [ -z "$env" ] || [ "$env" = "null" ]; then
        echo -e "${YELLOW}No valid AZD environment found (got: '${env}'). Skipping infrastructure teardown.${NC}"
        return
    fi
    
    echo -e "${GREEN}Found AZD environment: ${env}${NC}"
    
    # Check if infrastructure was actually deployed
    local deployment_exists=$(check_deployment_exists "$env")
    
    if [ "$deployment_exists" = "false" ]; then
        echo -e "${YELLOW}No deployed infrastructure found for environment '${env}'.${NC}"
        echo -e "${YELLOW}Environment was created but 'azd up' was never run or deployment failed.${NC}"
        echo -e "${YELLOW}Skipping infrastructure teardown.${NC}"
        return
    fi
    
    # Confirm and execute teardown
    echo -e "${YELLOW}======================================${NC}"
    echo -e "${YELLOW}WARNING: This will delete all Azure resources${NC}"
    echo -e "${YELLOW}======================================${NC}"
    
    read -p "Tear down Azure infrastructure? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        echo -e "${YELLOW}Running azd down --purge for environment: ${env}${NC}"
        if azd down --purge --force; then
            echo -e "${GREEN}✓ Azure infrastructure torn down successfully${NC}"
            INFRA_TORN_DOWN=true
        else
            echo -e "${RED}✗ Failed to tear down infrastructure${NC}"
        fi
    else
        echo -e "${YELLOW}Skipping infrastructure teardown${NC}"
    fi
}

delete_forbeginners_directory() {
    echo -e "${YELLOW}======================================${NC}"
    echo -e "${YELLOW}Preparing to delete ForBeginners directory${NC}"
    echo -e "${YELLOW}======================================${NC}"
    
    read -p "Delete ./ForBeginners directory? (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        echo -e "${YELLOW}Deleting ./ForBeginners directory...${NC}"
        if rm -rf ./ForBeginners; then
            echo -e "${GREEN}✓ ForBeginners directory deleted${NC}"
            DIRECTORY_DELETED=true
        else
            echo -e "${RED}✗ Failed to delete ForBeginners directory${NC}"
        fi
    else
        echo -e "${YELLOW}Skipping directory deletion${NC}"
    fi
}

print_summary() {
    echo -e "${YELLOW}======================================${NC}"
    echo -e "${GREEN}Teardown Complete${NC}"
    echo -e "${YELLOW}======================================${NC}"
    
    if [ "$INFRA_TORN_DOWN" = true ]; then
        echo -e "${GREEN}✓ Azure infrastructure torn down${NC}"
    else
        echo -e "${YELLOW}○ Azure infrastructure not torn down${NC}"
    fi
    
    if [ "$DIRECTORY_DELETED" = true ]; then
        echo -e "${GREEN}✓ ForBeginners directory deleted${NC}"
    else
        echo -e "${YELLOW}○ ForBeginners directory not deleted${NC}"
    fi
    
    echo -e "${YELLOW}======================================${NC}"
}

#==============================================================================
# Main Execution
#==============================================================================

echo -e "${YELLOW}Starting teardown process...${NC}"

# Check if ForBeginners directory exists
if [ ! -d "./ForBeginners" ]; then
    echo -e "${YELLOW}ForBeginners directory not found. Nothing to tear down.${NC}"
    exit 0
fi

echo -e "${GREEN}Found ForBeginners directory${NC}"

# Attempt infrastructure teardown if AZD project exists
if [ -d "./ForBeginners/.azd-setup" ] && [ -f "./ForBeginners/.azd-setup/azure.yaml" ]; then
    cd ./ForBeginners/.azd-setup
    teardown_azure_infrastructure
    cd "$ORIGINAL_DIR"
else
    echo -e "${YELLOW}No AZD project found. Skipping infrastructure teardown.${NC}"
fi


# Delete the ForBeginners directory
delete_forbeginners_directory

# Print summary
print_summary
