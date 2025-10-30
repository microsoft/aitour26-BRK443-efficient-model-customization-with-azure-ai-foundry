# Infrastructure Setup

This directory contains automation scripts for setting up, customizing, and managing Azure AI infrastructure for the AI Tour BRK443 lab.

## 📁 Directory Structure

```
infra/
├── README.md                          # This file
├── 1-setup.sh                         # Initial infrastructure setup
├── 2-add-models.sh                    # Add additional AI models
├── 3-teardown.sh                      # Cleanup and resource deletion
├── customization/                     # Additional model configurations
│   ├── add-models.bicep              # Bicep template for additional models
│   ├── add-models.json               # Low-capacity model configurations (capacity=20)
│   └── add-models-high-capacity.json # High-capacity model configurations (capacity=50)
└── ForBeginners/                      # Cloned repository (created by 1-setup.sh)
    └── .azd-setup/                   # Azure Developer CLI setup
        ├── .azure/                   # Environment data
        │   └── [env-name]/           # Environment-specific config
        │       └── .env              # Environment variables
        ├── infra/                    # Bicep infrastructure files
        │   ├── main.bicep           # Main infrastructure template
        │   └── core/                # Core Bicep modules
        └── api/                      # API application code
```

## 🚀 Quick Start

### Prerequisites

- Azure subscription with appropriate permissions
- Azure CLI (`az`) installed and logged in
- Azure Developer CLI (`azd`) installed
- `jq` installed (for JSON processing)
- Bash shell environment

```bash
az login
```

### Three Simple Steps

```bash
cd infra/

# 1. Set up infrastructure (first time)
./1-setup.sh

# 2. Add more AI models (optional, repeatable)
./2-add-models.sh

# 3. Clean up when done
./3-teardown.sh
```

---

## 📖 Detailed Script Guide

#### 1️⃣ **Initial Setup** - `1-setup.sh`

Sets up your Azure infrastructure from scratch.

**What happens:**
- Downloads the ForBeginners repository
- Creates an Azure Developer CLI environment
- Asks if you want Azure AI Search (for document search capabilities)
- Deploys all Azure resources to your subscription

**Usage:**
```bash
cd scripts/
./1-setup.sh
```

**You'll be asked for:**
- Environment name (e.g., "my-ai-lab")
- Azure region (e.g., "eastus")
- Azure subscription to use
- Whether to enable Azure AI Search (yes/no)

**Result:**
- Azure AI Foundry project created
- AI models deployed
- (Optional) Search service configured

---

#### 2️⃣ **Add Additional Models** - `2-add-models.sh`

Adds more AI models to your existing setup.

**Before running:** Complete `1-setup.sh` first

**What happens:**
- Shows which models are already deployed
- Shows which models are available to add
- Lets you pick which ones to deploy
- Deploys your selected models

**Usage:**
```bash
cd scripts/
./2-add-models.sh
```

**Available Models:**
- `model-router` - Routes requests to optimal models
- `gpt-4.1` - Latest GPT-4 model
- `gpt-4o` - Optimized GPT-4
- `gpt-4o-mini` - Smaller, faster GPT-4
- `gpt-4.1-mini` - Latest mini model
- `gpt-4.1-nano` - Smallest, fastest model
- `o3-mini` - OpenAI O-series mini
- `o4-mini` - OpenAI O-series v4
- `text-embedding-3-large` - Text embeddings

**How to select:**
- Type numbers: `1 3 5` (deploys models #1, #3, and #5)
- Type `all` (deploys all available models)
- Type `cancel` (exits without changes)

**Result:**
- New models available in your AI Foundry project

**Note about capacity:**
- Default configuration uses capacity=20 per model
- If you have higher Azure quota, swap to high-capacity version:
  ```bash
  cd customization/
  mv add-models.json add-models-low-capacity.json
  mv add-models-high-capacity.json add-models.json
  ```

---

#### 3️⃣ **Teardown** - `3-teardown.sh`

Removes all Azure resources and cleans up local files.

**Usage:**
```bash
cd scripts/
./3-teardown.sh
```

**⚠️ Warning:** This deletes everything and cannot be undone!

**What happens:**
- Deletes all Azure resources
- Removes local ForBeginners directory
- Frees up Azure quota

---

## 🔧 Configuration Files

### `customization/add-models.bicep`

This is a Bicep template that deploys additional AI models to your existing Azure AI account. The script uses this template automatically - you don't need to edit it.

### `customization/add-models.json`

Lists all available AI models that can be deployed. Each model includes:

```json
{
  "name": "gpt-4o",
  "model": {
    "format": "OpenAI",
    "name": "gpt-4o",
    "version": "2024-11-20"
  },
  "sku": {
    "name": "GlobalStandard",
    "capacity": 20
  }
}
```

**To add a custom model:**
1. Add an entry to `add-models.json` with the model details
2. Run `./2-add-models.sh`
3. Select your new model from the list

### `customization/add-models-high-capacity.json`

Same as `add-models.json` but with `capacity: 50` instead of `capacity: 20`. Use this if you have higher Azure quota limits.

---

## 🛠️ How It Works

### Adding Models - Technical Overview

When you run `2-add-models.sh`, here's what happens:

1. **Reads Configuration** - Script loads available models from `add-models.json`
2. **Checks What's Deployed** - Looks at your Azure environment to see what models are already there
3. **Shows Your Options** - Displays available models (hiding already-deployed ones)
4. **You Choose** - You select which models to add
5. **Deploys to Azure** - Uses the `add-models.bicep` template to deploy directly to your AI account

**Why this approach?**

The script uses a standalone Bicep template (instead of modifying the main infrastructure) because:
- ✅ Cleaner and simpler
- ✅ Doesn't modify the ForBeginners repository
- ✅ Can be run multiple times safely
- ✅ Easy to understand and maintain

**Bicep Template Structure:**

The template references your existing AI account and adds new model deployments:

```bicep
param accountName string                    // Your AI account name
param modelDeployments array               // Models to deploy

resource account 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = {
  name: accountName
}

resource additionalDeployments 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = [
  for deployment in modelDeployments: {
    parent: account
    name: deployment.name
    sku: deployment.sku
    properties: {
      model: deployment.model
    }
  }
]
```

---

## 🔍 Important Tips

### Always Run Scripts from the `scripts/` Directory

```bash
# ✅ Correct
cd scripts/
./1-setup.sh

# ❌ Wrong - will fail
./scripts/1-setup.sh
```

The scripts expect to be run from the `scripts/` directory.

### Azure AI Search Must Be Enabled During Initial Setup

If you want document search capabilities:
- Say "yes" when `1-setup.sh` asks about Azure AI Search
- Cannot be easily added later

### About Capacity and Quotas

Each model deployment requires "capacity" - think of it like reserving computing power.

- **Default:** Each model uses capacity=20
- **Your Azure subscription** has capacity limits (quotas)
- **If deployment fails** with "insufficient quota", you can:
  - Deploy fewer models at once
  - Request quota increase from Azure support
  - Use a different Azure region

---

## 🐛 Troubleshooting

### "No azd environment found"

**Problem:** You haven't run setup yet.

**Fix:**
```bash
cd scripts/
./1-setup.sh
```

### "ForBeginners directory not found"

**Problem:** The repository didn't download properly.

**Fix:**
1. Delete the ForBeginners folder if it exists
2. Run `./1-setup.sh` again

### "Insufficient quota for model deployment"

**Problem:** Your Azure subscription doesn't have enough capacity quota.

**Fix - Option 1 (Deploy fewer models):**
- Select fewer models when running `2-add-models.sh`

**Fix - Option 2 (Request more quota):**
1. Go to Azure Portal
2. Navigate to your AI Service
3. Request quota increase

**Fix - Option 3 (Use different region):**
- Run `./3-teardown.sh`
- Run `./1-setup.sh` and choose a different region

### "Models configuration file not found"

**Problem:** The `add-models.json` file is missing.

**Fix:**
```bash
cd scripts/
ls -la customization/add-models.json
```

If it doesn't exist, the file may have been accidentally deleted.

---

---

## 🐛 Troubleshooting

### Error: "No azd environment found"

**Cause:** `1-setup.sh` hasn't been run or failed to complete.

**Solution:**
```bash
cd scripts/
./1-setup.sh
```

### Error: "ForBeginners directory not found"

**Cause:** ForBeginners is in the wrong location or doesn't exist.

**Solution:**
1. Check if `scripts/ForBeginners/` exists
2. If not, run `./1-setup.sh`
3. If in wrong location, move it to `scripts/ForBeginners/`

### Error: "Models configuration file not found"

**Cause:** `customization/add-models.json` is missing.

**Solution:**
```bash
cd scripts/
ls -la customization/add-models.json
```

### Multiple ForBeginners Folders

**Problem:** ForBeginners exists in multiple locations.

**Solution:**
1. Navigate to the incorrect location
2. Run `azd down` to clean up resources (if deployed)
3. Delete the incorrect ForBeginners directory
4. Keep only `scripts/ForBeginners/`

---

## � Infrastructure Files

### Modified Bicep Files

The following Bicep files were modified to support dynamic model deployments:

**`ForBeginners/.azd-setup/infra/main.bicep`**
- Added `additionalModelDeployments` parameter
- Modified `aiDeployments` variable to concatenate additional models
- Maintains backward compatibility with empty array default

**`ForBeginners/.azd-setup/infra/main.parameters.json`**
- Added `ADDITIONAL_MODEL_DEPLOYMENTS` environment variable mapping
- Defaults to `[]` if not set

**`ForBeginners/.azd-setup/infra/core/ai/cognitiveservices.bicep`**
- Receives deployments array from main.bicep
- Provisions all model deployments with `@batchSize(1)` for sequential deployment
- Handles model format, version, SKU, and capacity configuration

---

## 📚 Additional Resources

- **ForBeginners Repository:** https://github.com/microsoft/ForBeginners
- **Azure AI Foundry:** https://learn.microsoft.com/azure/ai-studio/
- **Azure Developer CLI:** https://learn.microsoft.com/azure/developer/azure-developer-cli/
- **Azure OpenAI Models:** https://learn.microsoft.com/azure/ai-services/openai/concepts/models
- **Azure AI Agent Service:** https://learn.microsoft.com/azure/ai-services/agents/

---

## 🔄 Common Workflows

## 🔄 Common Workflows

### Fresh Start
```bash
cd scripts/
./1-setup.sh
# Answer the questions, wait for deployment
```

### Add Models to Existing Setup
```bash
cd scripts/
./2-add-models.sh
# Type model numbers like: 1 3 5
# Or type: all
```

### Switch to High-Capacity Models
```bash
cd scripts/customization/
mv add-models.json add-models-low-capacity.json
mv add-models-high-capacity.json add-models.json
cd ..
./2-add-models.sh
```

### Complete Cleanup
```bash
cd scripts/
./3-teardown.sh
# Confirm when asked
```

### Start Over with Different Configuration
```bash
cd scripts/
./3-teardown.sh
./1-setup.sh
# This time, answer the questions differently
```

---

## 📚 Learn More

- **Azure AI Foundry:** [learn.microsoft.com/azure/ai-studio](https://learn.microsoft.com/azure/ai-studio/)
- **Azure Developer CLI:** [learn.microsoft.com/azure/developer/azure-developer-cli](https://learn.microsoft.com/azure/developer/azure-developer-cli/)
- **Azure OpenAI Models:** [learn.microsoft.com/azure/ai-services/openai/concepts/models](https://learn.microsoft.com/azure/ai-services/openai/concepts/models)

---

## 📝 License

This project follows the license of the parent repository.


---

## ⚙️ Script Development Notes

### For Script Developers

**Key principles:**
- Always use absolute paths or paths relative to `scripts/` directory
- Check prerequisites before proceeding
- Provide clear, color-coded output (GREEN=success, YELLOW=warning, RED=error, BLUE=info)
- Use `set -e` to exit on errors
- Navigate to correct directories before running `azd` commands
- Validate environment exists before accessing environment variables

**Testing scripts:**
```bash
# Syntax check
bash -n script-name.sh

# Dry run with debugging
bash -x script-name.sh
```

---

## 📝 License

This project follows the license of the parent repository.
