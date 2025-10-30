# CHANGELOG

## Oct 29, 2025

Refactoring repository with updates for Nov 6 delivery. The breakout has two main "categories" for demos:

- Core Fine-Tuning = SFT & Distillation
- Hybrid Fine-Tuning + RAG = RAFT

This commit refactors the repo to remove unused content and create self-contained demo subfolders for easy maintenance:

**CORE FINE-TUNING** (`src/demo-core`)
- `data/` has all data required for these demos
- `0-custom-grader.ipynb` - eval-driven development
- `1-getting-started.ipynb` - context engineering
- `2-demo-sft.ipynb` - covers SFT
- `3-demo-distillation.ipynb` - covers distillation

**HYBRID FINE-TUNING** (`src/demo-raft`)
- `scripts/` was originally in root of repo
- `data/articles` was originally in root of repo
- everything else remains unchanged from previous

INFRA FOR **CORE DEMOS**

- The repository uses a custom branch of the [_ForBeginners AZD Template_](https://github.com/microsoft/ForBeginners/tree/aitour26-demos) that is based on the Get-Started-With-AI-Agents template for Azure AI Foundry.
    - This will set up a Azure resource group with an Azure AI Foundry resource and project that has {An AI Agent, A Chat Model, An Embedding Model, A Web App, An AI Search resource, Tracing & Monitoring activated}
    - Use the `infra/2-add-models.sh` script to deploy additional models after the initial provisioning step. 
- Pick the default base model for fine-tuning (default is gpt-4.1, but you may want gpt-4o)
- Pick a region that has support for [Fine-tuning](https://learn.microsoft.com/azure/ai-foundry/foundry-models/concepts/models-sold-directly-by-azure?pivots=azure-openai&tabs=global-standard-aoai%2Cstandard-chat-completions%2Cglobal-standard#fine-tuning-models) and [Azure AI Search](https://learn.microsoft.com/azure/search/search-region-support#americas).
    - Default is Sweden Central (base is gpt-4.1)
    - Backup is East US 2 (switch base to gpt-4o)
- **SETUP**:
    - `cd infra/`
    - `./1-setup.sh`
    - `./2-add-models.sh`
- **TEARDOWN**:
    - `cd infra/`
    - `./3-teardown.sh`

---
