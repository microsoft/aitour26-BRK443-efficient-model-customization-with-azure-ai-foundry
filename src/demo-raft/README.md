# Act 3: RAFT Finetuning on<sub><img src="./doc/azure-ai-foundry.png" width="65"></sub>Azure AI Foundry

This repository is a demo that will walk you through improving **Cora's** RAG chat bot precision using UC Berkeley's RAFT technique on Azure AI Foundry. **Cora** is the AI assistant for **Zava**, an enterprise DIY hardware store that helps customers like Bruno find the right products, enables store managers like Robin to drive customer loyalty and sales, and allows app developers like Kian to build cost-effective solutions.

RAFT (Retrieval Augmented Fine-Tuning) is a method that fine-tunes language models to better understand and utilize retrieved context for more accurate responses. In Zava's case, this means helping Cora provide more accurate DIY advice and recommendations based on their comprehensive knowledge base blog covering topics like paint techniques, tool usage, and home improvement projects.

This demo uses either [OpenAI GPT-4.1](https://azure.microsoft.com/en-us/blog/announcing-the-gpt-4-1-model-series-for-azure-ai-foundry-developers/) as a teacher model deployed on [Azure AI](https://aka.ms/c/learn-ai) to generate a synthetic dataset using [UC Berkeley's Gorilla](https://aka.ms/ucb-gorilla) project RAFT method (see [blog post](https://aka.ms/raft-blog)). The synthetically generated dataset will then be used to fine-tune a student model such as OpenAI GPT-4o-mini to improve Cora's RAG capabilities for answering questions based on Zava's knowledge base blog. Finally, we will deploy the fine-tuned model and evaluate its performance compared to a baseline model.

> **Note**: While this recipe involves using a larger model to generate training data for a smaller model (a form of distillation), the primary focus is on improving RAG system precision through RAFT fine-tuning rather than general model distillation.

<table>
    <tr>
        <td><img src="./doc/microsoft-logo.png" style="max-height:100px; height: auto;"/></td>
        <td><img src="./doc/openai-logo.png" style="max-height:100px; height: auto;"/></td>
        <td><img src="./doc/meta-logo.png" style="max-height:100px; height: auto;" /></td>
        <td><img src="./doc/ucb-logo.png" style="max-height:100px; height: auto;" /></td>
    </tr>
</table>


## More about RAFT

- [Microsoft/Meta Blog post](https://aka.ms/raft-blog): RAFT:  A new way to teach LLMs to be better at RAG
- [Paper](https://aka.ms/raft-paper): RAFT: Adapting Language Model to Domain Specific RAG
- [UC Berkeley blog post](https://aka.ms/raft-blog-ucb): RAFT: Adapting Language Model to Domain Specific RAG
- [Meta blog post](https://aka.ms/raft-blog-meta): RAFT: Sailing Llama towards better domain-specific RAG
- [Gorilla project home](https://aka.ms/gorilla-home): Large Language Model Connected with Massive APIs
- [RAFT Github project](https://aka.ms/raft-repo)

## Getting started / Provisioning Azure AI infrastructure

The infrastructure for this project is fully provisioned using the Azure Developer CLI ([AZD](https://aka.ms/c/learn/azd)). AZD simplifies the deployment process by automating the setup of all required Azure resources, ensuring that you can get started with minimal configuration. This approach allows you to focus on the core aspects of RAFT fine-tuning for RAG improvement, while AZD handles the complexities of cloud resource management behind the scenes. By leveraging AZD, the project maintains a consistent and reproducible environment, making it easier to collaborate and scale.

The easiest is to open the project in Codespaces (or in VS Code Dev Container locally). It comes with azd included.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Azure-Samples/raft-distillation-recipe)

[![Open in Dev Containers](https://img.shields.io/static/v1?style=for-the-badge&label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/Azure-Samples/raft-distillation-recipe)

### Login using azd

```
azd auth login --use-device-code
```

### Create azd environment

This creates a new azd environment and is a pre-requisite to configuring models in the next step.

```
azd env new
```

### Configure models & region

Configure which **models** you want to use for `teacher`, `student`, `embedding` and `baseline` (`baseline` usually equals `student`) as well as which **region** to deploy the project to.

> **Note**: Both OpenAI models and Azure Marketplace models are supported.


If in Codespaces or Dev Container:

```bash
python raft.py configure
```


# RAFT Demo – AI Tour 2026

This folder contains a demonstration of the RAFT (Retrieval Augmented Fine Tuning) methodology for improving **Cora**, the AI assistant for **Zava DIY store**, presented at Microsoft AI Tour 2026.

## What is RAFT?

RAFT is a toolkit for efficient model customization using Azure AI services. In the context of Zava's DIY hardware store, it enables:
- Generation of synthetic datasets based on Zava's knowledge base blog and DIY expertise articles
- Fine-tuning of models to better understand DIY questions and provide accurate guidance from the knowledge base  
- Deployment and evaluation of model performance for improved customer service

The RAFT workflow is designed to help Cora, Zava's AI assistant, provide more accurate DIY advice and technical guidance to customers by leveraging their extensive knowledge base blog covering paint techniques, tool usage, home improvement projects, and expert DIY tips.

## RAFT CLI Overview (`raft.py`)

The main entry point for this demo is the `raft.py` CLI. It provides a comprehensive set of commands for running the RAFT workflow:

### Quick Start

Run the complete workflow in one command:

```bash
python raft.py run
```

### Step-by-Step Workflow

1. `configure` – Configure AI models and deployments for RAFT workflows with Zava's requirements
2. `check` – Verify Azure AI endpoints and connectivity for Zava's infrastructure
3. `gen` – Generate synthetic training datasets based on Zava's knowledge base blog and DIY expertise articles
4. `finetune` – Fine-tune models with generated data to improve Cora's DIY knowledge and advice capabilities
5. `deploy` – Deploy fine-tuned models to Azure OpenAI for Zava's production environment
6. `eval` – Evaluate model performance and compare results against baseline Cora responses on DIY knowledge queries
7. `status` – Monitor progress and results of Cora's improvement process
8. `clean` – Clean up generated datasets and temporary files

### Utility & Interactive Commands

- `chat` – Start an interactive chat with Cora using a LangChain model to test DIY knowledge assistance

For more details on each command, run:

```bash
python raft.py --help
```

## RAFT Process Diagram

![RAFT Process](raft-process-eval.png)


## Azure Deployment

To provision the required Azure resources for this demo, use the Azure Developer CLI:

```bash
azd up
```

This command will set up the necessary infrastructure for running the RAFT workflow with Azure AI services.

## Run time and costs

**Warning**: The times and costs mentioned bellow are indications to give you a sense of what to expect but can vary dramatically depending on your experience, please monitor your usage to avoid surprises.

## Dormant infrastructure costs

While not used, the infrastructure of this project won't cost much but will still cost a bit.

**TODO**: provide costs estimations for dormant infra

## Configuration files

| File      | Explanation      |
| ------------- | ---------------- |
| [.env](./.env) | User provided environment variables read by notebooks and scripts |
| [.env.state](./.env.state) | Environment variables for resources created during notebooks execution and shared by all notebooks |
| [config.json](./config.json) | Configuration necessary to connect to the Azure AI Studio Hub (same as Azure ML Workspace) |

## Taking down the infrastructure

After you are done working with the project, you can take down the infrastructure with the following command.

**IMPORTANT**: Please be aware that this will **DELETE** everything related to Zava's RAFT knowledge base project including **generated datasets** and **fine-tuned Cora models**.

**IMPORTANT**: Save everything important to you before running this command.

```
azd down --purge
```

**Note**: The `--purge` parameter is important to reclaim quotas, for example for Azure OpenAI embedding models.
