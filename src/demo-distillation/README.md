# Fine Tuning & Distillation: Setup

_This document provides guidance for setting up and running a demo for distillation. To do this we will start with a basic fine tuning example (on an LLM) then show how distillation allows for "model compression", giving us the same effective response quality but using smaller, cheaper models._

---

## 1. Environment Variables

The `.env.sample` file shows the environment variables expected for all FT demos. To get started, copy this to `.env` and fill in the values. This is what the sample file looks like. 

```bash# Azure Open AI Config
AZURE_OPENAI_API_KEY=""
AZURE_OPENAI_ENDPOINT=""
AZURE_OPENAI_API_VERSION="2025-02-01-preview"  

# Model Options
GPT41_MODEL="gpt-4.1"
GPT41_MINI_MODEL="gpt-4.1-mini"
GPT4o_MINI_MODEL="gpt-4o-mini-2024-07-18"

# Model Selections
DEMO_BASIC_MODEL="gpt-4.1"
DEMO_DISTILL_MODEL="gpt-4.1-mini"
```

Use the _Project Overview_ tab of the Azure AI Foundry project to fill in the required key and endpoint values for now. Later, we'll refactor code to use Entra ID. Then make your selections for the basic Fine Tuning and Distillation demos. 

---

## 2. Model Deployments

For the Distillation demo, spin up base model deployments for all models you want to work with. The notebook recommends the following:
- o3, o4-mini
- gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
- gpt-4o, and gpt-4o-mini

You will need to verify you have quota for these deployments.

---

## 3. Zava Dataset Generation

We'll use GitHub Copilot Agent Mode with a custom prompt to generate the Zava dataset from a previously extracted _products.json_ file.

```bash
Study the format of the records in the qa.jsonl file
Now study the products in the zava-products.json file

- You are a shopping agent for a DIY hardware store that sells related products. 
- Think deeply about the kinds of questions a DIY project enthusiast might have about these products 
- Think especially in the context of HOME IMPROVEMENT PROJECTS THEY ARE DOING (e.g., paining the living room walls)
- Come up with a similar file called zava_qa.jsonl that has 500 samples
- Make sure that the answers here are factual and precise
- Try not to duplicate questions or answers.
```