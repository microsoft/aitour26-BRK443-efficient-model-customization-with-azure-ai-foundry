# Act 2: Try Model Customization

## What is Azure AI Foundry?

1. Unified plaform for e2e fine-tuning workflows
1. Flexible deployment: Serverless or Managed compute
1. Rich model choices: Azure OpenAI, Meta, Mistral, Phi etc.
1. Streamlined DX: Lighter workflows, faster models, developer tier

## Fine Tuning In Azure AI Foundry

1. Diverse FT options: SFT, RFT, DPO, Distillation, RAFT etc.
1. Model behavior adaptation => Start with Supervised Fine-Tuning (LoRA)
1. Model compression => Try Distillation with Graders 
1. Optimize reasoning => Try Rewards-based Reinforcement FT
1. Improve precision => Try Hybrid RAG+FT approaches (e.g., RAFT)

Good decision tree from [this post](https://gradientflow.com/post-training-rft-sft-rlhf/)

![Good decision tree](https://i0.wp.com/gradientflow.com/wp-content/uploads/2025/03/RFT-or-SFT-or-RLHF.png?resize=768%2C270&ssl=1) 


## Distillation Demo: For Zava

**How can I reduce model cost without losing accuracy?**

Distillation Workflow
    - Benchmark Models: Test base models with Zava test data
    - Pick Teacher: Identify model that performed the best
    - Pick Student: Identify model that was demonstrably bad
    - Run Distillation: Have teacher "distill" knowledge to student
    - Re-run Benchmarks: Test teacher, student & one control model

What you should see:
- Teacher & Control remain at previous levels 
- Student shows vast improvement, almost Teacher level
- Student is much smaller, faster, and cheaper than Teacher

Why it matters:
- Smaller models have lower token costs, respond faster
- Fine-tuning frees up token window for longer prompts, responses
- Distillation is perfect for task-focused agentic architectures

Distillation demo:
- Challenge: **Cora with GPT-4.1 is polite, helpful - costly & slow**
- I have access to GPT-4.1, GPT-4o and o series of models
- o3 model has best accuracy and highest latency/cost => Teacher
- gpt4.1-nano has worst accuracy but is cheap/fast => Student
- gpt4o-mini as a control => Middle of the pack
- Result: **Cora with GPT4.1-nano is polite, helpful - cheaper & faster**


## Reinforcement Fine Tuning: For Reasoning

What is it?
- Reinforcement learning - models learn by trial-and-error using rewards
- RFT use graders - sample responses & grade them, then use as training

Why is it relevant?
- Perfect for Reasoning models - helps develop or refine Chain-of-Thought
- Perfect for Data-constrained use - can be fine-tuned with < 100 examples
- RFT models can be distilled later - get model compression with accuracy
- RFT now has built-in observability - "auto-evals" dashboard on checkpoints

How do I use it?

- Scenario: [Countdown dataset from Predibase](https://github.com/azure-ai-foundry/fine-tuning/tree/main/Demos/RFT_Countdown) 
    - Task: Given 4 numbers, find expression that comes close to 5th number.
    - Challenge: o3 (base model) is costly! (100 training/50 validation data)
    - Solution: RFT with o4-mini (ft-model), o3-mini (model-based grader)

Where can I use it?
- See [OpenAI RFT Use cases](https://platform.openai.com/docs/guides/rft-use-cases#when-to-use-reinforcement-fine-tuning) guidance - instructions into working code, messy facts into structured data, apply complex rules correctly.
- Criteria: Decisions must be CORRECT (reasoning accuracy) and VERIFIABLE (grader evaluations)
- Zava Use Case 
    - 1: Apply complex rules correctly - for loyalty discounts
    - 2: Extract messy facts into structured data - for recommendations


## Other Fine Tuning Options

- Vision Fine Tuning
- Direct Preference Optimization
- Hybrid (RAG+FT) Approaches

Let's look at RAFT as one example, next.