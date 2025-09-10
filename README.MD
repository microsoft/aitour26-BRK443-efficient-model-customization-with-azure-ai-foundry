<p align="center">
<img src="img/Banner-MS-AI-Tour-26.png" alt="decorative banner" width="1200"/>
</p>

# [Microsoft AI Tour 2026](https://aitour.microsoft.com)

> Delivering this session? Check the [session-delivery-sources](./session-delivery-resources/) folder for resources & guidance.

## 🔥BRK443: Efficient Model Customization with Azure AI Foundry

[![Microsoft Azure AI Foundry Discord](https://dcbadge.limes.pink/api/server/ByRwuEEgH4)](https://discord.com/invite/ByRwuEEgH4)
[![Azure AI Foundry Developer Forum](https://img.shields.io/badge/GitHub-Azure_AI_Foundry_Developer_Forum-blue?style=for-the-badge&logo=github&color=adff2f&logoColor=fff)](https://aka.ms/foundry/forum)

### Session Description

You are an AI developer building a multi-agent AI solution for your enterprise retail store. You want to deliver a solution that is accurate, cost-effective, and performant. How do you select and optimize your model choice to deliver on these requirements? 

In this session, we put the spotlight on model customization. Learn what your options are, and why _fine-tuning_ might be the right option for your scenario. Learn about the various fine-tuning options in Azure AI Foundry, and build your intuition for using techniques like Distillation (for reducing cost) and RAFT (for improving precision) with hands-on demos. Walk away with resources and insights that help you streamline your end-to-end model optimization journey with Azure AI Foundry.


### 🧠 Learning Outcomes

By the end of this session, you should be able to:

1. Describe what model customization is, and the different techniques involved
1. Describe why fine-tuning matters, how it works, and when it is most applicable
1. Describe the different fine-tuning options in Azure AI Foundry
1. Use the Distillation technique to optimize costs with comparable accuracy
1. Use the RAFT technique to improve precision with hybrid RAG & Fine-Tuning
1. Build and optimize your AI application seamlessly, with Azure AI Foundry

### 💻 Technologies Used

1. [Azure AI Foundry](https://learn.microsoft.com/azure/ai-foundry/what-is-azure-ai-foundry) - the unified Azure platform-as-a-service for enterprise AI.
1. [Azure OpenAI Service](https://learn.microsoft.com/azure/ai-foundry/openai/overview/) - provide API access to OpenAI models in Azure AI Foundry
1. [Stored Completions & Distillation](https://learn.microsoft.com/azure/ai-foundry/openai/how-to/stored-completions#distillation) - fine-tuning to achieve model (size) compression
1. [Reinforcement Fine-Tuning](https://learn.microsoft.com/azure/ai-foundry/openai/how-to/reinforcement-fine-tuning) - fine-tuning reasoning models with rewards-based approach
1. [RAFT (Retrieval Augmented Fine-Tuning)](https://github.com/Azure-Samples/azureai-foundry-finetuning-raft) - Hybrid RAG+Fine-Tuning to improve precision

---


### 🔗 Session Resources
| Resources          | Links                             | Description        |
|:-------------------|:----------------------------------|:-------------------|
| Fine Tuning in Azure AI Foundry | https://aka.ms/aitour/fine-tuning/documentation | Introduction to Fine Tuning concepts & tools in Azure AI Foundry|
| Fine-tuning samples from Azure AI Foundry | https://aka.ms/aitour/fine-tuning/foundry-samples | Evolving set of open-source code samples for developers |
| Fine-tuning and distillation with Azure AI Foundry|https://aka.ms/aitour/fine-tuning/msbuild25-breakout | Microsoft Build 2025 Session on Model Customization |
 Fine-tuning and Distillation | https://aka.ms/aitour/fine-tuning/model-mondays-video | Hands-on Distillation demo on Azure AI Foundry  |
| Fine-tune a language model with Azure AI Foundry | https://aka.ms/aitour/fine-tuning/training-module | Microsoft Learn Training Module on Fine-Tuning |
| Fine-tune pre-trained models to your business needs with Azure AI Foundry |https://aka.ms/aitour/fine-tuning/whitepaper | (Requires signup) Whitepaper covering challenges, benefits etc.  |
| | | |

---
 
### 📚 Continued Learning Resources
| Resources          | Links                             | Description        |
|:-------------------|:----------------------------------|:-------------------|
| AI Tour 2026 Resource Center | https://aka.ms/AITour26-Resource-center | Links to all repos for AI Tour 26 Sessions |
| Azure AI Foundry Community Discord | [![Microsoft Azure AI Foundry Discord](https://dcbadge.limes.pink/api/server/ByRwuEEgH4)](https://discord.com/invite/ByRwuEEgH4)| Connect with the Azure AI Foundry Community! |
| Learn at AI Tour | https://aka.ms/LearnAtAITour | Continue learning on Microsoft Learn |

---

### 🌐 Multi-Language Support

Additional Languages Coming Soon


---

## Content Owners

<table>
<tr>
    <td align="center"><a href="https://github.com/nitya">
        <img src="https://github.com/nitya.png" width="100px;" alt="Nitya Narasimhan"/><br />
        <sub><b>Nitya Narasimhan</b></sub></a><br />
            <a href="https://github.com/nitya" title="talk">📢</a> 
    </td>
    <td align="center"><a href="https://github.com/cedricvidal">
        <img src="https://github.com/cedricvidal.png" width="100px;" alt="Cedric Vidal"/><br />
        <sub><b>Cedric Vidal</b></sub></a><br />
            <a href="https://github.com/cedricvidal" title="talk">📢</a> 
    </td>
</tr></table>

## 

## Responsible AI 

Microsoft is committed to helping our customers use our AI products responsibly, sharing our learnings, and building trust-based partnerships through tools like Transparency Notes and Impact Assessments. Many of these resources can be found at [https://aka.ms/RAI](https://aka.ms/RAI).
Microsoft’s approach to responsible AI is grounded in our AI principles of fairness, reliability and safety, privacy and security, inclusiveness, transparency, and accountability.

Large-scale natural language, image, and speech models - like the ones used in this sample - can potentially behave in ways that are unfair, unreliable, or offensive, in turn causing harms. Please consult the [Azure OpenAI service Transparency note](https://learn.microsoft.com/legal/cognitive-services/openai/transparency-note?tabs=text) to be informed about risks and limitations.

The recommended approach to mitigating these risks is to include a safety system in your architecture that can detect and prevent harmful behavior. [Azure AI Content Safety](https://learn.microsoft.com/azure/ai-services/content-safety/overview) provides an independent layer of protection, able to detect harmful user-generated and AI-generated content in applications and services. Azure AI Content Safety includes text and image APIs that allow you to detect material that is harmful. Within Azure AI Foundry portal, the Content Safety service allows you to view, explore and try out sample code for detecting harmful content across different modalities. The following [quickstart documentation](https://learn.microsoft.com/azure/ai-services/content-safety/quickstart-text?tabs=visual-studio%2Clinux&pivots=programming-language-rest) guides you through making requests to the service.

Another aspect to take into account is the overall application performance. With multi-modal and multi-models applications, we consider performance to mean that the system performs as you and your users expect, including not generating harmful outputs. It's important to assess the performance of your overall application using [Performance and Quality and Risk and Safety evaluators](https://learn.microsoft.com/azure/ai-studio/concepts/evaluation-metrics-built-in). You also have the ability to create and evaluate with [custom evaluators](https://learn.microsoft.com/azure/ai-studio/how-to/develop/evaluate-sdk#custom-evaluators).

You can evaluate your AI application in your development environment using the [Azure AI Evaluation SDK](https://microsoft.github.io/promptflow/index.html). Given either a test dataset or a target, your generative AI application generations are quantitatively measured with built-in evaluators or custom evaluators of your choice. To get started with the azure ai evaluation sdk to evaluate your system, you can follow the [quickstart guide](https://learn.microsoft.com/azure/ai-studio/how-to/develop/flow-evaluate-sdk). Once you execute an evaluation run, you can [visualize the results in Azure AI Foundry portal ](https://learn.microsoft.com/azure/ai-studio/how-to/evaluate-flow-results).
