<p align="center">
<img src="img/Banner-MS-AI-Tour-26.png" alt="decorative banner" width="1200"/>
</p>

# [Microsoft AI Tour 2026](https://aitour.microsoft.com)

## 🔥BRKXXX: The Best Way to Make Your Customized Models Smarter Without More Data

If you will be delivering this session, check the [session-delivery-sources](./session-delivery-resources/) folder for slides, scripts, and other resources.

### Session Description

You're building an enterprise retail chatbot but you need to customize the model with your data, and get high quality responses while keeping cost and complexity low. Do you use RAG, Fine-tuning - or both? Join us as we take you on a journey into Fine-Tuning with Azure AI Foundry and show you how support for lightweight processes, faster models, and the new developer tier, help streamline and simplify your end-to-end journey. Learn how fine-tuning approaches like Distillation can make your retail chatbot work on smaller, cheaper models. And how hybrid approaches like Retrieval Augmented Fine Tuning (RAFT) can help you improve the precision of your retail chatbot responses. Walk away with links to resources that you can explore for hands-on learning.


### 🧠 Learning Outcomes

By the end of this session, you should be able to:

1. Describe what fine-tuning is, and the benefits it offers for customization
1. Understand when to use different fine-tuning approaches, and why
1. Know how Distillation works, and how it improves your cost-effectiveness
1. Know how Retrieval-Augmented Fine Tuning works, and how it improves precision
1. Build & evaluate a RAFT-based variant of an Azure AI Foundry model


### 💻 Technologies Used

1. [Azure AI Foundry](https://learn.microsoft.com/en-us/azure/ai-foundry/quickstarts/get-started-code)
1. [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/overview/)
1. [Distillation (Fine-Tuning)](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/stored-completions#distillation)
1. [Reinforcement Fine-Tuning](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reinforcement-fine-tuning)
1. [RAFT (Retrieval Augmented Fine-Tuning)](https://github.com/Azure-Samples/azureai-foundry-finetuning-raft)
1. [Azure AI Evaluation SDK](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-evaluation-readme?view=azure-python)

---


### 🔗 Session Resources
| Resources          | Links                             | Description        |
|:-------------------|:----------------------------------|:-------------------|
| Fine Tuning in Azure AI Foundry | https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/fine-tuning-overview | Comprehensive guide to fine-tuning models with training techniques and best practices |
| Fine Tune OpenAI Models | https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/fine-tuning | Step-by-step guide to fine-tuning Azure OpenAI models using the studio and Python SDK |
| Reinforcement Fine Tuning | https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reinforcement-fine-tuning | Learn reinforcement fine-tuning (RFT) with o4-mini for complex reasoning tasks |
| Stored Completions and Distillation | https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/stored-completions | Use stored completions to create fine-tuning datasets through distillation |
| Safety Evaluation for Fine-Tuning | https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/fine-tuning-safety-evaluation | Automated safety evaluation for fine-tuned models to detect and prevent harmful content |
| Azure OpenAI Graders | https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/evaluation-evaluators/azure-openai-graders | Custom evaluation graders for model scoring, labeling, and text similarity assessment |

---
 
### 📚 Continued Learning Resources
| Resources          | Links                             | Description        |
|:-------------------|:----------------------------------|:-------------------|
| AI Tour 2026 Resource Center | https://aka.ms/AITour26-Resource-center | Links to all repos for AI Tour 26 Sessions |
| Azure AI Foundry Community Discord | [![Microsoft Azure AI Foundry Discord](https://dcbadge.limes.pink/api/server/ByRwuEEgH4)](https://discord.com/invite/ByRwuEEgH4)| Connect with the Azure AI Foundry Community! |
| Learn at AI Tour | https://aka.ms/LearnAtAITour | Continue learning on Microsoft Learn |

---

### 🌐 Multi-Language Support

*languages will go here when its time to localize*


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
