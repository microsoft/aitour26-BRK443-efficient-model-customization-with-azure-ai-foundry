---
title: "Speaker Notes"
marp: true
theme: default
paginate: true
backgroundColor: white
color: #0E1F2B
header: 'Track: Innovate with AI Apps & Agents'
footer: 'Session: Efficient Model Customization With Azure AI Foundry'
---



<!-- _paginate: skip -->
<!-- _color: white -->
![bg](bg.png)

##### INNOVATE WITH AI APPS & AGENTS
# BRK-000: Efficient Model Customization <br/> with Azure AI Foundry

Speaker One
Speaker Two

<br/>

Date · Location

<!-- 
Speaker Notes Here: 
-->
---



##### INNOVATE WITH AI APPS & AGENTS
# BRK-000: Efficient Model Customization <br/> with Azure AI Foundry


Nitya Narasimhan & Cedric Vidal
_Speaker Guidance_

<hr/>
<div>
<img width="15%" style="padding: 5px; margin: 5px;" src="https://github.com/nitya.png">
<img width="15%" style="padding: 5px; margin: 5px;" src="https://github.com/cedricvidal.png">
</div>

<!-- 
Speaker Notes Here: 
-->
---


<!-- _backgroundColor: #605CF3-->
<!-- _color: white-->
<!-- _header: ""-->
<!-- _footer: ""-->

# 1: Unlock Business Value

1. Define Scenario - Cora, The Zava Shopper Assistant
1. Define Requirements - 
1. Model Customization - What does this mean & why is it relevant?
1. Why Fine Tuning - How can I reach my objectives with this option?

<!-- 
Speaker Notes Here: 
-->
---

## 1.1 Set The Stage

- Zava is an enterprise retail organization for DIY project enthusiasts.
- Cora is their personalized shopping assistant for online & in-store use
- Bruno is a DIY enthusiast interested in repainting his living room
- Amira is a Zava store manager who wants to drive customer growth & revenue

<!-- 
Speaker Notes Here: 
-->
---

## 1.2 Define Business Goals

Business goals help us understand what outcomes our solution should try to maximize with the shopper assistant. For instance:

1. _Drive Sales_ - Cora can help shoppers find & purchase products faster
1. _Move Inventory_ - Cora can price-match alternatives if item is out of stock
1. _Build Loyalty_ - Cora can offer personalized discounts to retain customers


<!-- 
Speaker Notes Here: 
-->
---

## 1.3 Define Engineering Goals

<!-- 
Speaker Notes Here: 
-->

Engineering g


- Behavior - Cora must be polite & helpful, and deliver safe, relevant responses
- Cost - Optimize for cost (tokens) and performance (latency) in deployment
- 

---

## 1.5 Motivate Fine Tuning

<!-- 
Speaker Notes Here: 
-->
---

## 1.1 Set The Stage

<!-- 
Speaker Notes Here: 
-->
---

<!-- _backgroundColor: #605CF3-->
<!-- _color: white-->
<!-- _header: ""-->
<!-- _footer: ""-->

# 2: Know Customization Options

1. Set The Stage - Fine Tune for Behavior, Cost, Precision
1. Start with SFT - Understand workflow, Implement behavior
1. Try Distillation - Transfer intelligence (reuse data), Reduce cost
1. Understand RFT - Rewards-based (less data), Improve reasoning
1. Motivate Hybrid - Use RAG with FT, Improve response quality

<!-- 
Speaker Notes Here: 
-->
---


<!-- _backgroundColor: #605CF3-->
<!-- _color: white-->
<!-- _header: ""-->
<!-- _footer: ""-->

# 3: Hands-on With RAFT

1. Set The Stage - What is the desired objective? (Precision)
1. Understand RAFT - How does RAG with FT work in practice?
1. Prepare The Data - Understand oracle documents vs. distractors
1. Run Fine-Tuning Job - Execute the workflow in Azure AI Foundry
1. Validate Results - See how RAFT improves precision of responses

---

<!-- _backgroundColor: #050A39 -->
<!-- _color: white-->
<!-- _header: ""-->
<!-- _footer: ""-->

# 4: Summary & Next Steps

1. Why Azure AI Foundry - Unified Platform for seamless e2e experience
1. Why Fine Tuning - Critical skill to know, Richer options for optimization
1. What's New in Azure AI - Developer Tier, Global Training, Observability for FT
1. What's Next For You - Join The Community, Explore The Samples

---

## 1. Set The Stage

**Zava** is a retail enterprise selling home improvement products to DIY enthusiasts through physical and online stores. Cora is their AI-based Shopping Assistant, designed to assist customers find products and complete purchases online and in-store.


#### Feature Requirements:

1. **Persona** - Cora should be helpful and polite, and provide accurate information
1. **Sales** - Cora should offer price-matched alternatives for out-of-stock items
1. **Loyalty** - Cora should offer personalized loyalty-based discounts at checkout


---

## 2. Set The Business Goals


---

## The Session

In this session, you will learn how to customize AI models and optimize their performance (for your scenario) using targeted fine-tuning methods. 

- Discussion focuses on Distillation, RFT and RAFT approaches using Azure AI Foundry
- Demos show how you can reduce costs & improve precision with less data and complexity.

---

## Learning Objectives

By the end of this session, you should be able to:

1. Understand how model customization helps meet cost & performance objectives
1. Know the model customization options (Prompt Engineering, RAG, Fine-Tuning)
1. Explain how Fine-Tuning works, and its benefits & tradeoffs for customization
1. Describe Fine-Tuning options available in Azure AI Foundry (SFT, RFT & more)
1. Use Distillation (model compression) in Azure AI Foundry, to reduce cost
1. Use RAFT (hybrid RAG + Fine Tuning) in Azure AI Foundry, to improve precision


---

<!-- _backgroundColor:  -->

# Resources & Next Steps


---