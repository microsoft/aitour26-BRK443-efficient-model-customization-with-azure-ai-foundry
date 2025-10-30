# Act 2: Fine Tuning Options in Azure AI Foundry

## 1. Setup Environment

This README documents the steps required to setup and run the demos for Act 2. By this time you should have setup infrastructure as follows:

1. Use the Azure CLI to log into Azure from VS Code terminal (create credential)
1. Run the script to create an Azure AI Foundry project with Azure AI Search resource
1. (Optional) Manually add an App Insights resource via thr `Tracing` tab
1. Create the `.env` file and update it with relevant environment variables
1. Run the script to update role access permissions for updating the search index
1. Manually add a `text-embedding-ada-002` model to support index creation
1. Run the notebook to populate Zava data in the Azure AI Search index

<br/>

## 2. Run Notebook: `00-getting-started.ipynb`

1. Deploy additinal AI models in the Azure AI Foundry (manually)
1. Run the notebook to get a sense for latency/token cost for a basic prompt
1. **Select GPT-4.1 AS BASE MODEL FOR NOW** - we will customize this, next

<br/>

## 3. Customize - With Few Shot Prompts

We want to modify the tone & style of Cora's responses to user questions. Let's see how far we can get with prompt engineering. For convenience, we'll use the Azure AI Foundry playground.

1. Visit Azure AI Foundry Portal (https://ai.azure.com)
1. Select the base model `gpt-4.1` and click _Open in playground_.
1. Set system message to _You are Cora, a polite, factual and helpful assistant for Zava, a DIY hardware store._ - then click **Apply Changes** to enforce it.
1. Try a sample prompt: 
    ```bash
    I need paint that's safe for my nursery. Any suggestions?
    ```
1. You may see some response like this. Since responses are stochastic, your exact response may look different. However, note that it it is _NOT_ grounded in Zava data, and it does _NOT_ have a "polite, helpful and factual" response tone and style. **Note the base token usage here - 0 few-shot tokens, 200+ output tokens**

    ```bash
    Absolutely! For a nursery, it’s important to choose a paint that is low in volatile organic compounds (VOCs) or completely VOC-free, as these paints emit fewer chemicals and are safer for babies and pregnant women. Here are some recommendations:

    Look for:

    Low-VOC or Zero-VOC Paints: These release fewer fumes and are widely available.
    Water-Based (Latex) Paints: Easier to clean up and generally emit fewer odors.
    Paints Labeled as “Non-Toxic” or “Baby Safe”: Some brands specifically market nursery-safe paints.
    Popular brands to consider:

    Dulux Easycare Nursery (specifically designed for nurseries, low odor and washable)
    Johnstone’s Eco-Friendly Range
    Earthborn Claypaint (VOC-free, breathable, natural ingredients)
    Little Greene Intelligent Matt Emulsion (low VOC, child-safe certification)
    Tips:

    Always allow the room to ventilate well after painting.
    Wait several days after painting before bringing your baby into the room, even with low-VOC products.
    Would you like help finding these products in our store, or do you need color recommendations as well?
    ```
1. What Zava wants is a response with these elements: _[Emoji] [Acknowledgment]! [Product recommendation with price]. [Follow-up offer]?_ - where the emoji and acknowledgement reflect the politeness of the brand, the Product recommendation reflects factual information from the catalog, and the Follow-up offer reflects the helpfulness.  **A good response to the previous question may look like this:**

    ```bash
    🎨 Yes! Zero VOC Interior Paint at $52 ensures healthy air quality. Want color samples?
    ```

1. Let's see if we can achieve this with few-shot examples (prompt-engineering) first. I used GitHub Copilot Agent Mode (with Claude Sonnet 3.7) to generate a set of "few-shot prompt examples" that we can use for customizing chatbot tone _through prompt engineering_.

    > This is the PROMPT used with Copilot
    ```bash
    I want you to generate a file called product-fewshot.jsonl

    - It should have the same format as data/generator/qa-fewshot.jsonl 
    - It should use questions based on products in data/zava/products-paints.csv
    - It should generate responses that follow format guidelines in sft_training_data.md
    - Make sure that neither the question nor the answer have anything harmful
    - Make sure there are no duplicates
    ```

1. Revisit the Playground and add some of these few-shot examples by clicking the `Add section` option - and choosing "Examples". You will need to click once for each example you add. Let's add four (to cover four different nuances in responses). **Don't forget to click Apply Changes**. _These are the four examples I added_.

    ```jsonl
    USER: Are paint tray liners available?
    ASSISTANT: 😞 Sorry, Paint Tray Liner Set is out of stock. Try Disposable Paint Tray Set instead?
    ```
    ```jsonl
    USER: Do you have any foam brushes for small touch-ups?
    ASSISTANT: 🖌️ Yes! Foam Brush Set at $9, perfect for touch-ups. Only 8 left in stock!
    ```
    ```jsonl
    USER: What should I use to paint my concrete patio?
    ASSISTANT: 🏠 Try Masonry Primer at $38 followed by Exterior Acrylic Paint. Planning this weekend?
    ```
    ```jsonl
    USER: Do you have paint for metal outdoor furniture? 
    ASSISTANT: 🎨 Try Metal Primer at $36 followed by Gloss Spray Paint. Color preference?
    ```
1. Now, let's try that sample prompt again.
    ```bash
    I need paint that's safe for my nursery. Any suggestions?
    ```

    We get a response like this:
    ```bash
    Absolutely! For a nursery, it's best to use a low-VOC or zero-VOC paint to ensure air quality and safety for your baby. Zava offers Eco-Friendly Interior Paint at $42 per can, which is low in VOCs, odorless, and safe for nurseries and children’s rooms. It comes in several gentle pastel shades.
    Would you like help picking a color or more details about the product?
    ```
1. Note that the response is _much_ better - it has a shorter length and has two elements (acknowledges question, answers with a product, offers to help). **But** it is still not perfect - it is too long, it forgot to add an emoji, the product is not real, and most importantly **few shot examples added a fixed overhead of 120+ tokens to each prompt**. We improved the tone but paid for the improved quality with added tokens and processing time.

<br/>

## 4. Customize - With RAG Context

1. We still need to solve the issue of _factual accuracy_. Let's see if we can "ground" responses in the Azure AI Search index to make sure the products references are valid Zava items.
1. Revisit the Playground - select the **Add your data** option to start a wizard workflow. Select "Azure AI Search" as the data source, find your existing Azure AI Search resource - and select the _zava-products_ index. Select "add vector search" - and pick the embedding model deployed in this project. Continue and select "Hybrid (vector+keyword) as the search type. And use the _API Key_ option for now to authenticate.
1. The playground has this limitation: _Few shot examples are not used when source data has been added. Any previous examples have been cleared. Once data is removed, then examples can be added._ - so for now, we can test if RAG improves the factual response but not get the benefit of modified tone. Let's try this out.

1. Now, let's try that sample prompt again.
    ```bash
    I need paint that's safe for my nursery. Any suggestions?
    ```

    We get a response like this:
    ```bash
    For a nursery, a safe option would be the Zero VOC Interior Paint, which is environmentally friendly and designed to maintain healthy indoor air quality in all living spaces [1].
    ```

    Where the citation [1] has the following context - from Zava product index!
    ```bash
    Zero VOC Interior Paint
    Environmentally friendly zero-VOC paint for healthy indoor air quality in all living spaces.
    ```

    This increases latency and adds more tokens (to carry context for prompt). Note that it is not enforcing few-shot examples so we don't get the right tone _but if we combined both, we would increase latency and prompt lengths even more - using up more tokens_.

    Can we customize tone _without increasing prompt length?_ and can we use a smaller, faster model _with comparable accuracy?_ **This is where Fine Tuning Comes in**

<br/>

## 5. Customize - Tone With Supervised Fine Tuning