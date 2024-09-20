# Databricks notebook source
# MAGIC %md
# MAGIC # Inference for Fine-Tuned LLM on Retail Banking QA Pairs
# MAGIC
# MAGIC Welcome to this notebook! We will show how to perform inference using the Mistral-7B-Instruct model and a fine-tuned version of the same model. The fine-tuned model, `bitext/Mistral-7B-Retail-Banking-v1`, has been trained with the "Retail Banking QA Pairs for LLM Conversational Fine-Tuning" dataset available on Databricks Marketplace.
# MAGIC
# MAGIC Our fine-tuned model, unlike the base Mistral-7B-Instruct model, has been specifically trained to understand and respond to queries in the retail banking sector. 
# MAGIC
# MAGIC ## Dataset Description
# MAGIC
# MAGIC The "Retail Banking QA Pairs for LLM Conversational Fine-Tuning" dataset, used for fine-tuning our model, is designed to train Large Language Models (LLMs) like GPT, Llama3, and Mistral. Here are the details:
# MAGIC
# MAGIC - **Use Case**: Customer Service
# MAGIC - **Vertical**: Retail Banking
# MAGIC - **Intents**: 26 intents in 9 categories
# MAGIC - **Pairs**: 25,545 question/answer pairs
# MAGIC - **Entities/Slots**: 1,224 entity/slot types
# MAGIC - **Language Tags**: 12 types
# MAGIC - **Token Count**: 4.98 million tokens in 'instruction' and 'response'
# MAGIC
# MAGIC This dataset is perfect for training models to understand and respond to common retail banking queries.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting Up the Environment
# MAGIC
# MAGIC Before we begin, ensure you have the necessary libraries installed: `transformers` and `torch` for model inference.

# COMMAND ----------

!pip install transformers torch

# COMMAND ----------

# MAGIC %md
# MAGIC Next, restart the Python kernel to apply the changes.

# COMMAND ----------

# Restart kernel
%restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Login to Hugging Face
# MAGIC
# MAGIC Log in to Hugging Face to access the models. Generate an Access Token on Hugging Face with Read permissions and use it in the code below.

# COMMAND ----------

from huggingface_hub import login

# Replace 'your_token_here' with your actual Hugging Face token
login(token='your_token_here', add_to_git_credential=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference with Fine-Tuned Mistral 7B for Retail Banking
# MAGIC
# MAGIC Let's start with our fine-tuned model, `bitext/Mistral-7B-Retail-Banking-v1`, which has been trained specifically for retail banking customer service. Use a sample conversation to see how the model performs.

# COMMAND ----------

from transformers import pipeline

messages = [
    {"role": "user", "content": "I want to open an account"},
    {"role": "assistant", "content": "Sure, I can help you with that. Can you please provide your full name and address?"},
    {"role": "user", "content": "My name is John Doe and my address is 123 Main St."}
]
chatbot = pipeline("text-generation", model="bitext/Mistral-7B-Retail-Banking-v1")
response = chatbot(messages, max_new_tokens=100)
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cleaning Up Resources
# MAGIC
# MAGIC Before moving to our fine-tuned model, let's clean up the GPU resources and restart the Python kernel to ensure optimal performance.

# COMMAND ----------

import torch

# Restart kernel
%restart_python

# Clear CUDA cache
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference with Mistral 7B Instruct (Base Model)
# MAGIC
# MAGIC Now, use the base model, `mistralai/Mistral-7B-Instruct-v0.2`.

# COMMAND ----------

from transformers import pipeline

messages = [
    {"role": "user", "content": "I want to open an account"},
    {"role": "assistant", "content": "Sure, I can help you with that. Can you please provide your full name and address?"},
    {"role": "user", "content": "My name is John Doe and my address is 123 Main St."}
]
chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")
response = chatbot(messages, max_new_tokens=100)
print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC Compare the response with the base model to see the improvements.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tips for Fine-Tuning Mistral 7B
# MAGIC
# MAGIC To fine-tune the Mistral 7B model, follow these key steps and recommendations from the [Mistral 7B paper](https://arxiv.org/pdf/2310.06825) and our experience:
# MAGIC
# MAGIC 1. **Model Parameters**:
# MAGIC    - **dim**: 4096
# MAGIC    - **n_layers**: 32
# MAGIC    - **head_dim**: 128
# MAGIC    - **hidden_dim**: 14336
# MAGIC    - **n_heads**: 32
# MAGIC    - **n_kv_heads**: 8
# MAGIC    - **window_size**: 4096
# MAGIC    - **context_len**: 8192
# MAGIC    - **vocab_size**: 32000
# MAGIC
# MAGIC 2. **Sliding Window Attention**:
# MAGIC    - Use stacked Transformer layers to attend to information beyond the window size (4096 tokens).
# MAGIC
# MAGIC 3. **Rolling Buffer Cache**:
# MAGIC    - Maintain only the most recent tokens within the window size limit to manage context efficiently.
# MAGIC
# MAGIC 4. **Prefill and Chunking**:
# MAGIC    - Prefill the cache with known prompts and divide large prompts into smaller chunks.
# MAGIC
# MAGIC 5. **Content Moderation**:
# MAGIC    - Use self-reflection prompts to ensure safe and acceptable content generation.
# MAGIC
# MAGIC 6. **Comparison with Other Models**:
# MAGIC    - Mistral 7B is comparable to larger models like Llama 2 in efficiency and performance.
# MAGIC
# MAGIC ### Hyperparameters Used for Fine-Tuning
# MAGIC
# MAGIC - **Optimizer**: AdamW
# MAGIC - **Learning Rate**: 0.0002 with a cosine scheduler
# MAGIC - **Epochs**: 1
# MAGIC - **Batch Size**: 16
# MAGIC - **Gradient Accumulation Steps**: 16
# MAGIC - **Maximum Sequence Length**: 1024 tokens
# MAGIC
# MAGIC These settings help achieve efficient and effective fine-tuning.
