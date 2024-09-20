# Databricks notebook source
# MAGIC %md
# MAGIC # Fine tune Bitext's `Mistral-7B for Retail Banking` model from marketplace with QLORA
# MAGIC
# MAGIC This is a tutorial on how to fine tune Bitext's `Mistral-7B for Retail Banking` model that is hosted in [Databricks marketplace](https://marketplace.databricks.com/). This model is originally tailored for the Banking domain.
# MAGIC
# MAGIC The goal of this demo is to show that a generic verticalized model makes customization for a final use case much easier. For example, if you are "ACME Bank", you can create your own customized model by using this model and a doing an additional fine-tuning using a small amount of your own data. An overview of this approach can be found at: [From General-Purpose LLMs to Verticalized Enterprise Models](https://www.bitext.com/blog/general-purpose-models-verticalized-enterprise-genai/)
# MAGIC
# MAGIC We will fine-tune the original Retail Banking model on a sample dataset from the mortgage and loans domain, that was taken from [bitext/Bitext-mortgage-loans-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-mortgage-loans-llm-chatbot-training-dataset). Then, we will show some test examples.
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 15.4 GPU ML Runtime
# MAGIC - Instance: `g5.8xlarge` on AWS, `Standard_NV36ads_A10_v5` on Azure, `g2-standard-8` or `a2-highgpu-1g` on GCP
# MAGIC
# MAGIC We leverage the PEFT library from Hugging Face, as well as QLoRA for more memory efficient fine tuning.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install required packages
# MAGIC
# MAGIC Run the cells below to setup and install the required libraries. For our experiment we will need `accelerate`, `peft`, `transformers`, `datasets` and TRL to leverage the recent [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer). We will use `bitsandbytes` to [quantize the base model into 4bit](https://huggingface.co/blog/4bit-transformers-bitsandbytes).

# COMMAND ----------

# To access models in Unity Catalog, ensure that MLflow is up to date
%pip install --upgrade "mlflow-skinny[databricks]==2.16.0"
%pip install peft==0.12.0
%pip install datasets==2.19.1 bitsandbytes==0.43.3 einops==0.8.0 trl==0.10.1
%pip install accelerate==0.31.0 transformers==4.41.2
dbutils.library.restartPython()

# COMMAND ----------

import mlflow

# Set mlflow registry to databricks-uc
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

catalog_name = "bitext_innovation_international_fine_tuned_mistral_7b_for_retail_banking_customer_service" # Default catalog name when installing the model from Databricks Marketplace

version = 2

model_mlflow_path = f"models:/{catalog_name}.marketplace.mistral7B-retail-banking/{version}"

train_data_catalog_path = f"/Volumes/{catalog_name}/marketplace/mortgage_loans_data/Mortgage_Loans_train_sample.csv"


model_local_path = "/tmp/model/mistral7B-retail-banking/"
model_output_local_path = "/tmp/model/mistral7B-retail-banking_lora_fine_tune_2"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset
# MAGIC
# MAGIC We will use a sample training dataset from the mortgage and loans domain, and which was taken from [bitext/Bitext-mortgage-loans-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-mortgage-loans-llm-chatbot-training-dataset) to fine-tune the original Retail Banking model.

# COMMAND ----------

import pandas as pd

# Load the dataset
dataset = pd.read_csv("hf://datasets/bitext/Bitext-mortgage-loans-llm-chatbot-training-dataset/bitext-mortgage-loans-llm-chatbot-training-dataset.csv")

# Define the prompt template function
def apply_prompt_template(row):
    if 'instruction' in row and 'response' in row:
        question = row['instruction']
        response = row['response']
        return f"<s>[INST] {question} [/INST] {response}</s>"
    else:
        raise KeyError("Missing 'question' or 'response' in row")

# Apply the function across the DataFrame and create a new column
dataset['text'] = dataset.apply(apply_prompt_template, axis=1)

# If you want to see the updated DataFrame
dataset.head()


# COMMAND ----------

print(dataset["text"][556])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the model
# MAGIC
# MAGIC In this section we will load Bitext's [Mistral-7B-Banking](https://huggingface.co/bitext/Mistral-7B-Banking-v2) model installed from [Databricks marketplace](https://marketplace.databricks.com/) saved in Unity Catalog to local disk, quantize it in 4bit and attach LoRA adapters on it.

# COMMAND ----------

import os
from mlflow.artifacts import download_artifacts

try:
    path = download_artifacts(artifact_uri=model_mlflow_path, dst_path=model_local_path)
    print(f"Artifacts downloaded to: {path}")
except PermissionDenied as e:
    print(f"Permission denied error when trying to download artifacts: {e}")
except Exception as e:
    print(f"An error occurred: {e}")


# COMMAND ----------

import os
from mlflow.artifacts import download_artifacts

path = download_artifacts(artifact_uri=model_mlflow_path, dst_path=model_local_path)

tokenizer_path = os.path.join(path, "components", "tokenizer")
model_path = os.path.join(path, "model")

# COMMAND ----------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
)
model.config.use_cache = False

# COMMAND ----------

# MAGIC %md
# MAGIC Load the configuration file in order to create the LoRA model. 
# MAGIC
# MAGIC According to QLoRA paper, it is important to consider all linear layers in the transformer block for maximum performance. 

# COMMAND ----------

# Choose all linear layers from the model
import bitsandbytes as bnb

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

linear_layers = find_all_linear_names(model)
print(f"Linear layers in the model: {linear_layers}")

# COMMAND ----------

from peft import LoraConfig

lora_alpha = 16
lora_dropout = 0.1
lora_r = 8

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=linear_layers,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the trainer

# COMMAND ----------

# MAGIC %md
# MAGIC Here we will use the [`SFTTrainer` from TRL library](https://huggingface.co/docs/trl/main/en/sft_trainer) that gives a wrapper around transformers `Trainer` to easily fine-tune models on instruction based datasets using PEFT adapters. Let's first load the training arguments below.

# COMMAND ----------

from transformers import TrainingArguments

output_dir = "/local_disk0/results"
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
optim = "adamw_bnb_8bit"
save_strategy = "epoch"
logging_steps = 100
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 500
warmup_ratio = 0.03
lr_scheduler_type = "cosine"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_strategy=save_strategy,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    bf16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    ddp_find_unused_parameters=False,
)

# COMMAND ----------

# MAGIC %md
# MAGIC Then finally pass everthing to the trainer

# COMMAND ----------

from trl import SFTTrainer

max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

# COMMAND ----------

# MAGIC %md
# MAGIC We will also pre-process the model by upcasting the layer norms in float 32 for more stable training

# COMMAND ----------

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the model

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's train the model! Simply call `trainer.train()`

# COMMAND ----------

trainer.train()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the LORA model

# COMMAND ----------

trainer.save_model(model_output_local_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the fine tuned model to MLFlow

# COMMAND ----------

import mlflow
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel, PeftConfig

class FINETUNED_QLORA(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts['tokenizer'])
    self.tokenizer.pad_token = self.tokenizer.eos_token
    config = PeftConfig.from_pretrained(context.artifacts['lora'])
    base_model = AutoModelForCausalLM.from_pretrained(
      context.artifacts['base_model'], 
      return_dict=True, 
      load_in_4bit=True, 
      device_map={"":0},
      trust_remote_code=True,
    )
    self.model = PeftModel.from_pretrained(base_model, context.artifacts['lora'])
  
  def predict(self, context, model_input):
    prompt = model_input["prompt"][0]
    temperature = model_input.get("temperature", [0.0])[0]
    max_tokens = model_input.get("max_tokens", [512])[0]
    batch = self.tokenizer(prompt, padding=True, truncation=True,return_tensors='pt').to('cuda')
    with torch.cuda.amp.autocast():
      output_tokens = self.model.generate(
          input_ids = batch.input_ids, 
          max_new_tokens=max_tokens,
          temperature=temperature,
          top_p=0.7,
          num_return_sequences=1,
          do_sample=True,
          pad_token_id=self.tokenizer.eos_token_id,
          eos_token_id=self.tokenizer.eos_token_id,
      )
    generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return generated_text

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
import pandas as pd
import mlflow

# Define input and output schema
input_schema = Schema([
    ColSpec(DataType.string, "prompt"), 
    ColSpec(DataType.double, "temperature", required=False), 
    ColSpec(DataType.long, "max_tokens", required=False)])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example=pd.DataFrame({
            "prompt":["<s>[INST] give me information about the details of my home loan [/INST]"], 
            "temperature": [0.1],
            "max_tokens": [512]})

with mlflow.start_run() as run:  
    mlflow.pyfunc.log_model(
        "model",
        python_model=FINETUNED_QLORA(),
        artifacts={'tokenizer' : tokenizer_path, "base_model": model_path,  "lora": model_output_local_path},
        pip_requirements=["torch", "transformers", "accelerate", "einops", "loralib", "bitsandbytes", "peft"],
        input_example=pd.DataFrame({"prompt":["<s>[INST] give me information about the details of my home loan [/INST]"], "temperature": [0.0],"max_tokens": [512]}),
        signature=signature
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Run model inference with the model logged in MLFlow.

# COMMAND ----------

import mlflow
import pandas as pd

prompt = "<s>[INST] give me information about the details of my home loan [/INST]"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/model")

text_example=pd.DataFrame({
            "prompt":[prompt], 
            "temperature": [0.1],
            "max_tokens": [100]})

# Predict on a Pandas DataFrame.
loaded_model.predict(text_example)

# COMMAND ----------

prompt = "<s>[INST] could ya help me to apply for a student loan wit a family member [/INST]"

text_example=pd.DataFrame({
            "prompt":[prompt], 
            "temperature": [0.1],
            "max_tokens": [512]})

# Predict on a Pandas DataFrame.
print(loaded_model.predict(text_example))

# COMMAND ----------

prompt = "<s>[INST] I would like to add a co-borrower [/INST]"

text_example=pd.DataFrame({
            "prompt":[prompt], 
            "temperature": [0.1],
            "max_tokens": [512]})

# Predict on a Pandas DataFrame.
print(loaded_model.predict(text_example))

# COMMAND ----------

prompt = "<s>[INST] I'd like to know about my application status [/INST]"

text_example=pd.DataFrame({
            "prompt":[prompt], 
            "temperature": [0.1],
            "max_tokens": [512]})

# Predict on a Pandas DataFrame.
print(loaded_model.predict(text_example))

# COMMAND ----------

prompt = "<s>[INST] wanna submit the required documentation [/INST]"

text_example=pd.DataFrame({
            "prompt":[prompt], 
            "temperature": [0.1],
            "max_tokens": [512]})

# Predict on a Pandas DataFrame.
print(loaded_model.predict(text_example))

# COMMAND ----------

prompt = "<s>[INST] I'm looking for information about bi-weekly payments, where can I get it? [/INST]"

text_example=pd.DataFrame({
            "prompt":[prompt], 
            "temperature": [0.1],
            "max_tokens": [512]})

# Predict on a Pandas DataFrame.
print(loaded_model.predict(text_example))
