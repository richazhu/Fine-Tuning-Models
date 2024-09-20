# Databricks notebook source
# MAGIC %md
# MAGIC # Overview of mistral models in Databricks Marketplace Listing
# MAGIC
# MAGIC The mistral models offered in Databricks Marketplace are text generation models released by Mistral AI. They are [MLflow](https://mlflow.org/docs/latest/index.html) models that packages
# MAGIC [Hugging Face’s implementation for mistral models](https://huggingface.co/mistralai)
# MAGIC using the [transformers](https://mlflow.org/docs/latest/models.html#transformers-transformers-experimental)
# MAGIC flavor in MLflow.
# MAGIC
# MAGIC **Input:** string containing the text of instructions
# MAGIC
# MAGIC **Output:** string containing the generated response text
# MAGIC
# MAGIC For example notebooks of using the mistral model in various use cases on Databricks, please refer to [the Databricks ML example repository](https://github.com/databricks/databricks-ml-examples/tree/master/llm-models/mistral).

# COMMAND ----------

# MAGIC %md
# MAGIC # Listed Marketplace Models
# MAGIC - mistral_7b_instruct_v0_1:
# MAGIC   - It packages [Hugging Face’s implementation for the mistral_7b_instruct_v0_1 model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1).
# MAGIC   - It has 7 billion parameters.
# MAGIC   - It offers competitive processing speed for the generation quality.

# COMMAND ----------

# MAGIC %md
# MAGIC # Suggested environment
# MAGIC Creating and querying serving endpoint don't require specific runtime versions and GPU instance types, but
# MAGIC for batch inference we suggest the following:
# MAGIC
# MAGIC - Databricks Runtime for Machine Learning version 14.2 or greater
# MAGIC - Recommended instance types:
# MAGIC   | Model Name      | Suggested instance type (AWS) | Suggested instance type (AZURE) | Suggested instance type (GCP) |
# MAGIC   | --------------- | ----------------------------- | ------------------------------- | ----------------------------- |
# MAGIC   | `mistral_7b_instruct_v0_1` | `g5.8xlarge` | `Standard_NV36ads_A10_v5` | `g2-standard-4`|
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Install Dependencies
# MAGIC To create and query the model serving endpoint, Databricks recommends to install the newest Databricks SDK for Python.

# COMMAND ----------

# Upgrade to use the newest Databricks SDK
%pip install --upgrade databricks-sdk
# Install the dependencies for batch inference
%pip install --upgrade transformers>=4.34.0
dbutils.library.restartPython()

# COMMAND ----------

# Select the model from the dropdown list
model_names = ['mistral_7b_instruct_v0_1']
dbutils.widgets.dropdown("model_name", model_names[0], model_names)

# COMMAND ----------

# Default catalog name when installing the model from Databricks Marketplace.
# Replace with the name of the catalog containing this model
catalog_name = "databricks_mistral_models"

# You should specify the newest model version to load for inference
version = "1"
model_name = dbutils.widgets.get("model_name")
model_uc_path = f"{catalog_name}.models.{model_name}"
endpoint_name = f'{model_name}_marketplace'

# Choose the right workload types based on the model size
workload_type = "GPU_MEDIUM"

# COMMAND ----------

# MAGIC %md
# MAGIC # Usage

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks recommends that you primarily work with this model via Model Serving
# MAGIC ([AWS](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints)).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploying the model to Model Serving
# MAGIC
# MAGIC You can deploy this model directly to a Databricks Model Serving Endpoint
# MAGIC ([AWS](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints)).
# MAGIC
# MAGIC Note: Model serving is not supported on GCP. On GCP, Databricks recommends running `Batch inference using Spark`, 
# MAGIC as shown below.
# MAGIC
# MAGIC We recommend the below workload types for each model size:
# MAGIC | Model Name      | Suggested workload type (AWS) | Suggested workload type (AZURE) |
# MAGIC | --------------- | ----------------------------- | ------------------------------- |
# MAGIC | `mistral_7b_instruct_v0_1` | GPU_MEDIUM | GPU_LARGE |
# MAGIC
# MAGIC You can create the endpoint by clicking the “Serve this model” button above in the model UI. And you can also
# MAGIC create the endpoint with Databricks SDK as following:
# MAGIC

# COMMAND ----------

import datetime

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput
w = WorkspaceClient()

config = EndpointCoreConfigInput.from_dict({
    "served_models": [
        {
            "name": endpoint_name,
            "model_name": model_uc_path,
            "model_version": version,
            "workload_type": workload_type,
            "workload_size": "Small",
            "scale_to_zero_enabled": "False",
        }
    ]
})
model_details = w.serving_endpoints.create(name=endpoint_name, config=config)
model_details.result(timeout=datetime.timedelta(minutes=60))

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL transcription using ai_query
# MAGIC
# MAGIC To generate the text using the endpoint, use `ai_query`
# MAGIC to query the Model Serving endpoint. The first parameter should be the
# MAGIC name of the endpoint you previously created for Model Serving. The second
# MAGIC parameter should be a `named_struct` with name `prompt` and value is the 
# MAGIC column name that containing the instruction text. Extra parameters can be added
# MAGIC to the named_struct too. For supported parameters, please refer to [MLFlow AI gateway completion routes](https://mlflow.org/docs/latest/gateway/index.html#completions)
# MAGIC The third and fourth parameters set the return type, so that
# MAGIC `ai_query` can properly parse and structure the output text.
# MAGIC
# MAGIC NOTE: `ai_query` is currently in Public Preview. Please sign up at [AI Functions Public Preview enrollment form](https://docs.google.com/forms/d/e/1FAIpQLScVyh5eRioqGwuUVxj9JOiKBAo0-FWi7L3f4QWsKeyldqEw8w/viewform) to try out the new feature.
# MAGIC
# MAGIC ```sql
# MAGIC SELECT 
# MAGIC ai_query(
# MAGIC   <endpoint name>,
# MAGIC   named_struct("prompt", "What is ML?",  "max_tokens", 256),
# MAGIC   'returnType',
# MAGIC   'STRUCT<candidates:ARRAY<STRUCT<text:STRING, metadata:STRUCT<finish_reason:STRING>>>, metadata:STRUCT<input_tokens:float, output_tokens:float, total_tokens:float> >'
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC You can use `ai_query` in this manner to generate text in
# MAGIC SQL queries or notebooks connected to Databricks SQL Pro or Serverless
# MAGIC SQL Endpoints.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate the text by querying the serving endpoint
# MAGIC With the Databricks SDK, you can query the serving endpoint as follows:

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# Change it to your own input
dataframe_records = [
    {"prompt": "What is ML?", "max_tokens": 512}
]

w = WorkspaceClient()
w.serving_endpoints.query(
    name=endpoint_name,
    dataframe_records=dataframe_records,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch inference using Spark
# MAGIC
# MAGIC You can also directly load the model as a Spark UDF and run batch
# MAGIC inference on Databricks compute using Spark. We recommend using a
# MAGIC GPU cluster with Databricks Runtime for Machine Learning version 14.1
# MAGIC or greater.

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")

logged_model = f"models:/{catalog_name}.models.{model_name}/{version}"
generate = mlflow.pyfunc.spark_udf(spark, logged_model, "string")

# COMMAND ----------

import pandas as pd

df = spark.createDataFrame(pd.DataFrame({"text": pd.Series("What is ML?")}))

# You can use the UDF directly on a text column
generated_df = df.select(generate(df.text).alias('generated_text'))

# COMMAND ----------

display(generated_df)
