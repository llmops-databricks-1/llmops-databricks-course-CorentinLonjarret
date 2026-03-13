# Databricks notebook source
"""
Hello World notebook — verifies the environment is set up correctly.
"""

# COMMAND ----------

print("Hello, world!")
print("Environment is ready.")

# COMMAND ----------

from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.profile("DEV").serverless(True).getOrCreate()
df = spark.read.table("samples.nyctaxi.trips")
df.show(5)

# COMMAND ----------
spark.stop()
# COMMAND ----------
