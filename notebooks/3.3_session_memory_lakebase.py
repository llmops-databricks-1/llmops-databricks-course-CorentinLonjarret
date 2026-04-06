# Databricks notebook source

# Project
#  └── Branches (main, development, staging, etc.)
#        ├── Computes (R/W compute)
#        ├── Roles (Postgres roles)
#        └── Databases (Postgres databases)

import json
import urllib.parse
from uuid import uuid4

import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import (
    PostgresAPI,
    Project,
    ProjectDefaultEndpointSettings,
    ProjectSpec,
)
from google.protobuf.duration_pb2 import Duration
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from arxiv_curator.config import ProjectConfig, get_env, load_config
from arxiv_curator.memory import LakebaseMemory

# COMMAND ----------
cfg = ProjectConfig.from_yaml("../project_config.yml")

w = WorkspaceClient()
pg_api = PostgresAPI(w.api_client)

project_id = cfg.lakebase_project_id

try:
    project = pg_api.get_project(name=f"projects/{project_id}")
except Exception:
    project = pg_api.create_project(
        project_id=project_id,
        project=Project(
            spec=ProjectSpec(
                display_name=project_id,
                # budget_policy_id=cfg.usage_policy_id,
                default_endpoint_settings=ProjectDefaultEndpointSettings(
                    autoscaling_limit_min_cu=1,
                    autoscaling_limit_max_cu=4,
                    suspend_timeout_duration=Duration(seconds=300),
                ),
            ),
        ),
    ).wait()

# COMMAND ----------
# Get endpoint, host, and generate credential
default_branch = next(iter(pg_api.list_branches(parent=project.name)))  # type: ignore
endpoint = next(iter(pg_api.list_endpoints(parent=default_branch.name)))  # type: ignore
host = endpoint.status.hosts.host  # type: ignore

# Get username (works with user credentials in notebook)
user = w.current_user.me()
pg_credential = pg_api.generate_database_credential(endpoint=endpoint.name)  # type: ignore
username = urllib.parse.quote_plus(user.user_name)  # type: ignore
conn_string = f"postgresql://{username}:{pg_credential.token}@{host}:5432/databricks_postgres?sslmode=require"


# COMMAND ----------
with psycopg.connect(conn_string) as conn:
    # Create session_messages table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_messages (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            message_data JSONB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_messages_session_id
        ON session_messages(session_id)
    """)

# COMMAND ----------

test_session_id = f"test-session-{uuid4()}"
test_messages = [
    {"role": "user", "content": "Hello, what can you help me with?"},
    {"role": "assistant", "content": "I can help you find research papers."},
    {"role": "user", "content": "Find papers about LLM reasoning"},
]

with psycopg.connect(conn_string) as conn:
    for msg in test_messages:
        conn.execute(
            "INSERT INTO session_messages (session_id, message_data) VALUES (%s, %s)",
            (test_session_id, json.dumps(msg)),
        )

# COMMAND ----------

# Load messages back
with psycopg.connect(conn_string) as conn:
    result = conn.execute(
        """
        SELECT message_data, created_at FROM session_messages
        WHERE session_id = %s
        ORDER BY created_at ASC
        """,
        (test_session_id,),
    ).fetchall()

    logger.info(f"Loaded {len(result)} messages:")
    for row in result:
        logger.info(f"  [{row[1]}] {row[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test with LakebaseMemory Class

# COMMAND ----------

memory = LakebaseMemory(
    project_id=project_id,
)

# COMMAND ----------

# Test save
session_id = f"memory-test-{uuid4()}"
messages = [
    {"role": "user", "content": "What papers discuss transformer architectures?"},
    {"role": "assistant", "content": "Here are some relevant papers..."},
]

memory.save_messages(session_id, messages)
logger.info(f"Saved messages to session: {session_id}")

# COMMAND ----------

# Test load
loaded = memory.load_messages(session_id)
logger.info(f"Loaded {len(loaded)} messages:")
for msg in loaded:
    logger.info(f"  {msg['role']}: {msg['content'][:50]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using Memory with an LLM

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

# Create OpenAI client for Databricks
client = OpenAI(
    api_key=w.tokens.create(lifetime_seconds=1200).token_value, base_url=f"{w.config.host}/serving-endpoints"
)


def chat_with_memory(session_id: str, user_message: str, memory: LakebaseMemory) -> str:
    """Chat with LLM using session memory for context."""
    # Load previous messages
    previous_messages = memory.load_messages(session_id)

    # Build messages with system prompt
    messages = (
        [{"role": "system", "content": "You are a helpful research assistant."}]
        + previous_messages
        + [{"role": "user", "content": user_message}]
    )

    # Call LLM
    response = client.chat.completions.create(
        model=cfg.llm_endpoint,
        messages=messages,  # type: ignore
    )

    assistant_response = response.choices[0].message.content

    # Save new messages to memory
    memory.save_messages(
        session_id,
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response},
        ],
    )

    return assistant_response  # type: ignore


logger.info("✓ Chat function with memory created")

# COMMAND ----------
# Create a new session with memory
agent_session_id = f"agent-session-{uuid4()}"

# First query
response1 = chat_with_memory(agent_session_id, "What is RAG in the context of LLMs?", memory)
logger.info(f"Response 1: {response1[:200]}...")

# COMMAND ----------
# Follow-up query with context (memory is automatically loaded)
response2 = chat_with_memory(agent_session_id, "What are the main components?", memory)
logger.info(f"Response 2: {response2[:200]}...")

# COMMAND ----------
# View full conversation
full_agent_conversation = memory.load_messages(agent_session_id)

logger.info(f"✓ Full agent conversation ({len(full_agent_conversation)} messages):")
for i, msg in enumerate(full_agent_conversation, 1):
    role = msg["role"]
    content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
    logger.info(f"  {i}. [{role}] {content}")
