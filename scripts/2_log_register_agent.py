"""Script d'orchestration — evaluation, log et register de l'agent arXiv.

Usage:
    python scripts/2_log_register_agent.py --env dev --git_sha abc123 --run_id 12345
"""

import mlflow

from arxiv_curator.agent import log_register_agent
from arxiv_curator.config import load_config, resolve_project_path
from arxiv_curator.evaluation import evaluate_agent
from arxiv_curator.utils import get_job_parameter

env = get_job_parameter("env", "dev") or "dev"
git_sha = get_job_parameter("git_sha", "local") or "local"
run_id = get_job_parameter("run_id", "local") or "local"

cfg = load_config("project_config.yml", env=env)

mlflow.set_experiment(cfg.experiment_name)

model_name = f"{cfg.catalog}.{cfg.schema}.{cfg.agent_name}"

results = evaluate_agent(cfg, eval_inputs_path=str(resolve_project_path("arxiv_agent/eval_inputs.txt")))

log_register_agent(
    cfg=cfg,
    git_sha=git_sha,
    run_id=run_id,
    agent_code_path=str(resolve_project_path("arxiv_agent/arxiv_agent.py")),
    model_name=model_name,
    evaluation_metrics=results.metrics,
)
