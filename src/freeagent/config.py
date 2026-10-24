"""Central configuration, shared by the CLI and dashboard."""
from pathlib import Path
import os
import yaml


def load_config(path: Path | None = None) -> dict:
    selected=path or Path(os.getenv('FREEAGENT_CONFIG','config/default.yaml'))
    config=yaml.safe_load(selected.read_text())
    required={'seed','bootstrap_draws','simulations','min_player_minutes','ridge_alphas','match_decay_days','qualification_places','relegation_places','financial_tolerance','verdict_probability'}
    if not isinstance(config,dict) or not required.issubset(config): raise ValueError('Configuration lacks required fields')
    return config
