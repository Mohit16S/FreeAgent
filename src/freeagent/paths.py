"""Runtime storage, independent of any developer machine or virtual environment."""
from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class RuntimePaths:
    root: Path

    @classmethod
    def discover(cls, default: Path | None = None) -> 'RuntimePaths':
        return cls(Path(os.environ.get('FREEAGENT_HOME', str(default or Path.cwd()))).resolve())

    @property
    def data(self) -> Path:
        return self.root / 'data'

    @property
    def artifact(self) -> Path:
        return self.root / 'models/artifacts/core.joblib'

    @property
    def reports(self) -> Path:
        return self.root / 'reports/backtest'

    @property
    def status(self) -> Path:
        return self.root / 'data/setup-status.json'
