from __future__ import annotations

import toml
from typing import Dict, List, Any, Union


def load_config() -> Dict:
    with open('config.toml') as f:
        config = toml.load(f)
    return config
