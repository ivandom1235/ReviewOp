from __future__ import annotations
import json
from pathlib import Path
from typing import Any

DEFAULT_DOMAIN_CONFIG_DIR = Path("dataset_builder/config/domains")

class DomainRegistry:
    _cache: dict[str, dict[str, Any]] = {}
    
    @classmethod
    def get_config(cls, domain: str | None, config_dir: Path = DEFAULT_DOMAIN_CONFIG_DIR) -> dict[str, Any]:
        domain = str(domain or "generic").lower()
        if domain in cls._cache:
            return cls._cache[domain]
            
        # Load generic base
        generic_path = config_dir / "generic.json"
        config = {}
        if generic_path.exists():
            config = json.loads(generic_path.read_text(encoding="utf-8"))
            
        # Load domain-specific override
        domain_path = config_dir / f"{domain}.json"
        if domain_path.exists() and domain != "generic":
            overrides = json.loads(domain_path.read_text(encoding="utf-8"))
            # Merge
            for key, value in overrides.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value
                    
        cls._cache[domain] = config
        return config

    @classmethod
    def get_latent_families(cls, domain: str | None) -> dict[str, list[str]]:
        return cls.get_config(domain).get("latent_families", {})

    @classmethod
    def get_broad_labels(cls, domain: str | None) -> list[str]:
        return cls.get_config(domain).get("broad_labels", [])

    @classmethod
    def get_domain_map(cls, domain: str | None) -> dict[str, str]:
        return cls.get_config(domain).get("domain_maps", {})
