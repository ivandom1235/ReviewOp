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
        if not generic_path.exists():
            # In strict mode or diagnostics we might want this to be fatal
            # For now just ensure it's not silent
            raise FileNotFoundError(f"Mandatory generic.json missing in {config_dir}")
            
        config = json.loads(generic_path.read_text(encoding="utf-8"))
        cls._validate_schema(config, "generic")
            
        # Load domain-specific override
        if domain != "generic":
            domain_path = config_dir / f"{domain}.json"
            if domain_path.exists():
                overrides = json.loads(domain_path.read_text(encoding="utf-8"))
                cls._validate_schema(overrides, domain)
                # Deep merge
                cls._deep_merge(config, overrides)
                    
        cls._cache[domain] = config
        return config

    @classmethod
    def _deep_merge(cls, base: dict[str, Any], overrides: dict[str, Any]) -> None:
        """Recursively merge dictionaries."""
        for key, value in overrides.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                cls._deep_merge(base[key], value)
            else:
                base[key] = value

    @classmethod
    def _validate_schema(cls, config: dict[str, Any], domain: str) -> None:
        """Simple schema validation for domain config files."""
        required_dicts = ["domain_maps", "modifier_maps", "latent_families"]
        for key in required_dicts:
            if key in config and not isinstance(config[key], dict):
                raise ValueError(f"invalid config schema in {domain}: {key} must be a dict")
        
        if "broad_labels" in config and not isinstance(config["broad_labels"], list):
            raise ValueError(f"invalid config schema in {domain}: broad_labels must be a list")

    @classmethod
    def get_latent_families(cls, domain: str | None) -> dict[str, list[str]]:
        return cls.get_config(domain).get("latent_families", {})

    @classmethod
    def get_broad_labels(cls, domain: str | None) -> list[str]:
        return cls.get_config(domain).get("broad_labels", [])

    @classmethod
    def get_domain_map(cls, domain: str | None) -> dict[str, str]:
        return cls.get_config(domain).get("domain_maps", {})
