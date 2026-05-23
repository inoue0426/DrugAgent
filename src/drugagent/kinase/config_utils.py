import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv


def _resolve_repo_root() -> Path:
    """Resolve the repository root by walking up to pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parents[3]


def _find_env_path() -> Path:
    """Return the preferred .env path for Azure OpenAI config."""
    explicit_path = os.getenv("AZURE_OPENAI_ENV_PATH")
    if explicit_path:
        return Path(explicit_path)

    project_root = _resolve_repo_root()
    candidates = [
        Path.cwd() / ".env",
        project_root / ".env",
        project_root / "latest_DrugAgent" / ".env",
    ]

    for parent in project_root.parents:
        if parent.name == "latest_DrugAgent":
            candidates.insert(0, parent / ".env")
            break

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return project_root / ".env"


def load_azure_openai_config() -> Dict[str, str]:
    env_path = _find_env_path()
    print(f"[config] loading .env from: {env_path}")
    load_dotenv(str(env_path), override=True)

    config = {
        "api_key": os.getenv("AZURE_OPENAI_API_LLM_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        "embedding_api_version": os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
        "embedding_endpoint": os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
        "embedding_api_key": os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
        "claude_endpoint": os.getenv("CLAUDE_ENDPOINT")
    }

    # Raise if required keys are missing.
    required_keys = ["api_key", "endpoint", "deployment_name", "api_version"]
    missing = [key for key in required_keys if not config[key]]
    if missing:
        raise RuntimeError(
            f"Missing Azure OpenAI environment variables: {', '.join(missing)}"
        )

    if not config["embedding_api_version"]:
        config["embedding_api_version"] = config["api_version"]
    if not config["embedding_endpoint"]:
        config["embedding_endpoint"] = config["endpoint"]
    if not config["embedding_api_key"]:
        config["embedding_api_key"] = config["api_key"]

    return config


def get_reasoning_settings(
    env_var: str = "REASONING_EFFORT",
) -> Optional[Dict[str, str]]:
    """Return reasoning settings based on an environment variable.

    Args:
        env_var: Environment variable name for reasoning effort.

    Returns:
        Dictionary suitable for the "reasoning" parameter, or None to disable it.
    """
    load_dotenv()
    raw_value = os.getenv(env_var)
    if raw_value is None:
        return None
    value = raw_value.strip().lower()
    if not value or value == "none":
        return None
    if value in {"low", "medium", "high"}:
        return {"effort": value}
    raise ValueError(
        f"Invalid {env_var} value: {raw_value}. Expected low, medium, high, or none."
    )
