"""Configuration management for Mirror Mirror using pydantic-settings."""

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="MIRROR_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8080

    zmq_host: str = "localhost"
    zmq_bind_host: str = "0.0.0.0"  # Host for ZMQ binds (gateway binds on all interfaces)
    zmq_frames_port: int = 5555
    zmq_audio_port: int = 5556
    zmq_transformed_port: int = 5557
    zmq_prompt_port: int = 5558  # Audio → Gateway
    zmq_prompt_pub_port: int = 5559  # Gateway → Diffusion (republish)
    zmq_state_port: int = 5560  # ROUTER for state requests (snapshot pattern)
    zmq_debug_assets_port: int = 5561

    ipc_dir: str = "/tmp/mirror"
    ipc_ui_diffusion: str = "ui_diffusion"
    ipc_ui_vignette: str = "ui_vignette"
    ipc_ui_prompt: str = "ui_prompt"

    frame_size: int = 512

    gemma_model: str = "google/gemma-3n-E4B-it"

    # Database
    db_path: str = "./mirror.db"
    default_prompt: str = "oil painting style, artistic, painterly"

    # Logging
    log_level: str = "INFO"

    # ZMQ connect addresses (for workers connecting to gateway)
    @computed_field
    @property
    def zmq_frames_addr(self) -> str:
        return f"tcp://{self.zmq_host}:{self.zmq_frames_port}"

    @computed_field
    @property
    def zmq_audio_addr(self) -> str:
        return f"tcp://{self.zmq_host}:{self.zmq_audio_port}"

    @computed_field
    @property
    def zmq_transformed_addr(self) -> str:
        return f"tcp://{self.zmq_host}:{self.zmq_transformed_port}"

    @computed_field
    @property
    def zmq_prompt_addr(self) -> str:
        return f"tcp://{self.zmq_host}:{self.zmq_prompt_port}"

    @computed_field
    @property
    def zmq_prompt_pub_addr(self) -> str:
        """Address for diffusion to receive republished prompts from gateway."""
        return f"tcp://{self.zmq_host}:{self.zmq_prompt_pub_port}"

    @computed_field
    @property
    def zmq_state_addr(self) -> str:
        """Address for workers to request state snapshots."""
        return f"tcp://{self.zmq_host}:{self.zmq_state_port}"

    # ZMQ bind addresses (for gateway binding on all interfaces)
    @computed_field
    @property
    def zmq_frames_bind_addr(self) -> str:
        return f"tcp://{self.zmq_bind_host}:{self.zmq_frames_port}"

    @computed_field
    @property
    def zmq_audio_bind_addr(self) -> str:
        return f"tcp://{self.zmq_bind_host}:{self.zmq_audio_port}"

    @computed_field
    @property
    def zmq_transformed_bind_addr(self) -> str:
        return f"tcp://{self.zmq_bind_host}:{self.zmq_transformed_port}"

    @computed_field
    @property
    def zmq_prompt_bind_addr(self) -> str:
        return f"tcp://{self.zmq_bind_host}:{self.zmq_prompt_port}"

    @computed_field
    @property
    def zmq_prompt_pub_bind_addr(self) -> str:
        """Bind address for gateway to republish prompts."""
        return f"tcp://{self.zmq_bind_host}:{self.zmq_prompt_pub_port}"

    @computed_field
    @property
    def zmq_state_bind_addr(self) -> str:
        return f"tcp://{self.zmq_bind_host}:{self.zmq_state_port}"

    @computed_field
    @property
    def ipc_ui_diffusion_addr(self) -> str:
        return f"ipc://{self.ipc_dir}/{self.ipc_ui_diffusion}"

    @computed_field
    @property
    def ipc_ui_vignette_addr(self) -> str:
        return f"ipc://{self.ipc_dir}/{self.ipc_ui_vignette}"

    @computed_field
    @property
    def ipc_ui_prompt_addr(self) -> str:
        return f"ipc://{self.ipc_dir}/{self.ipc_ui_prompt}"


config = Config()
