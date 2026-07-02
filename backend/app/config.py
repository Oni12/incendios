from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    opentopography_api_key: str = ""
    windninja_cli_path: str = r"C:\WindNinja\WindNinja-3.12.0\bin\WindNinja_cli.exe"
    grid_size: int = 1000
    default_humidity: float = 0.4
    default_temperature: float = 20.0
    default_detection_time: float = 30.0
    default_wind_speed: float = 15.5
    default_wind_direction: float = 180.0

    model_config = {"env_prefix": "", "env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
