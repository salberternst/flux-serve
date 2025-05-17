from main import Settings
from diffusers import DiffusionPipeline

if __name__ == "__main__":
    settings = Settings()
    DiffusionPipeline.download(settings.model_name)