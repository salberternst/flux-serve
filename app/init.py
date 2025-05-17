"""FastAPI application for text-to-image generation using a pretrained model."""
from diffusers import DiffusionPipeline
from main import Settings

if __name__ == "__main__":
    settings = Settings()
    DiffusionPipeline.download(settings.model_name)
