from main import Settings, TextToImagePipeline
from huggingface_hub import snapshot_download


if __name__ == "__main__":
    settings = Settings()

    snapshot_download(repo_id=settings.model_name)
