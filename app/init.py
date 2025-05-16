from main import Settings, TextToImagePipeline


if __name__ == "__main__":
    settings = Settings()

    TextToImagePipeline(
        model_name=settings.model_name,
        device=settings.device,
        dtype=settings.dtype
    )
