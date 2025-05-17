"""FastAPI application for text-to-image generation using a pretrained model."""
import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional, Literal
import torch
from diffusers import FluxPipeline
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the application."""
    model_name: str = Field(default="black-forest-labs/FLUX.1-schnell",
                            description="Pretrained model identifier")
    device: Literal["cuda", "mps", "cpu"] = Field(
        default="cpu" if torch.cuda.is_available() else "cpu",
        description="Device to run the model on (cuda, mps, or cpu)"
    )
    dtype: Literal["bfloat16", "float16", "float32"] = Field(
        default="float32" if device == "cuda" else "bfloat16",
        description="Data type for model weights (bfloat16, float16, or float32)"
    )
    model_config = SettingsConfigDict(env_prefix='APP_')


class TextToImagePipelineInput(BaseModel):
    """Input for the text-to-image pipeline."""
    prompt: str = Field(..., min_length=1,
                        description="Text prompt for image generation")
    height: int = Field(default=512, ge=64, le=2048,
                        description="Image height in pixels")
    width: int = Field(default=512, ge=64, le=2048,
                       description="Image width in pixels")
    guidance_scale: float = Field(
        default=0.0, ge=0.0, le=20.0, description="Guidance scale for model")
    num_inference_steps: int = Field(
        default=1, ge=1, le=100, description="Number of inference steps")
    max_sequence_length: int = Field(
        default=256, ge=1, le=512, description="Maximum sequence length")
    seed: int = Field(
        default=0, ge=0, description="Random seed for reproducibility")

    @field_validator('height', 'width')
    @classmethod
    def check_multiple_of_8(cls, v: int, field) -> int:
        """Ensure height and width are multiples of 8."""
        if v % 8 != 0:
            raise ValueError(
                f"{field.name} must be a multiple of 8 for stable diffusion models")
        return v


class TextToImagePipeline:
    """Text-to-image pipeline using a pretrained model."""
    def __init__(self, model_name: str, device: Optional[str] = None, dtype: str = "bfloat16"):
        self.model_name = model_name
        self.device = device
        if self.device not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"Unsupported device: {self.device}")
        dtype_map = {"bfloat16": torch.bfloat16,
                     "float16": torch.float16, "float32": torch.float32}
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")
        self.dtype = dtype_map[dtype]
        self.pipeline = FluxPipeline.from_pretrained(
            model_name, torch_dtype=self.dtype
        ).to(device=self.device)
        if self.device == "cuda":
            self.pipeline.enable_model_cpu_offload()


    def generate(self, pipeline_input: TextToImagePipelineInput) -> bytes:
        """Generate an image from a text prompt."""
        images = self.pipeline(
            pipeline_input.prompt,
            height=pipeline_input.height,
            width=pipeline_input.width,
            guidance_scale=pipeline_input.guidance_scale,
            num_inference_steps=pipeline_input.num_inference_steps,
            max_sequence_length=pipeline_input.max_sequence_length,
            generator=torch.Generator(
                device=self.device).manual_seed(pipeline_input.seed)
        )

        image = images.images[0]

        byte_io = BytesIO()
        image.save(byte_io, format='PNG')
        byte_io.seek(0)

        return byte_io.getvalue()


settings = Settings()
pipeline: Optional[TextToImagePipeline] = None

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Lifespan context manager for FastAPI."""
    global pipeline
    pipeline = TextToImagePipeline(
        model_name=settings.model_name,
        device=settings.device,
        dtype=settings.dtype
    )
    yield

app = FastAPI(lifespan=lifespan)


@app.post(
    "/generate",
    responses={
        200: {
            "content": {"image/png": {}}
        }
    },
    response_class=Response
)
async def generate(request: TextToImagePipelineInput):
    """Generate an image from a text prompt."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, pipeline.generate, request)
    return Response(content=result, media_type="image/png")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
