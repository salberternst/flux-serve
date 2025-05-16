import asyncio
import torch
from contextlib import asynccontextmanager
from diffusers import FluxPipeline
from fastapi import FastAPI, Response
from io import BytesIO
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional, Literal


class Settings(BaseSettings):
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

    class Config:
        env_prefix = "APP_"


settings = Settings()


class TextToImagePipelineInput(BaseModel):
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
        if v % 8 != 0:
            raise ValueError(
                f"{field.name} must be a multiple of 8 for stable diffusion models")
        return v


class TextToImagePipeline:
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

    def generate(self, input: TextToImagePipelineInput) -> bytes:
        images = self.pipeline(
            input.prompt,
            height=input.height,
            width=input.width,
            guidance_scale=input.guidance_scale,
            num_inference_steps=input.num_inference_steps,
            max_sequence_length=input.max_sequence_length,
            generator=torch.Generator(
                device=self.device).manual_seed(input.seed)
        )

        image = images.images[0]

        byte_io = BytesIO()
        image.save(byte_io, format='PNG')
        byte_io.seek(0)

        return byte_io.getvalue()


pipeline: Optional[TextToImagePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, pipeline.generate, request)
    return Response(content=result, media_type="image/png")


@app.get("/health")
async def health():
    return {"status": "healthy"}
