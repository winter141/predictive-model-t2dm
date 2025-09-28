from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv
from pathlib import Path

from process_data.food_categories import FOOD_CATEGORIES

# Load env token from root .env
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("HUGGING_FACE_AI_TOKEN")

# Initialize client with explicit provider (Hugging Face inference)
client = InferenceClient(token=api_key, provider="hf-inference")


image_path = "00000049-PHOTO-2019-11-20-12-25-0.jpg"

response = client.zero_shot_image_classification(
    image=image_path,
    model="openai/clip-vit-base-patch32",
    candidate_labels=FOOD_CATEGORIES
)

top = response[0]
print(f"Predicted Food Type: {top.label}, Score: {top.score:.2f}")
