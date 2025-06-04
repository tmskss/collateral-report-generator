from typing import TypedDict
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

import torch
import os

# Define the state for the graph
class ImageProcessingState(TypedDict):
    images_dir: str
    images: list[str] = None
    image_names: list[str] = None
    classifications: list[str] = None
    features: list[str] = None
    texts: list[str] = None
    aggregated_info = None
    description = None

# Load the image classification parameters
CLASSIFICATION_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
CLASSIFICATION_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
CLASSIFICATION_LABELS = ["An image where the main information is the text.", "An image where the main information is the object."]


def load_images(state: ImageProcessingState) -> ImageProcessingState: 
    """
    Load images from a directory and return them as a list of PIL Image objects.
    
    Args:
        images_dir (str): The directory containing the images.
    
    Returns:
        list: A list of PIL Image objects.
    """
    for filename in os.listdir(state["images_dir"]):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(state["images_dir"], filename)
            state["images"].append(Image.open(img_path))
            state["image_names"].append(filename)

    print(f"Loaded {len(state['images'])} images from {state['images_dir']}")
    return state

def classify_images(state: ImageProcessingState) -> ImageProcessingState:
    for image in state["images"]:
        inputs = CLASSIFICATION_PROCESSOR(text=CLASSIFICATION_LABELS, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = CLASSIFICATION_MODEL(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            state["classifications"].append(probs.argmax())
    return state