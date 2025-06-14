from typing import TypedDict
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from model import LLM
from utils.examples import REPORT_EXAMPLE

import torch
import pytesseract
import os
import yaml
import json

# Define the state for the graph
class ImageProcessingState(TypedDict):
    images_dir: str
    images: list[str]
    image_names: list[str]
    image_paths: list[str]
    features: list[str]
    aggregated_info: str

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
            state["image_paths"].append(img_path)

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

    print(f"Classified {len(state['images'])} images.")
    return state

def extract_text(state: ImageProcessingState) -> ImageProcessingState:
    """
    Extract text from images using Tesseract OCR with optimizations for industrial imagery.
    
    Args:
        state (ImageProcessingState): The current state containing images and classifications.
    
    Returns:
        ImageProcessingState: Updated state with extracted text.
    """
    
    text_images = [img for img, cls in zip(state["images"], state["classifications"]) if cls == 0]
        # Customize OCR based on image classification
        # if state["classifications"][i] == 0:  # Text-focused image (dashboard, ID plates)
        #     # For text-heavy images, use page segmentation mode for structured text
        #     custom_config = r'--oem 3 --psm 6 -l eng'
        # else:  # Object-focused image (tire numbers, small labels)
        #     # For sparse text on objects, use sparse text mode with minimal whitespace
        #     custom_config = r'--oem 3 --psm 11 -l eng'
        
    for img in text_images:
        extracted_text = pytesseract.image_to_string(img)
        extracted_text = extracted_text.strip()

        state["texts"].append(extracted_text)
    
    print(f"Extracted text from {len(text_images)} images.")
    return state

def describe_images(state: ImageProcessingState) -> ImageProcessingState:
    vision_model = LLM()

    for img_path in state["image_paths"]:
        extracted_text = vision_model.describe_image(img_path, "\n".join(state["features"]))
        state["features"].append(extracted_text)
        print(f"Extracted text ({img_path}):\n{extracted_text}\n{'-'*20}")

    print(f"Extracted text from {len(state['image_paths'])} images.")

    with open("/Users/tmskss/Development/nlp-homework-raiffeisen/data/runs/features.yaml", "w") as f:
        yaml.dump(state["features"], f)
    return state
    # with open("/Users/tmskss/Development/nlp-homework-raiffeisen/data/runs/features.yaml", "r") as f:
    #     state["features"] = yaml.safe_load(f)

    # return state

def aggregate_info(state: ImageProcessingState) -> ImageProcessingState:
    llm = LLM()
    system_prompt = """
    You are a helpful assistant who receives descriptions of multiple images of one object used as collateral and creates a report from it. A lot of images can focus on a certain part of an object, for example the tire of a car. Always focus on the object as a whole and not on a specific part in the report.

    If you are unclear about an importart detail, indicate it in the report.
    """
    features = '\n'.join(state['features'])
    user_prompt = f"Create a report from these image descriptions: {features}\n\nThese are examples on what a report should look like:\n{REPORT_EXAMPLE}"
    
    state["aggregated_info"] = llm.create_report(user_prompt, system_prompt, model="gpt-4o")

    with open("/Users/tmskss/Development/nlp-homework-raiffeisen/data/runs/aggregated_info.json", "w") as f:
        json.dump(state["aggregated_info"], f, indent=4)
    return state
