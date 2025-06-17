from PIL import Image
from model import LLM
from utils.examples import REPORT_EXAMPLE
from utils.state import ImageProcessingState, ReportSchema

import os
import yaml
import json


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

def describe_images(state: ImageProcessingState) -> ImageProcessingState:
    vision_model = LLM()

    for img_path in state["image_paths"]:
        print(img_path)
        extracted_text = vision_model.describe_image(img_path, "\n".join(feature['extracted_text'] for feature in state["features"]))
        state["features"].append({"image_path": img_path, "extracted_text": extracted_text})
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
    features = '\n'.join(feature["extracted_text"] for feature in state['features'])
    user_prompt = f"Create a report from these image descriptions: {features}\n\nThese are examples on what a report should look like:\n{REPORT_EXAMPLE}"
    
    state["aggregated_info"] = llm.create_report(user_prompt, system_prompt, model="o4-mini")

    # Saving json for debug purposes
    with open("/Users/tmskss/Development/nlp-homework-raiffeisen/data/runs/aggregated_info.json", "w") as f:
        json.dump(state["aggregated_info"], f, indent=4)

    # with open("/Users/tmskss/Development/nlp-homework-raiffeisen/data/runs/aggregated_info.json", "r") as f:
    #     state["aggregated_info"] = json.load(f)

    state["final_report_markdown"] = json_to_markdown(state["aggregated_info"])
    return state

def finish_report(state: ImageProcessingState) -> ImageProcessingState:
    llm = LLM()

    system_prompt = f"You are a helpful assistant who finishes an almost done report. Your most important task is to rephrase the report to sound more professional, and follow the correct formatting.\n\nExample finished reports:\n{REPORT_EXAMPLE}"
    user_prompt = f"Rephrase and modify the structure of the following report:\n\n{state['final_report_markdown']}"

    state["final_report_markdown"] = llm.invoke(user_prompt, system_prompt)

    return state

def json_to_markdown(json_data: str) -> str:
    data = json.loads(json_data)
    
    markdown = ""

    for key, value in data.items():
        formatted_key = key.replace('_', ' ').title()
        value = value.replace('\\n', '\n')
        markdown += f"**{formatted_key}:**\n{value}\n"

    return markdown

def convert_report(report: ReportSchema) -> str:
    return f"### Identification & General Data\n{report.identification}\n\n### Inspection Methods\n{report.inspection_methods}\n\n### Condition Assessment\n{report.condition_assessment}\n\n### Documentation & Accessories\n{report.documentation_and_accessories}"

