from PIL import Image
from utils.model import LLM
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
    """
        Extract text descriptions from images using a vision model.
        
        This function processes each image in the state, using a vision model to generate
        textual descriptions. The descriptions are stored in the state's features list.
        Results are also saved to a YAML file for persistence.
        
        Args:
            state (ImageProcessingState): The current state containing image paths
        
        Returns:
            ImageProcessingState: Updated state with extracted features from images
    """
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.getcwd(), "data", "runs")
    os.makedirs(data_dir, exist_ok=True)
    features_path = os.path.join(data_dir, "features.yaml")

    # Uncomment this to start from a saved state
    # features_path = os.path.join(data_dir, "features.yaml")
    # with open(features_path, "r") as f:
    #     state["features"] = yaml.safe_load(f)
    # return state

    vision_model = LLM()

    for img_path in state["image_paths"]:
        print(img_path)
        extracted_text = vision_model.describe_image(img_path, "\n".join(feature['extracted_text'] for feature in state["features"]))
        state["features"].append({"image_path": img_path, "extracted_text": extracted_text})
        print(f"Extracted text ({img_path}):\n{extracted_text}\n{'-'*20}")

    print(f"Extracted text from {len(state['image_paths'])} images.")

    with open(features_path, "w") as f:
        yaml.dump(state["features"], f)
    
    return state

def aggregate_info(state: ImageProcessingState) -> ImageProcessingState:
    """
        Aggregate information from all image descriptions into a structured report.
        
        This function uses an LLM to analyze all extracted text features and generate
        a comprehensive report in JSON format. The report is then converted to markdown
        and stored in the state.
        
        Args:
            state (ImageProcessingState): The current state containing extracted features
        
        Returns:
            ImageProcessingState: Updated state with aggregated information and markdown report
    """

    data_dir = os.path.join(os.getcwd(), "data", "runs")
    os.makedirs(data_dir, exist_ok=True)
    aggreagated_info_path = os.path.join(data_dir, "aggregated_info.json")


    llm = LLM()
    system_prompt = """
    You are a helpful assistant who receives descriptions of multiple images of one object used as collateral and creates a report from it. A lot of images can focus on a certain part of an object, for example the tire of a car. Always focus on the object as a whole and not on a specific part in the report.

    If you are unclear about an importart detail, indicate it in the report.
    """
    features = '\n'.join(feature["extracted_text"] for feature in state['features'])
    user_prompt = f"Create a report from these image descriptions: {features}\n\nThese are examples on what a report should look like:\n{REPORT_EXAMPLE}"
    
    state["aggregated_info"] = llm.create_report(user_prompt, system_prompt, model="o4-mini")

    # Saving json for debug purposes
    with open(aggreagated_info_path, "w") as f:
        json.dump(state["aggregated_info"], f, indent=4)

    # Uncomment this to start from a saved state
    # with open(aggreagated_info_path, "r") as f:
    #     state["aggregated_info"] = json.load(f)

    state["final_report_markdown"] = json_to_markdown(state["aggregated_info"])
    return state

def finish_report(state: ImageProcessingState) -> ImageProcessingState:
    """
        Finalize and polish the report to make it more professional.
        
        This function takes the draft markdown report and uses an LLM to improve
        its phrasing, formatting, and overall professional tone.
        
        Args:
            state (ImageProcessingState): The current state containing the draft report
        
        Returns:
            ImageProcessingState: Updated state with the finalized professional report
    """
    llm = LLM()

    system_prompt = f"You are a helpful assistant who finishes an almost done report. Your most important task is to rephrase the report to sound more professional, and follow the correct formatting.\n\nExample finished reports:\n{REPORT_EXAMPLE}"
    user_prompt = f"Rephrase and modify the structure of the following report:\n\n{state['final_report_markdown']}"

    state["final_report_markdown"] = llm.invoke(user_prompt, system_prompt)

    return state

def json_to_markdown(json_data: str) -> str:
    """
        Convert a JSON string to formatted markdown.
        
        This function parses a JSON string and converts it to markdown format
        with section titles and formatted content.
        
        Args:
            json_data (str): JSON string to convert
        
        Returns:
            str: Formatted markdown string
    """
    data = json.loads(json_data)
    
    markdown = ""

    for key, value in data.items():
        formatted_key = key.replace('_', ' ').title()
        value = value.replace('\\n', '\n')
        markdown += f"**{formatted_key}:**\n{value}\n"

    return markdown

def convert_report(report: ReportSchema) -> str:
    """
        Convert a ReportSchema object to a formatted markdown string.
        
        This function takes a ReportSchema object and formats it into a
        structured markdown document with appropriate section headers.
        
        Args:
            report (ReportSchema): Report object to convert
        
        Returns:
            str: Formatted markdown report
    """
    return f"### Identification & General Data\n{report.identification}\n\n### Inspection Methods\n{report.inspection_methods}\n\n### Condition Assessment\n{report.condition_assessment}\n\n### Documentation & Accessories\n{report.documentation_and_accessories}"

