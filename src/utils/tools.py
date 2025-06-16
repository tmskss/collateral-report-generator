from langchain.tools import tool
from model import LLM
from typing import List, Dict

from typing import List, Dict
from langchain.tools import tool

import yaml

@tool(parse_docstring=True)
def select_relevant_images(missing_info_summary: str) -> List[str]:
    """
    A tool to select relevant images based on their descriptions and a summary of missing information in the report.

    Args:
        missing_info_summary: A concise summary of the missing or incomplete information in the report.

    Returns:
        List[str]: A list of image paths that likely contain the missing information.
    """
    llm = LLM()

    with open("../data/runs/features.yaml", "r") as f:
        features: List[Dict[str, str]] = yaml.safe_load(f)
        descriptions_formatted = "\n\n".join(feature["image_path"] + ":\n" + feature["extracted_text"] for feature in features)

    system_prompt = (
        "You are a helpful assistant. Based on the missing information in a report, identify which image descriptions are likely to contain relevant data. Return ONLY a JSON list of image paths that should be examined further."
    )

    user_prompt = (
        f"""
        Missing information:\n{missing_info_summary}

        Image descriptions:\n{descriptions_formatted}

        Return a JSON list of image paths.
        """
    )

    result = llm.invoke(user_prompt, system_prompt)
    print(f"[DEBUG] Result of select_relevant_images with missing_info_summary: {missing_info_summary}\n{result}")
    try:
        import json
        return json.loads(result)
    except Exception:
        return result


@tool(parse_docstring=True)
def analyze_images(image_paths: List[str], information: str) -> str:
    """
    Analyzes a list of image files to extract missing information from them.

    Args:
        image_paths: A list of image file paths.
        information: A string containing the missing information that needs to be found.

    Returns:
        A string summarizing the found information from the images.
    
    Example Input:
        image_paths = ["front.jpg", "interior.jpg"]
        information = "Model of the car."
    """

    llm = LLM()

    aggregated_information = ""
    for path in image_paths:
        response = llm.find_information(path, information, aggregated_information)
        aggregated_information += f"\n{response}"

    system_prompt = "You are a helpful assistant who receives descriptions from images which contain missing information from a report."

    user_prompt = f"Give answers to the missing information based on the image descriptions.\n\nMissing information:\n{information}\n\nImage descriptions:\n{aggregated_information}"


    result = llm.invoke(user_prompt, system_prompt)
    print(f"[DEBUG] Result of analyze_images for information '{information}': {result}")
    return result

