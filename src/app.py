from pprint import pprint
from langgraph.graph import StateGraph, START, END
from utils.helpers import ImageProcessingState, load_images, describe_images, aggregate_info
from dotenv import load_dotenv
from utils.agent import RefiningAgent

import os


def create_default_state(images_dir: str) -> ImageProcessingState:
    """
    Create a default ImageProcessingState with initial values.

    Args:
        images_dir (str): The directory containing the images.

    Returns:
        ImageProcessingState: A state object with default values.
    """
    return {
        "images_dir": images_dir,
        "images": [],
        "image_names": [],
        "image_paths": [],
        "features": [],
        "aggregated_info": "",
        "final_report_markdown": ""
    }

def main():
    graph_builder = StateGraph(ImageProcessingState)

    # Define the basic data processing nodes
    graph_builder.add_node("load_images", load_images)
    graph_builder.add_node("extract_text_vision_model", describe_images)
    graph_builder.add_node("aggregate_info", aggregate_info)
    graph_builder.add_node("refine_agent", RefiningAgent())
    
    # Define the main flow
    graph_builder.add_edge(START, "load_images")
    graph_builder.add_edge("load_images", "extract_text_vision_model")
    graph_builder.add_edge("extract_text_vision_model", "aggregate_info")
    graph_builder.add_edge("aggregate_info", "refine_agent")
    graph_builder.add_edge("refine_agent", END)
    
    
    # Compile the graph
    app = graph_builder.compile()
    
    images_dir = os.getenv("IMAGES_DIR")
    state = create_default_state(images_dir)
    
    final_state = app.invoke(state)

    print("Final report:")
    pprint(final_state["final_report_markdown"])

    with open("/Users/tmskss/Development/nlp-homework-raiffeisen/data/runs/final_report.md", "w") as f:
        f.write(final_state["final_report_markdown"])


if __name__ == "__main__":
    load_dotenv()
    main()