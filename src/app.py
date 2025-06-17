from pprint import pprint
from langgraph.graph import StateGraph, START, END
from utils.helpers import ImageProcessingState, load_images, describe_images, aggregate_info, finish_report
from dotenv import load_dotenv
from utils.agent import RefiningAgent

import os
from flask import Flask, request, jsonify

app = Flask(__name__)

load_dotenv()

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
        "final_report_markdown": "",
        "messages": []
    }

@app.route("/process_images", methods=["POST"])
def process_images():
    data = request.json
    images_dir = data.get("images_dir")

    if not images_dir or not os.path.exists(images_dir):
        return jsonify({"error": "Invalid image directory"}), 400

    graph_builder = StateGraph(ImageProcessingState)

    # Define the basic data processing nodes
    graph_builder.add_node("load_images", load_images)
    graph_builder.add_node("extract_text_vision_model", describe_images)
    graph_builder.add_node("aggregate_info", aggregate_info)
    graph_builder.add_node("refine_agent", RefiningAgent())
    graph_builder.add_node("finish_report", finish_report)
    
    # Define the main flow
    graph_builder.add_edge(START, "load_images")
    graph_builder.add_edge("load_images", "extract_text_vision_model")
    graph_builder.add_edge("extract_text_vision_model", "aggregate_info")
    graph_builder.add_edge("aggregate_info", "refine_agent")
    graph_builder.add_edge("refine_agent", "finish_report")
    graph_builder.add_edge("finish_report", END)
    
    
    # Compile the graph
    graph = graph_builder.compile()

    state = create_default_state(images_dir)
    
    final_state = graph.invoke(state)

    print("Final report:")
    pprint(final_state["final_report_markdown"])

    return jsonify({"report": final_state["final_report_markdown"]}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)