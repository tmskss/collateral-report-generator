from pprint import pprint
from langgraph.graph import StateGraph, START, END
from utils.helpers import ImageProcessingState, load_images, classify_images
from dotenv import load_dotenv

# Step 1: Load images


# Step 2: Classify images: containing meaningful text/no meaningful text
# def step_classify_images(state: ImageProcessingState):
#     # TODO: Implement image classification
#     state.classifications = ["class1" for _ in state.images]
#     return state

# # Step 3a: Extract features (what is on the image, possibility of web search for more info, use previous knowledge from other images)
# def step_extract_features(state: ImageProcessingState):
#     # TODO: Implement feature extraction
#     state.features = ["features" for _ in state.images]
#     return state

# # Step 3b: Extract text from images, use previous knowledge from other images
# def step_extract_text(state: ImageProcessingState):
#     # TODO: Implement text extraction
#     state.texts = ["text" for _ in state.images]
#     return state

# # Step 4: Aggregate information from features and texts
# def step_aggregate_information(state: ImageProcessingState):
#     # TODO: Implement aggregation logic
#     state.aggregated_info = {
#         "summary": "Aggregated info from features and texts"
#     }
#     return state

# # Step 5: Create description
# def step_create_description(state: ImageProcessingState):
#     # TODO: Implement description creation
#     state.description = f"Description based on: {state.aggregated_info}"
#     return state

# Build the agentic graph

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
        "classifications": [],
        "features": [],
        "texts": [],
        "aggregated_info": None,
        "description": None,
    }

def main():

    graph_builder = StateGraph(ImageProcessingState)

    graph_builder.add_node("load_images", load_images)
    graph_builder.add_node("classify_images", classify_images)
    # graph_builder.add_node("extract_features", step_extract_features)
    # graph_builder.add_node("extract_text", step_extract_text)
    # graph_builder.add_node("aggregate_information", step_aggregate_information)
    # graph_builder.add_node("create_description", step_create_description)

    # Define the flow
    graph_builder.add_edge(START, "load_images")
    graph_builder.add_edge("load_images", "classify_images")
    # graph_builder.add_edge("classify_images", "extract_features")
    # graph_builder.add_edge("classify_images", "extract_text")
    # graph_builder.add_edge("extract_features", "aggregate_information")
    # graph_builder.add_edge("extract_text", "aggregate_information")
    # graph_builder.add_edge("aggregate_information", "create_description")
    graph_builder.add_edge("classify_images", END)


    # Compile the graph_builder
    app = graph_builder.compile()

    images_dir = "/Users/tmskss/Development/nlp-homework-raiffeisen/data/157515_v2"
    state = create_default_state(images_dir)
    final_state = app.invoke(state)
    
    print("Image names:")
    pprint(final_state['image_names'])

    print("Classifications:")
    pprint(final_state['classifications'])
# Example usage
if __name__ == "__main__":
    load_dotenv()
    main()