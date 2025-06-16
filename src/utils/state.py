from typing import TypedDict

class ImageProcessingState(TypedDict):
    images_dir: str
    images: list[str]
    image_names: list[str]
    image_paths: list[str]
    features: list[str]
    aggregated_info: str
    final_report_markdown: str