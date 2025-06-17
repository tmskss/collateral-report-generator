
"""
This module defines the state schema which is used in the graph, and the report output schema.
"""

from typing import TypedDict
from pydantic import BaseModel, Field

class ImageProcessingState(TypedDict):
    images_dir: str
    images: list[str]
    image_names: list[str]
    image_paths: list[str]
    features: list[str]
    aggregated_info: str
    final_report_markdown: str
    messages: list

class ReportSchema(BaseModel):
    identification: str = Field(description="Identification & general data of the asset")
    inspection_methods: str = Field(description="Methods and tools used during inspection")
    condition_assessment: str = Field(description="Overall condition assessment of the asset")
    documentation_and_accessories: str = Field(description="List of documentation and available accessories")