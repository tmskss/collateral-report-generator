# Collateral Report Generator

This application automatically generates professional reports about objects by analyzing images. Using advanced computer vision and language models, it extracts key information from photos, creates structured reports, and refines them through an intelligent agent system.

## Features

- **Image Analysis**: Automatically extracts detailed information from multiple images of a collateral object
- **Report Generation**: Creates structured, professional reports with consistent formatting
- **Information Refinement**: Uses an AI agent to identify and fill information gaps
- **Web Search Integration**: Supplements image information with relevant web search results
- **Markdown Output**: Delivers reports in clean, formatted markdown

## System Architecture

The application follows a pipeline architecture:

1. **Image Loading**: Loads all images from a specified directory (input of the frontend)
2. **Feature Extraction**: Uses GPT-4.1-mini to analyze and describe each image one by one, continously gathering information and feeding it to the next image
3. **Information Aggregation**: Combines information from all images into a draft report
4. **Report Refinement**: Uses an agent-based approach to identify and fill missing information by looking at the images again looking for specific details
5. **Final Formatting**: Polishes the report for professional presentation

## Requirements

- OpenAI API key
- Docker

## Installation

1. Clone this repository:
```bash
git clone https://github.com/tmskss/collateral-report-generator.git
cd collateral-report-generator
```

2. Create a `.env` file based on the `.env.sample` file

## Usage

### Basic Usage

1. Run the application with:

```bash
docker-compose up --build
```
2. The backend will start at:

```bash
http://localhost:5001
```

3. The frontend will start at:
```bash
http://localhost:7860
```

4. You can upload images on the frontend and see the generated report at the bottom of the page after the backend is finished (might take a few minutes).


## Example Output

The generated reports include sections such as:

### **Identification:**
Detailed information about the object including make, model, and unique identifiers.

### **Inspection Methods:**
Description of how the object was inspected and what aspects were examined.

### **Condition Assessment:**
Detailed evaluation of the object's condition, noting any damage or wear.

### **Documentation & Accessories:**
List of accompanying documents and accessories included with the object.