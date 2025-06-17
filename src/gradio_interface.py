import gradio as gr
import requests
import os
from PIL import Image

def process_images(images):
    # First, return an immediate "In progress" message
    yield "Processing your images... Please wait."
    
    # Create a temporary directory to store uploaded images
    temp_dir = "./temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Check if images is a list (multiple images) or a single image
    if not isinstance(images, list):
        images = [images]
    
    # Save uploaded images to the temp directory
    image_paths = []
    for i, img_tuple in enumerate(images):
        # Gallery component returns a tuple with the image path as the first element
        img_path = img_tuple
        if isinstance(img_tuple, tuple):
            img_path = img_tuple[0]  # Extract the image path from the tuple
            
        # Generate a new path in the temp directory
        new_path = os.path.join(temp_dir, f"image_{i}.jpg")
        
        # Copy the image to the temp directory
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img.save(new_path)
            image_paths.append(new_path)
    
    if not image_paths:
        yield "No valid images were uploaded."
        return
    
    try:
        yield "Images uploaded. Waiting for AI to create report. This may take a few minutes..."
        
        # Send the directory path to your Flask endpoint
        response = requests.post(
            "http://127.0.0.1:5001/process_images",
            json={"images_dir": temp_dir}
        )
        
        # Print debug information
        print(f"Status code: {response.status_code}")
        print(f"Response text: {response.text}")
        
        # Clean up temporary files after processing
        for path in image_paths:
            if os.path.exists(path):
                os.remove(path)
        
        # Check if response is valid JSON
        if response.status_code == 200:
            try:
                result = response.json()["report"]
                yield result
            except Exception as json_err:
                yield f"Error parsing JSON response: {str(json_err)}\nResponse text: {response.text[:500]}"
        else:
            yield f"Server error: {response.status_code} - {response.text}"
    except Exception as e:
        yield f"Error processing images: {str(e)}"

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Image Processing Tool")
    
    with gr.Row():
        image_input = gr.Gallery(label="Upload Images")
    
    with gr.Row():
        submit_btn = gr.Button("Process Images")
    
    with gr.Row():
        output = gr.Markdown(label="Processing Results")
    
    # Connect the submit button to process images
    submit_btn.click(fn=process_images, inputs=[image_input], outputs=[output])

if __name__ == "__main__":
    demo.launch()