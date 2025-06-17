from openai import OpenAI
from utils.examples import OUTPUT_FORMAT
    
class LLM:
    """
    A wrapper class for OpenAI API interactions that handles various LLM operations.
    
    This class provides methods for interacting with OpenAI models, including
    text generation, image analysis, and structured report creation. It abstracts
    away the details of API calls and provides a simple interface for the rest
    of the application.
    """
    
    def __init__(self):
        """
        Initialize the LLM class with an OpenAI client.
        """
        self.client = OpenAI()
        
    def create_file(self, file_path):
        """
        Upload a file to OpenAI for vision-based processing.
        
        Args:
            file_path (str): Path to the image file to be uploaded
            
        Returns:
            str: The file ID assigned by OpenAI, used for subsequent API calls
        """
        with open(file_path, "rb") as file_content:
            result = self.client.files.create(
                file=file_content,
                purpose="vision",
            )
            return result.id
        
    def invoke(self, user_prompt: str, system_prompt: str, model: str = "gpt-4o-mini"):
        """
        Generate a text response using the OpenAI chat completions API.
        
        This method sends a system prompt and user prompt to the specified model
        and returns the generated text response.
        
        Args:
            user_prompt (str): The prompt text from the user's perspective
            system_prompt (str): Instructions for the model about its role and task
            model (str, optional): The OpenAI model to use. Defaults to "gpt-4o-mini"
            
        Returns:
            str: The generated text response from the model
        """
        resp = self.client.chat.completions.create(
            model = model,
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        return resp.choices[0].message.content
        
    def describe_image(self, img_path: str, additional_info: str | None = None, model: str = "gpt-4.1-mini") -> str:
        """
        Generate a descriptive text of an image using OpenAI's vision capabilities.
        
        This method uploads an image, sends it to the vision model with a prompt
        to describe the image, and returns the generated description. It can incorporate
        additional context from previous image descriptions.
        
        Args:
            img_path (str): Path to the image file to be described
            additional_info (str | None, optional): Additional context from previous images. Defaults to None
            model (str, optional): The OpenAI vision model to use. Defaults to "gpt-4.1-mini"
            
        Returns:
            str: The generated description of the image
        """
        file_id = self.create_file(img_path)
        additional_info_text = 'This is additional information generated from other images of the same object: ' + additional_info + '\nOnly include the new information in the description, and not the information from the previous images.' if additional_info != "" else ""
        prompt = f"Describe this image. You are analyzing an image to be included as collateral. Focus on the condition, brand, and specifications of the item. {additional_info_text}"

        response = self.client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "file_id": file_id,
                    },
                ],
            }],
        )

        return response.output_text
    
    def find_information(self, img_path: str, information: str, additional_info: str | None = None, model: str = "gpt-4.1-mini") -> str:
        """
        Extract specific information from an image using OpenAI's vision capabilities.
        
        This method uploads an image, sends it to the vision model with a prompt
        to find specific information in the image, and returns the extracted information.
        
        Args:
            img_path (str): Path to the image file to analyze
            information (str): Description of the specific information to extract
            additional_info (str | None, optional): Additional context from previous analyses. Defaults to None
            model (str, optional): The OpenAI vision model to use. Defaults to "gpt-4.1-mini"
            
        Returns:
            str: The extracted information from the image
        """
        file_id = self.create_file(img_path)

        additional_info_text = 'This is additional description generated from other images related to the same information: ' + additional_info + "\nYou can disregard this if it is not relevant, or doesn't contain the information you need." if additional_info != "" else ""
        prompt = f"Describe this image. You are analyzing an image to be included as collateral. You need to find this information in the image: {information}\n\n{additional_info_text}"

        response = self.client.responses.create(
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "file_id": file_id,
                    },
                ],
            }],
        )

        return response.output_text
    
    def create_report(self, user_prompt: str, system_prompt: str, model: str = "gpt-4o-mini"):
        """
        Generate a structured JSON report using the OpenAI chat completions API.
        
        This method is similar to invoke() but configures the response to be in
        a specific structured format defined by OUTPUT_FORMAT.
        
        Args:
            user_prompt (str): The prompt text from the user's perspective
            system_prompt (str): Instructions for the model about its role and task
            model (str, optional): The OpenAI model to use. Defaults to "gpt-4o-mini"
            
        Returns:
            str: The generated report in the specified format (typically JSON)
        """
        resp = self.client.chat.completions.create(
            model = model,
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            response_format=OUTPUT_FORMAT,
        )

        return resp.choices[0].message.content