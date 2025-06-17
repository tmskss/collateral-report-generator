from openai import OpenAI
from utils.examples import OUTPUT_FORMAT
    
class LLM:
    def __init__(self):
        self.client = OpenAI()
        
    def create_file(self, file_path):
        with open(file_path, "rb") as file_content:
            result = self.client.files.create(
                file=file_content,
                purpose="vision",
            )
            return result.id
        
    def invoke(self, user_prompt: str, system_prompt: str, model: str = "gpt-4o-mini"):
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
