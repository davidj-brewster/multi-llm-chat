Vision - OPENAI (Claude, o1)

Several OpenAI models have vision capabilities, meaning the models can take images as input and answer questions about them. Historically, language models were limited to a single input modality: text.

Currently, models that can take images as input include o1, gpt-4o, gpt-4o-mini, and gpt-4-turbo.

Quickstart
Images are made available to the model in two main ways: by passing a link to the image or by passing the Base64 encoded image directly in the request. Images can be passed in the user messages.

Analyze the content of an image
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0])
The model is best at answering general questions about what is present in the images. Although it understands the relationship between objects in images, it's not yet optimized to answer detailed questions about objects' locations in an image. For example, you can ask it what color a car is, or for some dinner ideas based on what's in your fridge, but if you show the model an image of a room and ask where the chair is, it may not answer the question correctly.

Keep model limitations in mind as you explore use cases for visual understanding.

Video understanding with vision
Learn how to use use GPT-4 with Vision to understand videos in the OpenAI Cookbook

Uploading Base64 encoded images
If you have an image or set of images locally, pass them to the model in Base64 encoded format:

import base64
from openai import OpenAI

client = OpenAI()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Path to your image
image_path = "path_to_your_image.jpg"

# Getting the Base64 string
base64_image = encode_image(image_path)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        }
    ],
)

print(response.choices[0])
Multiple image inputs
The Chat Completions API is capable of taking in and processing multiple image inputs, in Base64 encoded format or as an image URL. The model processes each image and uses information from all images to answer the question.

Multiple image inputs
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are in these images? Is there any difference between them?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)
print(response.choices[0])
Here, the model is shown two copies of the same image. It can answer questions about both images or each image independently.

Low or high fidelity image understanding
The detail parameter—which has three options, low, high, and auto—gives you control over how the model processes the image and generates its textual understanding. By default, the model will use the auto setting, which looks at the image input size and decides if it should use the low or high setting.

low enables the "low res" mode. The model receives a low-resolution 512px x 512px version of the image. It represents the image with a budget of 85 tokens. This allows the API to return faster responses and consume fewer input tokens for use cases that do not require high detail.
high enables "high res" mode, which first lets the model see the low-resolution image (using 85 tokens) and then creates detailed crops using 170 tokens for each 512px x 512px tile.
Choosing the detail level
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                        "detail": "high",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0].message.content)
Managing images
Unlike the Assistants API, the Chat Completions API isn't stateful. That means you have to manage messages (including images) you pass to the model. To pass the same image to the model multiple times, you have to pass the image each time you make a request to the API.

For long-running conversations, we suggest passing images via URLs instead of Base64. The latency of the model can also be improved by downsizing your images ahead of time to less than the maximum size.

Image size guidelines
We restrict image uploads to 20MB per image. Here are our image size expectations.

Mode	Expected image size
Low-res

512px x 512px

High res

Short side: less than 768px

Long side: less than 2,000px

After an image has been processed by the model, it's deleted from OpenAI servers and not retained. We do not use data uploaded via the OpenAI API to train our models.

