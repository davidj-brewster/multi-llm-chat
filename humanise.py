import re
from bs4 import BeautifulSoup

# Load the HTML file
input_file = "conversation.html"
output_file = "cleaned_conversation.html"

with open(input_file, "r", encoding="utf-8") as file:
    soup = BeautifulSoup(file, "html.parser")

with open(input_file, "r", encoding="utf-8") as file:
    content = file.read()

# Remove "Goal" and "Thinking" lines in human messages
content = re.sub(r"🧑(.*)human:.*(.*?(Goal|goal).*)?(.*?Thinking:.*)", "🧑 human:", content, flags=re.DOTALL)

# Save the cleaned conversation to a new file
with open(output_file, "w", encoding="utf-8") as file:
    file.write(content)

print(f"Cleaned text-based conversation saved to: {output_file}")

