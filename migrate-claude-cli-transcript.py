import os
import re

def parse_conversation(txt_file: str) -> list:
    """
    Parses the conversation text file into a list of messages.
    Each message is a dictionary with 'role' and 'content'.
    
    Args:
        txt_file (str): Path to the conversation text file.
    
    Returns:
        list: List of message dictionaries.
    """
    with open(txt_file, "r", encoding="utf-8") as f:
        content = f.read()
w
    # Split messages by two consecutive blank lines
    raw_messages = re.split(r'\n\s*\n\s*\n', content.strip())
    print(raw_messages)

    messages = []
    for raw_msg in raw_messages:
        lines = raw_msg.strip().split('\n')
        if not lines:
            continue

        # Determine the role based on the first line
        first_line = lines[0].strip().lower()
        if first_line.startswith("🧑 human:"):
            role = "human"
            # Remove the role prefix from the first line
            lines[0] = lines[0].replace("🧑 human:", "").strip()
        elif first_line.startswith("assistant:"):
            role = "assistant"
            # Remove the role prefix from the first line
            lines[0] = lines[0].replace("assistant:", "").strip()
        elif first_line.startswith("🧑 human"):
            role = "human"
            # Remove the role prefix from the first line
            lines[0] = re.sub(r"🧑 human:? ?", "", lines[0], flags=re.IGNORECASE).strip()
        else:
            # Default to assistant if role not specified
            role = "assistant"

        # Combine the lines back into a single string
        content = "\n".join(lines).strip()
        messages.append({"role": role, "content": content})

    return messages

def generate_html(messages: list) -> str:
    """
    Generates HTML content from the list of messages.
    
    Args:
        messages (list): List of message dictionaries.
    
    Returns:
        str: HTML content as a string.
    """
    html_messages = []

    for msg in messages:
        if msg['role'] == 'human':
            template = """
        <div class="message human">
            <strong>🧑 Human:</strong>
            <div class="content">
                {content}
            </div>
        </div>
            """
        else:
            template = """
        <div class="message assistant">
            <strong>Assistant:</strong>
            <div class="content">
                {content}
            </div>
        </div>
            """
        # Escape HTML special characters to prevent rendering issues
        escaped_content = (msg['content']
                           .replace("&", "&amp;")
                           .replace("<", "&lt;")
                           .replace(">", "&gt;")
                           .replace('"', "&quot;")
                           .replace("'", "&#39;"))
        
        # Preserve line breaks
        formatted_content = escaped_content.replace('\n', '<br>')

        html_messages.append(template.format(content=formatted_content))

    # Combine all messages into a single HTML string
    return "\n".join(html_messages)

def append_messages_to_html(html_file: str, messages_html: str):
    """
    Appends message HTML blocks to the existing HTML file before the closing </body> tag.
    If the file doesn't exist, creates a new HTML file with basic structure.
    
    Args:
        html_file (str): Path to the HTML file.
        messages_html (str): HTML string of messages to append.
    """
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as hf:
            existing_html = hf.read()
    else:
        # Minimal HTML structure if file doesn't exist
        existing_html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Claude Conversation</title>
    <style>
        body {
            font-family: system-ui, -apple-system, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f3f4f6;
        }
        .message {
            margin: 20px 0;
            padding: 16px;
            border-radius: 8px;
        }
        .human {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
        }
        .assistant {
            background: #ffffff;
            border: 1px solid #e5e7eb;
        }
        .header {
            display: flex;
            align-items: center;
            margin-bottom: 12px;
            font-size: 14px;
            color: #4b5563;
        }
        .header-box {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            padding: 1rem;
            margin: 1rem 0;
            font-family: system-ui, -apple-system, sans-serif;
        }
        .content {
            font-size: 15px;
            line-height: 1.6;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-box">
            <h1>Claude Conversation</h1>
        </div>
    </div>
</body>
</html>
"""
    
    # Find the position to insert new messages (before </body>)
    insertion_point = existing_html.lower().rfind("</body>")
    if insertion_point != -1:
        new_html = (
            existing_html[:insertion_point]
            + "\n" + messages_html + "\n"
            + existing_html[insertion_point:]
        )
    else:
        # If </body> not found, append at the end
        new_html = existing_html + "\n" + messages_html

    with open(html_file, "w", encoding="utf-8") as hf:
        hf.write(new_html)

def migrate_conversation(txt_file: str, html_file: str):
    """
    Migrates conversation from a text file to an HTML file.
    
    Args:
        txt_file (str): Path to the conversation text file.
        html_file (str): Path to the HTML file.
    """
    messages = parse_conversation(txt_file)
    messages_html = generate_html(messages)
    append_messages_to_html(html_file, messages_html)
    print(f"Successfully migrated {len(messages)} messages from {txt_file} to {html_file}.")

if __name__ == "__main__":
    # Example usage:
    migrate_conversation("ClaudeConversation.txt", "ClaudeConversation.html")
