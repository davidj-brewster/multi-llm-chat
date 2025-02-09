from bs4 import BeautifulSoup
import re

def format_code_blocks(html):
    def replace_code(match):
        code = match.group(1)
        return f'\n<pre><code>{code}</code></pre>\n'
    return re.sub(r'```python\n(.*?)(\n)?```', replace_code, html, flags=re.DOTALL)

with open("conversation.html", "r") as f_in, open("fixed_conversation.html", "w") as f_out:
    soup = BeautifulSoup(f_in, "html.parser")
    
    # Format <thinking> tags
    for thinking_tag in soup.find_all('thinking'):
        thinking_tag.wrap(soup.new_tag('pre'))

    # Format code blocks
    formatted_html = format_code_blocks(str(soup))
    
    f_out.write(formatted_html)
