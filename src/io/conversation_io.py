"""Conversation I/O operations for saving and loading conversations."""
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


async def save_conversation(
    conversation: List[Dict[str, str]],
    filename: str,
    human_model: str,
    ai_model: str,
    file_data: Dict[str, Any] = None,
    mode: str = None,
    signal_history: List[Dict[str, Any]] = None,
) -> None:
    """Save an AI conversation to an HTML file with proper encoding.

    Args:
    conversation (List[Dict[str, str]]): List of conversation messages with 'role' and 'content'
    filename (str): Output HTML file path
    human_model (str): Name of the human/user model
    ai_model (str): Name of the AI model
    file_data (Dict[str, Any], optional): Any associated file content (images, video, text)
    mode (str, optional): Conversation mode ('human-ai' or 'ai-ai')
    signal_history (List[Dict[str, Any]], optional): Per-turn NLP signal snapshots

    Raises:
    Exception: If saving fails or template is missing
    """
    try:
        with open("templates/conversation.html", "r") as f:
            template = f.read()

        conversation_html = ""

        # Add file content if present
        if file_data:
            # Handle multiple files (list of file data)
            if isinstance(file_data, list):
                for idx, file_item in enumerate(file_data):
                    if isinstance(file_item, dict) and "type" in file_item:
                        if file_item["type"] == "image" and "base64" in file_item:
                            # Add image to the conversation
                            mime_type = file_item.get("mime_type", "image/jpeg")
                            conversation_html += f'<div class="file-content"><h3>File {idx+1}: {file_item.get("path", "Image")}</h3>'
                            conversation_html += f'<img src="data:{mime_type};base64,{file_item["base64"]}" alt="Input image" style="max-width: 100%; max-height: 500px;"/></div>\n'
                        elif (
                            file_item["type"] == "video"
                            and "key_frames" in file_item
                            and file_item["key_frames"]
                        ):
                            # Add first frame of video
                            frame = file_item["key_frames"][0]
                            conversation_html += f'<div class="file-content"><h3>File {idx+1}: {file_item.get("path", "Video")} (First Frame)</h3>'
                            conversation_html += f'<img src="data:image/jpeg;base64,{frame["base64"]}" alt="Video frame" style="max-width: 100%; max-height: 500px;"/></div>\n'
                        elif (
                            file_item["type"] in ["text", "code"]
                            and "text_content" in file_item
                        ):
                            # Add text content
                            conversation_html += f'<div class="file-content"><h3>File {idx+1}: {file_item.get("path", "Text")}</h3><pre>{file_item["text_content"]}</pre></div>\n'
            # Handle single file (original implementation)
            elif isinstance(file_data, dict) and "type" in file_data:
                if file_data["type"] == "image" and "base64" in file_data:
                    # Add image to the conversation
                    mime_type = file_data.get("mime_type", "image/jpeg")
                    conversation_html += f'<div class="file-content"><h3>File: {file_data.get("path", "Image")}</h3>'
                    conversation_html += f'<img src="data:{mime_type};base64,{file_data["base64"]}" alt="Input image" style="max-width: 100%; max-height: 500px;"/></div>\n'
                elif (
                    file_data["type"] == "video"
                    and "key_frames" in file_data
                    and file_data["key_frames"]
                ):
                    # Add first frame of video
                    frame = file_data["key_frames"][0]
                    conversation_html += f'<div class="file-content"><h3>File: {file_data.get("path", "Video")} (First Frame)</h3>'
                    conversation_html += f'<img src="data:image/jpeg;base64,{frame["base64"]}" alt="Video frame" style="max-width: 100%; max-height: 500px;"/></div>\n'
                elif (
                    file_data["type"] in ["text", "code"]
                    and "text_content" in file_data
                ):
                    # Add text content
                    conversation_html += f'<div class="file-content"><h3>File: {file_data.get("path", "Text")}</h3><pre>{file_data["text_content"]}</pre></div>\n'

        for msg in conversation:
            role = msg["role"]
            content = msg.get("content", "")
            if isinstance(content, (list, dict)):
                content = str(content)

            if role == "system":
                conversation_html += (
                    f'<div class="system-message">{content} ({mode})</div>\n'
                )
            elif role in ["user", "human"]:
                conversation_html += f'<div class="human-message"><strong>Human ({human_model}):</strong> {content}</div>\n'
            elif role == "assistant":
                conversation_html += f'<div class="ai-message"><strong>AI ({ai_model}):</strong> {content}</div>\n'

            # Check if message contains file content (for multimodal messages)
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        # Extract image data
                        image_data = item.get("image_url", {}).get("url", "")
                        if image_data.startswith("data:"):
                            conversation_html += f'<div class="message-image"><img src="{image_data}" alt="Image in message" style="max-width: 100%; max-height: 300px;"/></div>\n'

        # Append NLP signal dashboard if signal history available
        if signal_history:
            conversation_html += _render_signal_dashboard(signal_history)

        with open(filename, "w") as f:
            f.write(template % {"conversation": conversation_html})
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")


def _render_signal_dashboard(signal_history: List[Dict[str, Any]]) -> str:
    """Render an inline HTML dashboard of NLP signals collected during conversation."""

    def _color(value: float, low: float, high: float, invert: bool = False) -> str:
        """Return green/yellow/red based on value thresholds."""
        if invert:
            if value <= low: return "#4caf50"
            elif value <= high: return "#ff9800"
            else: return "#f44336"
        else:
            if value >= high: return "#4caf50"
            elif value >= low: return "#ff9800"
            else: return "#f44336"

    html = """
<div class="signal-dashboard" style="margin-top:40px; padding:20px; background:#1a1a2e; border-radius:8px; font-family:monospace; color:#e0e0e0;">
<h2 style="color:#00d4ff; border-bottom:2px solid #00d4ff; padding-bottom:8px;">NLP Signal Dashboard</h2>
"""

    # Signal timeline table
    html += """
<h3 style="color:#aaa;">Signal Timeline</h3>
<div style="overflow-x:auto;">
<table style="border-collapse:collapse; width:100%; font-size:12px;">
<tr style="background:#16213e;">
  <th style="padding:6px; border:1px solid #333; color:#00d4ff;">Turn</th>
  <th style="padding:6px; border:1px solid #333; color:#00d4ff;">Role</th>
  <th style="padding:6px; border:1px solid #333; color:#00d4ff;">Flesch</th>
  <th style="padding:6px; border:1px solid #333; color:#00d4ff;">Fog</th>
  <th style="padding:6px; border:1px solid #333; color:#00d4ff;">Vocab</th>
  <th style="padding:6px; border:1px solid #333; color:#00d4ff;">Variety</th>
  <th style="padding:6px; border:1px solid #333; color:#00d4ff;">Repet.</th>
  <th style="padding:6px; border:1px solid #333; color:#00d4ff;">Agree.</th>
  <th style="padding:6px; border:1px solid #333; color:#00d4ff;">Form.</th>
  <th style="padding:6px; border:1px solid #333; color:#00d4ff;">Coher.</th>
  <th style="padding:6px; border:1px solid #333; color:#00d4ff;">Phase</th>
  <th style="padding:6px; border:1px solid #333; color:#00d4ff;">Interventions</th>
</tr>
"""

    for entry in signal_history:
        interventions_text = entry.get("interventions", "").replace("\n", "<br>")[:200]
        html += f"""<tr style="background:#0f3460;">
  <td style="padding:4px; border:1px solid #333; text-align:center;">{entry.get('turn', '?')}</td>
  <td style="padding:4px; border:1px solid #333;">{entry.get('role', '?')} ({entry.get('model', '')[:20]})</td>
  <td style="padding:4px; border:1px solid #333; background:{_color(entry.get('flesch', 50), 40, 60)}; color:#000; text-align:center;">{entry.get('flesch', 0):.1f}</td>
  <td style="padding:4px; border:1px solid #333; background:{_color(entry.get('fog', 12), 10, 14, invert=True)}; color:#000; text-align:center;">{entry.get('fog', 0):.1f}</td>
  <td style="padding:4px; border:1px solid #333; background:{_color(entry.get('vocabulary_richness', 0.5), 0.35, 0.6)}; color:#000; text-align:center;">{entry.get('vocabulary_richness', 0):.2f}</td>
  <td style="padding:4px; border:1px solid #333; background:{_color(entry.get('sentence_variety', 0.5), 0.15, 0.4)}; color:#000; text-align:center;">{entry.get('sentence_variety', 0):.2f}</td>
  <td style="padding:4px; border:1px solid #333; background:{_color(entry.get('repetition', 0), 0.3, 0.6, invert=True)}; color:#000; text-align:center;">{entry.get('repetition', 0):.2f}</td>
  <td style="padding:4px; border:1px solid #333; background:{_color(entry.get('agreement', 0), 0.4, 0.7, invert=True)}; color:#000; text-align:center;">{entry.get('agreement', 0):.2f}</td>
  <td style="padding:4px; border:1px solid #333; background:{_color(entry.get('formulaic', 0), 0.3, 0.6, invert=True)}; color:#000; text-align:center;">{entry.get('formulaic', 0):.2f}</td>
  <td style="padding:4px; border:1px solid #333; background:{_color(entry.get('coherence', 1), 0.3, 0.6)}; color:#000; text-align:center;">{entry.get('coherence', 0):.2f}</td>
  <td style="padding:4px; border:1px solid #333; text-align:center;">{entry.get('phase', 0):.2f}</td>
  <td style="padding:4px; border:1px solid #333; font-size:10px; color:#ff9800;">{interventions_text if interventions_text else '<span style="color:#4caf50;">none</span>'}</td>
</tr>
"""

    html += "</table></div>"

    # Per-participant summary
    participants = {}
    for entry in signal_history:
        role = entry.get("role", "unknown")
        if role not in participants:
            participants[role] = {"flesch": [], "fog": [], "vocab": [], "variety": [], "rep": [], "form": [], "model": entry.get("model", "")}
        participants[role]["flesch"].append(entry.get("flesch", 50))
        participants[role]["fog"].append(entry.get("fog", 12))
        participants[role]["vocab"].append(entry.get("vocabulary_richness", 0.5))
        participants[role]["variety"].append(entry.get("sentence_variety", 0.5))
        participants[role]["rep"].append(entry.get("repetition", 0))
        participants[role]["form"].append(entry.get("formulaic", 0))

    if participants:
        html += """<h3 style="color:#aaa; margin-top:20px;">Per-Participant NLP Profile</h3>
<table style="border-collapse:collapse; width:60%; font-size:13px;">
<tr style="background:#16213e;">
  <th style="padding:6px; border:1px solid #333; color:#00d4ff;">Participant</th>
  <th style="padding:6px; border:1px solid #333; color:#00d4ff;">Avg Flesch</th>
  <th style="padding:6px; border:1px solid #333; color:#00d4ff;">Avg Fog</th>
  <th style="padding:6px; border:1px solid #333; color:#00d4ff;">Avg Vocab</th>
  <th style="padding:6px; border:1px solid #333; color:#00d4ff;">Avg Variety</th>
  <th style="padding:6px; border:1px solid #333; color:#00d4ff;">Avg Repet.</th>
  <th style="padding:6px; border:1px solid #333; color:#00d4ff;">Avg Formulaic</th>
</tr>
"""
        for role, data in participants.items():
            avg = lambda vals: sum(vals) / len(vals) if vals else 0
            html += f"""<tr style="background:#0f3460;">
  <td style="padding:4px; border:1px solid #333;">{role} ({data['model'][:25]})</td>
  <td style="padding:4px; border:1px solid #333; text-align:center;">{avg(data['flesch']):.1f}</td>
  <td style="padding:4px; border:1px solid #333; text-align:center;">{avg(data['fog']):.1f}</td>
  <td style="padding:4px; border:1px solid #333; text-align:center;">{avg(data['vocab']):.2f}</td>
  <td style="padding:4px; border:1px solid #333; text-align:center;">{avg(data['variety']):.2f}</td>
  <td style="padding:4px; border:1px solid #333; text-align:center;">{avg(data['rep']):.2f}</td>
  <td style="padding:4px; border:1px solid #333; text-align:center;">{avg(data['form']):.2f}</td>
</tr>
"""
        html += "</table>"

    html += "</div>"
    return html


def _sanitize_filename_part(prompt: str) -> str:
    """
    Convert spaces, non-ASCII, and punctuation to underscores,
    then trim to something reasonable such as 30 characters.
    """
    # Remove non-alphanumeric/punctuation
    sanitized = re.sub(r"[^\w\s-]", "", prompt)
    # Convert spaces to underscores
    sanitized = re.sub(r"\s+", "_", sanitized.strip())
    # Limit length
    return sanitized[:50]
