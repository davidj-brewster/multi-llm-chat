"""Enhanced Gemini client with image and video support"""
import os
import time
import logging
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from google import genai
from google.genai import types

from model_clients import GeminiClient, BaseClient

logger = logging.getLogger(__name__)

@dataclass
class MediaFile:
    """Represents an uploaded media file"""
    name: str
    uri: str
    display_name: str
    state: str
    mime_type: str

class GeminiVisionClient(GeminiClient):
    """Extended Gemini client with media handling capabilities"""
    
    def __init__(self, mode: str, role: str, api_key: str, domain: str, 
                 model: str = "models/gemini-2-pro"):
        super().__init__(mode=mode, role=role, api_key=api_key, domain=domain, model=model)
        self.uploaded_files: Dict[str, MediaFile] = {}
        
    def upload_media(self, file_path: str, display_name: Optional[str] = None) -> Optional[MediaFile]:
        """Upload media file to Gemini File API"""
        try:
            display_name = display_name or os.path.basename(file_path)
            
            # Upload file
            file_response = genai.upload_file(
                path=file_path,
                display_name=display_name
            )
            
            # Create MediaFile object
            media_file = MediaFile(
                name=file_response.name,
                uri=file_response.uri,
                display_name=display_name,
                state="PROCESSING",
                mime_type=file_response.mime_type
            )
            
            # Store file reference
            self.uploaded_files[file_path] = media_file
            
            logger.info(f"Uploaded file '{display_name}' as: {file_response.uri}")
            return media_file
            
        except Exception as e:
            logger.error(f"Error uploading media file: {e}")
            return None
            
    def wait_for_processing(self, media_file: MediaFile, timeout: int = 300) -> bool:
        """Wait for media file processing to complete"""
        start_time = time.time()
        while media_file.state == "PROCESSING":
            if time.time() - start_time > timeout:
                logger.error(f"Timeout waiting for file {media_file.display_name} to process")
                return False
                
            time.sleep(10)
            file_info = genai.get_file(media_file.name)
            media_file.state = file_info.state.name
            
            if media_file.state == "FAILED":
                logger.error(f"File {media_file.display_name} processing failed")
                return False
                
        return media_file.state == "ACTIVE"
        
    def analyze_media(self, 
                     file_path: str,
                     prompt: str,
                     conversation_context: Optional[str] = None) -> Optional[str]:
        """Analyze image or video using Gemini vision models"""
        try:
            # Upload file if not already uploaded
            if file_path not in self.uploaded_files:
                media_file = self.upload_media(file_path)
                if not media_file:
                    return None
            else:
                media_file = self.uploaded_files[file_path]
                
            # Wait for processing to complete
            if not self.wait_for_processing(media_file):
                return None
                
            # Build conversation-aware prompt
            if conversation_context:
                prompt = f"In the context of our discussion about {conversation_context}, {prompt}"
                
            # Generate content with media
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    prompt,
                    media_file
                ],
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    maxOutputTokens=2048,
                    candidateCount=1,
                    tools=[types.Tool(
                        google_search=types.GoogleSearchRetrieval(
                            dynamic_retrieval_config=types.DynamicRetrievalConfig(
                                dynamic_threshold=0.6))
                    )]
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error analyzing media: {e}")
            return None
            
    def cleanup_media(self, file_path: Optional[str] = None):
        """Delete uploaded media files"""
        try:
            if file_path:
                # Delete specific file
                if file_path in self.uploaded_files:
                    media_file = self.uploaded_files[file_path]
                    genai.delete_file(media_file.name)
                    del self.uploaded_files[file_path]
                    logger.info(f"Deleted file {media_file.uri}")
            else:
                # Delete all files
                for path, media_file in self.uploaded_files.items():
                    genai.delete_file(media_file.name)
                    logger.info(f"Deleted file {media_file.uri}")
                self.uploaded_files.clear()
                
        except Exception as e:
            logger.error(f"Error cleaning up media files: {e}")
            
    def generate_response(self,
                         prompt: str,
                         system_instruction: str = None,
                         history: List[Dict[str, str]] = None,
                         role: str = None,
                         model_config: Optional[Any] = None) -> str:
        """Override to handle media content in prompts"""
        try:
            # Check if prompt contains media file reference
            if isinstance(prompt, dict) and "media_file" in prompt:
                return self.analyze_media(
                    file_path=prompt["media_file"],
                    prompt=prompt.get("text", "Analyze this media"),
                    conversation_context=self.domain
                )
            
            # Regular text prompt
            return super().generate_response(
                prompt=prompt,
                system_instruction=system_instruction,
                history=history,
                role=role,
                model_config=model_config
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
            
    def __del__(self):
        """Cleanup any remaining media files on deletion"""
        self.cleanup_media()