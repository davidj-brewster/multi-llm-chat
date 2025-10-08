import os
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any
import io

try:
    import ffmpeg
except ImportError as e:
    ffmpeg = None
    raise

from model_clients import GeminiClient
from sandbox_manager import SandboxManager
from configdataclasses import DemoGenerationConfig

logger = logging.getLogger(__name__)


class DemoGenerator:
    """
    Generates a 'making-of' video demo from a conversation history between AI agents.
    """

    def __init__(self, gemini_client: GeminiClient, config: DemoGenerationConfig):
        """
        Initializes the DemoGenerator.

        Args:
            gemini_client (GeminiClient): An instance of the GeminiClient for TTS.
            config (DemoGenerationConfig): The configuration for demo generation.
        """
        self.gemini_client = gemini_client
        self.config = config
        self.sandbox_manager = SandboxManager(config=self.config)
        if ffmpeg is None:
            logger.warning("The 'ffmpeg-python' library is not installed. Video synthesis will not be possible.")

    def generate(self, conversation_history: List[Dict[str, Any]]) -> str:
        """
        Main orchestration method to create the demo video.
        """
        logger.info("Starting demo video generation process...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            narrative_script = self._analyze_conversation_narrative(conversation_history)
            if not narrative_script:
                logger.error("Failed to generate narrative script. Aborting demo generation.")
                return ""

            final_code = self._extract_final_code(conversation_history)
            image_data_list = []
            if final_code:
                try:
                    image_data_list = self.sandbox_manager.execute_code(final_code)
                except Exception as e:
                    logger.error(f"Code execution in sandbox failed: {e}. Proceeding without generated images.")
            else:
                logger.warning("No code found in conversation. The demo video will not have visual output from code execution.")

            try:
                audio_data = self._generate_narration(narrative_script)
                audio_path = temp_path / "narration.wav"
                audio_path.write_bytes(audio_data)
            except Exception as e:
                logger.error(f"Failed to generate narration audio: {e}. Aborting demo generation.")
                return ""

            output_video_path = temp_path / self.config.output_filename
            try:
                video_path = self._synthesize_video(image_data_list, audio_path, output_video_path)
                final_output_path = Path(os.getcwd()) / self.config.output_filename
                os.rename(video_path, final_output_path)
                logger.info(f"Demo video successfully generated at: {final_output_path}")
                return str(final_output_path)
            except Exception as e:
                logger.error(f"Failed to synthesize video: {e}. Aborting demo generation.")
                return ""

    def _analyze_conversation_narrative(self, conversation_history: List[Dict[str, Any]]) -> str:
        """
        Uses an LLM to analyze the conversation and generate a 'making-of' script.
        """
        logger.info(f"Analyzing conversation with model: {self.config.analysis_model}...")
        condensed_history = "\n".join([f"{msg['role']}: {msg['content'][:500]}" for msg in conversation_history])
        prompt = f"""
        You are a storyteller. Your task is to create a compelling narrative script for a 'making-of' video.
        This video will showcase a collaboration between a 'product manager' AI and an 'engineer' AI.
        Based on the following conversation history, create a voiceover script that tells the story of how the final code was developed.
        Highlight key moments, requested changes, feedback, and the evolution of the solution.
        The script should be engaging and explain the development process to a non-technical audience.
        Keep the script concise, suitable for a short video.

        Conversation History:
        ---
        {condensed_history}
        ---

        Please generate the voiceover script now.
        """
        try:
            response = self.gemini_client.generate_response(prompt=prompt, model_config={'model': self.config.analysis_model})
            logger.info(f"Generated narrative script: '{response[:100]}...'")
            return response
        except Exception as e:
            logger.error(f"Failed to generate narrative from LLM: {e}")
            return ""

    def _extract_final_code(self, conversation_history: List[Dict[str, Any]]) -> str:
        """
        Extracts the final block of Python code from the conversation history.
        """
        logger.info("Extracting final code from conversation...")
        for msg in reversed(conversation_history):
            if msg.get("role") == 'assistant' and "```python" in msg.get("content", ""):
                try:
                    code = msg["content"].split("```python")[1].split("```")[0]
                    logger.info("Found final code block from assistant.")
                    return code.strip()
                except IndexError:
                    continue
        logger.warning("No final Python code block found from assistant.")
        return ""

    def _generate_narration(self, script: str) -> bytes:
        """
        Converts the narrative script to audio using the Gemini TTS client.
        """
        logger.info(f"Generating narration audio using model: {self.config.tts_model}...")
        try:
            audio_data = self.gemini_client.generate_speech(text=script, model=self.config.tts_model)
            logger.info(f"Successfully generated {len(audio_data)} bytes of audio.")
            return audio_data
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise

    def _synthesize_video(self, image_data_list: List[bytes], audio_path: Path, output_path: Path) -> str:
        """
        Combines images and audio into a final MP4 video using ffmpeg.
        """
        if ffmpeg is None:
            raise ImportError("ffmpeg-python library is required for video synthesis.")
        
        temp_dir = audio_path.parent
        
        # Probe audio to get its duration
        try:
            probe = ffmpeg.probe(str(audio_path))
            audio_duration = float(probe['format']['duration'])
        except ffmpeg.Error as e:
            logger.error(f"Failed to probe audio file: {e.stderr.decode()}")
            raise

        video_input = None
        if not image_data_list:
            logger.warning("No images provided. Creating a black screen video.")
            video_input = ffmpeg.input(f'color=c=black:s=1280x720:d={audio_duration}', f='lavfi')
        elif len(image_data_list) == 1:
            logger.info("One image provided. Looping image for audio duration.")
            img_path = temp_dir / "image_001.png"
            img_path.write_bytes(image_data_list[0])
            video_input = ffmpeg.input(str(img_path), loop=1, t=audio_duration)
        else:
            logger.info(f"{len(image_data_list)} images provided. Creating video from image sequence.")
            # Calculate framerate to match audio duration
            framerate = len(image_data_list) / audio_duration
            image_files_path = str(temp_dir / "image_%03d.png")
            for i, img_data in enumerate(image_data_list):
                (temp_dir / f"image_{i:03d}.png").write_bytes(img_data)
            video_input = ffmpeg.input(image_files_path, framerate=framerate)

        audio_input = ffmpeg.input(str(audio_path))

        try:
            (
                ffmpeg
                .output(video_input, audio_input, str(output_path), vcodec='libx264', acodec='aac', pix_fmt='yuv420p', shortest=None)
                .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
            )
            logger.info(f"Video successfully synthesized at {output_path}")
            return str(output_path)
        except ffmpeg.Error as e:
            logger.error("ffmpeg error during video synthesis.")
            logger.error(f"ffmpeg stdout: {e.stdout.decode('utf-8', 'ignore')}")
            logger.error(f"ffmpeg stderr: {e.stderr.decode('utf-8', 'ignore')}")
            raise
