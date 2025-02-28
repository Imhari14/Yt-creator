import os
from typing import List, Tuple, Dict
import numpy as np
import cv2
import google.generativeai as genai
from PIL import Image
import time
import random
import asyncio
import json

class GeminiHandler:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=api_key)

        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        self.vision_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",  # CORRECTED MODEL NAME
            generation_config=self.generation_config
        )

        self.chat_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",  # CORRECTED MODEL NAME
            generation_config=self.generation_config
        )

        self.chat = self.chat_model.start_chat(history=[])
        self.max_frames = 8  # Maximum frames per batch
        # self.max_batches = 4 # Not needed
        self.retry_count = 3
        self.base_delay = 2
        self.last_request_time = 0
        self.min_request_interval = 1.0
        self.model_info = None  # Store model info for context window limits

    def _prepare_frames(self, frames: List[Tuple[np.ndarray, float]] = None) -> List[Tuple[Image.Image, float]]:
        """
        Simplifies frame preparation, processing up to max_frames.
        """
        if not frames:
            return []

        processed_frames = []
        for frame, timestamp in frames[:self.max_frames]:  # Limit to max_frames
            if frame is None:
                continue
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width = rgb_frame.shape[:2]
                if width > 640:
                    new_width = 640
                    new_height = int(height * (640 / width))
                    rgb_frame = cv2.resize(rgb_frame, (new_width, new_height))
                pil_image = Image.fromarray(rgb_frame)
                processed_frames.append((pil_image, timestamp))
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue  # Continue even if one frame fails

        return processed_frames


    async def _wait_for_rate_limit(self):
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last_request)
        self.last_request_time = time.time()

    async def _generate_with_context(self, prompt: str, frames: List[Tuple[np.ndarray, float]], transcript: str = None):
        """
        Generates content with visual and transcript context.
        Returns the response text and the raw response object for token counting
        """
        processed_frames = self._prepare_frames(frames)
        if not processed_frames:
             response = await self.chat_with_history(prompt)
             return response.text, response

        target_time = self._extract_target_time(prompt)

        # Different prompts based on whether timestamp is requested
        if target_time is not None:
            context = f"""
            Analyze this specific moment in the video.

            Available Information:
            Visual Content:
            - Multiple frames from this timestamp
            - Pay attention to what is shown and any UI elements

            Audio/Transcript:
            {transcript if transcript else 'No transcript available'}

            Question: {prompt}

            Provide a detailed analysis of this specific timestamp, including:
            - What is visually happening
            - What is being discussed
            - Important elements shown
            """
        else:
            context = f"""
            Analyze this video segment by combining both visual content and spoken information.

            Available Information:
            Visual Content:
            - Multiple frames showing the video content
            - Pay attention to what is shown and any UI elements

            Audio/Transcript:
            {transcript if transcript else 'No transcript available'}

            Question: {prompt}

            Provide a natural response that:
            - Combines what you see and what is being said
            - Focuses on the main points and content
            - Gives a clear understanding of what's happening
            Do not mention timestamps unless specifically asked about them.
            """

        prompt_parts = [context]
        prompt_parts.extend([frame for frame, _ in processed_frames])  # Add processed frames

        try:
            await self._wait_for_rate_limit()
            response = self.vision_model.generate_content(prompt_parts)
            return response.text, response  # Return both text and raw response

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None, None  # Return None for both text and raw response




    async def generate_flashcards(self, transcript: str = None, frames: List[Tuple[np.ndarray, float]] = None) -> List[Dict]:
        context = """
        Create educational flashcards based on this video segment.
        Combine visual content and spoken information to create comprehensive cards.

        Each flashcard should:
        1. Cover key concepts shown and discussed
        2. Include both visual and spoken information
        3. Focus on important points and details

        Format as JSON array of objects with 'question' and 'answer' fields.
        Generate 5-8 high-quality flashcards.
        Do not include timestamps in the cards.
        """

        return await self._generate_learning_content(context, frames, transcript, "flashcards")

    async def generate_quiz(self, transcript: str = None, frames: List[Tuple[np.ndarray, float]] = None) -> List[Dict]:
        context = """
        Create a multiple-choice quiz for this video segment.
        Combine visual content and spoken information in your questions.

        Each question should:
        1. Test understanding of key concepts shown and discussed
        2. Include both visual and spoken information
        3. Be clear and comprehensive

        Format as JSON array of objects with 'question', 'options', and 'correct_answer' fields.
        Generate 5 challenging but fair questions.
        Do not include timestamps in the questions.
        """

        return await self._generate_learning_content(context, frames, transcript, "quiz")

    async def _generate_learning_content(self, context: str, frames: List[Tuple[np.ndarray, float]],
                                      transcript: str, content_type: str):
        """
        Returns a tuple of (content, response) where content is the formatted data and
        response is the raw response object for token counting
        """
        processed_frames = self._prepare_frames(frames) #Simplified
        if not processed_frames:
            return None, None

        prompt_parts = [
            f"{context}\n\nTranscript:\n{transcript if transcript else 'No transcript available'}"
        ]
        prompt_parts.extend([frame for frame, _ in processed_frames]) # Add processed frames


        try:
            await self._wait_for_rate_limit()
            response = self.vision_model.generate_content(prompt_parts)
            content = response.text.strip()
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str), response  # Return content and raw response
        except Exception as e:
            print(f"Error generating {content_type}: {str(e)}")
            return None, None  # Return None for both content and raw response

    async def generate_response(self, prompt: str, frames: List[Tuple[np.ndarray, float]],
                             transcript: str = None):
        """Returns a tuple of (text_response, raw_response)"""
        response = await self._generate_with_context(prompt, frames, transcript)
        if not response or not response[0]:
            error_message = "I apologize, but I couldn't analyze the video segment properly. Please try again or ask a different question."
            return error_message, None
        return response  # Already returns (text, raw_response) tuple

    async def chat_with_history(self, message: str):
        delay = self.base_delay
        last_exception = None

        for _ in range(self.retry_count):
            try:
                await self._wait_for_rate_limit()
                response = self.chat.send_message(message)
                return response
            except Exception as e:
                last_exception = e
                if "429" in str(e):  # Check for rate limit error
                    jitter = random.uniform(0, 0.1) * delay
                    await asyncio.sleep(delay + jitter)
                    delay *= 2
                else:
                    # Re-raise if not a rate limit error
                    raise
        raise Exception(f"Failed after {self.retry_count} attempts. Last error: {str(last_exception)}")

    def _format_timestamp(self, seconds: float) -> str:
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def _extract_target_time(self, prompt: str) -> float:
        import re

        minute_patterns = [
            r'(\d+)(?:th|st|nd|rd)?\s*minute',
            r'(\d+):(\d{2})',
            r'(\d+)\s*min(?:ute)?s?\s+(\d+)\s*sec(?:ond)?s?',
            r'at the beginning',  # Added pattern for "beginning"
            r'towards the end',  # Added pattern for "end"
        ]

        for pattern in minute_patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                if pattern == r'at the beginning':
                    return 0.0  # Return 0 for "beginning"
                elif pattern == r'towards the end':
                    return float('inf')  # Return infinity for "end"
                elif len(match.groups()) == 1:
                    return float(match.group(1)) * 60
                elif len(match.groups()) == 2:
                    minutes = float(match.group(1))
                    seconds = float(match.group(2))
                    minutes = float(match.group(1))
                    seconds = float(match.group(2))
                    return minutes * 60 + seconds

        return None
        
    def get_model_info(self):
        """
        Get model information including token limits for the context window.
        Returns model metadata that includes input_token_limit and output_token_limit.
        """
        try:
            if not self.model_info:
                # Cache model info to avoid repeated API calls
                self.model_info = genai.get_model("models/gemini-1.5-flash")
            return self.model_info
        except Exception as e:
            print(f"Error getting model info: {str(e)}")
            # Return default values if API call fails
            return None
