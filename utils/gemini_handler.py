import os
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import cv2
import google.generativeai as genai
from PIL import Image
import time
import random
import asyncio
import json
import math
import hashlib
import streamlit as st

class GeminiHandler:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=api_key)

        self.generation_config = {
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "stop_sequences": [],
        }

        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]

        # Initialize models
        self.vision_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        
        self.chat_model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )

        # Session state
        self.chat = None
        self.processed_frames = None
        self.current_segment_hash = None
        self.is_initialized = False
        
        # Configuration
        self.max_frames = 3000
        self.target_frames = 200
        self.frame_batch_size = 50
        self.retry_count = 3
        self.base_delay = 2
        self.last_request_time = 0
        self.min_request_interval = 1.0
        self.model_info = None

    def _get_segment_hash(self, frames: List[Tuple[np.ndarray, float]], transcript: str = None) -> str:
        """Generate a unique hash for the segment based on frame data and transcript"""
        if not frames:
            return None
            
        # Create a unique identifier using timestamps and frame count
        hasher = hashlib.md5()
        if frames:
            start_time = frames[0][1]
            end_time = frames[-1][1]
            frame_count = len(frames)
            hasher.update(f"{start_time}-{end_time}-{frame_count}".encode())
        if transcript:
            hasher.update(transcript.encode())
            
        return hasher.hexdigest()

    def _calculate_sampling_rate(self, frames: List[Tuple[np.ndarray, float]]) -> int:
        """Calculate adaptive sampling rate based on number of frames"""
        total_frames = len(frames)
        if total_frames <= self.target_frames:
            return 1  # Use all frames
        return math.ceil(total_frames / self.target_frames)

    def _prepare_frames(self, frames: List[Tuple[np.ndarray, float]]) -> List[Tuple[Image.Image, float]]:
        """Process frames with adaptive sampling"""
        if not frames:
            return []

        # Calculate sampling rate
        total_frames = len(frames)
        sampling_rate = math.ceil(total_frames / self.target_frames) if total_frames > self.target_frames else 1
        print(f"Processing {total_frames} frames with sampling rate 1/{sampling_rate}")

        processed_frames = []
        sampled_frames = frames[::sampling_rate]
        total_to_process = len(sampled_frames)

        for idx, (frame, timestamp) in enumerate(sampled_frames, 1):
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
                
                if idx % 10 == 0:
                    print(f"Initial setup: Processed {idx}/{total_to_process} frames")
            except Exception as e:
                print(f"Error processing frame {idx}: {str(e)}")
                continue

        print(f"Completed processing {len(processed_frames)} frames")
        return processed_frames

    async def _wait_for_rate_limit(self):
        """Wait between API requests to respect rate limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last_request)
        self.last_request_time = time.time()

    async def _initialize_chat_with_context(self, frames: List[Tuple[np.ndarray, float]], transcript: str = None):
        """Initialize chat with visual context if not already initialized for this segment"""
        # Generate hash for current segment
        segment_hash = self._get_segment_hash(frames, transcript)
        print(f"Current segment hash: {segment_hash}")
        print(f"Previous segment hash: {self.current_segment_hash}")
        print(f"Is chat initialized: {self.is_initialized}")
        
        # Check if we're already initialized for this segment
        if self.is_initialized and segment_hash == self.current_segment_hash and self.chat is not None:
            print("Using existing chat context - no reinitialization needed")
            return None

        print("Initializing new chat context (this may take a minute)...")
        
        # Process frames if not already processed for this segment
        if not self.processed_frames or segment_hash != self.current_segment_hash:
            self.processed_frames = self._prepare_frames(frames)
            print("Frame processing complete")

        # Prepare initial context
        context = """
        I am analyzing this video segment. I have access to:
        1. Multiple frames from the segment (sampled adaptively)
        2. The transcript of the spoken content

        I will maintain this context for our conversation about this video segment.
        """

        # Prepare prompt parts with batched frames
        prompt_parts = [context]
        total_batches = math.ceil(len(self.processed_frames) / self.frame_batch_size)
        
        print("Adding frames to context in batches...")
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.frame_batch_size
            end_idx = start_idx + self.frame_batch_size
            batch = self.processed_frames[start_idx:end_idx]
            prompt_parts.extend([frame for frame, _ in batch])
            print(f"Added batch {batch_idx + 1}/{total_batches}")

        if transcript:
            prompt_parts.append(f"\nTranscript:\n{transcript}")

        # Initialize new chat
        print("Creating new chat session...")
        self.chat = self.chat_model.start_chat(history=[])
        await self._wait_for_rate_limit()
        
        # Send initial context
        print("Sending initial context to Gemini...")
        st.session_state.token_counts['last_operation'] = 'Initialize Context'
        response = self.chat.send_message(prompt_parts)
        
        # Update state
        self.current_segment_hash = segment_hash
        self.is_initialized = True
        print("Chat context initialized and ready for queries")
        return response

    async def generate_response(self, prompt: str, frames: List[Tuple[np.ndarray, float]], transcript: str = None) -> Tuple[Optional[str], Optional[Any]]:
        """Generate response using stored context"""
        try:
            # Ensure chat is initialized
            await self._initialize_chat_with_context(frames, transcript)
            
            # Send query using existing context
            print("Sending query using existing context...")
            await self._wait_for_rate_limit()
            response = self.chat.send_message(prompt)
            return response.text, response
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None, None

    async def generate_flashcards(self, transcript: str = None, frames: List[Tuple[np.ndarray, float]] = None) -> Tuple[Optional[List[Dict]], Optional[Any]]:
        """Generate flashcards using stored context"""
        context = """
        Create educational flashcards based on this video segment.
        Each flashcard should:
        1. Cover key concepts shown and discussed
        2. Include both visual and spoken information
        3. Focus on important points and details
        4. Be detailed enough for learning but concise

        Format as JSON array of objects with 'question' and 'answer' fields.
        Generate 5-8 high-quality flashcards.
        """
        return await self._generate_learning_content(context, frames, transcript, "flashcards")

    async def generate_quiz(self, transcript: str = None, frames: List[Tuple[np.ndarray, float]] = None) -> Tuple[Optional[List[Dict]], Optional[Any]]:
        """Generate quiz using stored context"""
        context = """
        Create a multiple-choice quiz for this video segment.
        Each question should:
        1. Test understanding of key concepts
        2. Include both visual and spoken information
        3. Be clear and comprehensive
        4. Have carefully crafted distractors

        Format as JSON array of objects with 'question', 'options', and 'correct_answer' fields.
        Generate 5 challenging but fair questions.
        """
        return await self._generate_learning_content(context, frames, transcript, "quiz")

    async def _generate_learning_content(self, context: str, frames: List[Tuple[np.ndarray, float]], 
                                      transcript: str, content_type: str) -> Tuple[Optional[List[Dict]], Optional[Any]]:
        """Generate learning content using stored context"""
        try:
            # Ensure chat is initialized
            await self._initialize_chat_with_context(frames, transcript)
            
            # Generate content using existing context
            print(f"Generating {content_type} using existing context...")
            await self._wait_for_rate_limit()
            response = self.chat.send_message(context)
            content = response.text.strip()
            
            # Extract JSON content
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                return json.loads(json_str), response
            else:
                import re
                json_pattern = r'\[\s*\{.*\}\s*\]'
                match = re.search(json_pattern, content, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    return json.loads(json_str), response
                else:
                    print(f"Could not extract JSON from response: {content[:100]}...")
                    return None, response
        except Exception as e:
            print(f"Error generating {content_type}: {str(e)}")
            return None, None

    def get_model_info(self) -> Optional[Any]:
        """Get model information including token limits"""
        try:
            if not self.model_info:
                self.model_info = genai.get_model("models/gemini-2.0-flash")
            return self.model_info
        except Exception as e:
            print(f"Error getting model info: {str(e)}")
            return None