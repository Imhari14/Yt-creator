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
import base64
import io
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
        self.target_frames = 200  # Keep original value that works reliably with Gemini's payload limits
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

        # Get segment time range
        start_time = self.processed_frames[0][1] if self.processed_frames else 0
        end_time = self.processed_frames[-1][1] if self.processed_frames else 0
        time_range = f"{start_time:.1f}s to {end_time:.1f}s"

        # Prepare initial context with timeline info
        context = f"""
        I am analyzing this video segment from {time_range}. I have access to:
        1. Multiple frames from the segment with timestamps ({len(self.processed_frames)} frames)
        2. The transcript of the spoken content

        The video segment spans from {start_time:.1f} seconds to {end_time:.1f} seconds.
        When asked about specific timestamps, I will refer to this range.

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
            # Add frames with their timestamps
            for frame, timestamp in batch:
                frame_info = f"Frame at {timestamp:.1f}s"
                prompt_parts.append(frame_info)
                prompt_parts.append(frame)
            print(f"Added batch {batch_idx + 1}/{total_batches}")

        if transcript:
            prompt_parts.append(f"\nTranscript:\n{transcript}")

        # Initialize new chat
        print("Creating new chat session...")
        self.chat = self.chat_model.start_chat(history=[])
        await self._wait_for_rate_limit()
        
        # Send initial context
        print("Sending initial context to Gemini...")
        if 'token_counts' in st.session_state:
            st.session_state.token_counts['last_operation'] = 'Initialize Context'
        response = self.chat.send_message(prompt_parts)
        
        # Update state
        self.current_segment_hash = segment_hash
        self.is_initialized = True
        print("Chat context initialized and ready for queries")
        return response

    async def generate_response(self,
                           prompt: str,
                           frames: List[Tuple[np.ndarray, float]],
                           transcript: str = None,
                           uploaded_image: str = None,
                           uploaded_file: Dict = None) -> Tuple[Optional[str], Optional[Any]]:
        """Generate response using stored context"""
        try:
            # Ensure chat is initialized
            response = await self._initialize_chat_with_context(frames, transcript)
            
            # If initialization just happened, update token counts
            if response and 'token_counts' in st.session_state:
                self._update_initial_context_tokens(response)
            
            context_parts = [prompt]
            
            if uploaded_image:
                try:
                    image_bytes = base64.b64decode(uploaded_image)
                    image = Image.open(io.BytesIO(image_bytes))
                    context_parts.append(image)
                    context_parts.insert(0, "Please consider both the video content and the uploaded image in your response.")
                except Exception as e:
                    print(f"Error processing uploaded image: {str(e)}")
            
            if uploaded_file:
                try:
                    if uploaded_file["type"].startswith('text/'):
                        context_parts.insert(0,
                            f"Consider the following uploaded text content:\n{uploaded_file['content']}\n"
                            "Please include this information in your response along with the video content.")
                except Exception as e:
                    print(f"Error processing uploaded file: {str(e)}")
            
            print("Sending query with all context...")
            await self._wait_for_rate_limit()
            
            if 'token_counts' in st.session_state:
                st.session_state.token_counts['last_operation'] = 'Chat Response'
                
            response = self.chat.send_message(context_parts)
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

        Provide output in the following strict JSON format:
        [
            {
                "question": "Question text here",
                "answer": "Answer text here"
            },
            ...
        ]
        
        Generate 5-8 high-quality flashcards maintaining this exact JSON structure.
        """
        
        if 'token_counts' in st.session_state:
            st.session_state.token_counts['last_operation'] = 'Generate Flashcards'
            
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

        Provide output in the following strict JSON format:
        [
            {
                "question": "Question text here",
                "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
                "correct_answer": "Correct option text here"
            },
            ...
        ]
        
        Generate 5 challenging but fair questions maintaining this exact JSON structure.
        """
        
        if 'token_counts' in st.session_state:
            st.session_state.token_counts['last_operation'] = 'Generate Quiz'
            
        return await self._generate_learning_content(context, frames, transcript, "quiz")

    async def _generate_learning_content(self, context: str, frames: List[Tuple[np.ndarray, float]], 
                                      transcript: str, content_type: str) -> Tuple[Optional[List[Dict]], Optional[Any]]:
        """Generate learning content using stored context"""
        try:
            # Ensure chat is initialized
            response = await self._initialize_chat_with_context(frames, transcript)
            
            # If initialization just happened, update token counts
            if response and 'token_counts' in st.session_state:
                self._update_initial_context_tokens(response)
            
            # Generate content using existing context
            print(f"Generating {content_type} using existing context...")
            await self._wait_for_rate_limit()
            response = self.chat.send_message(context)
            content = response.text.strip()
            
            def clean_json_str(text):
                """Clean and extract valid JSON array from text"""
                # Find the first '[' and last ']'
                start_idx = text.find('[')
                end_idx = text.rfind(']') + 1
                if start_idx == -1 or end_idx == 0:
                    return None
                
                json_str = text[start_idx:end_idx]
                # Remove any markdown backticks
                json_str = json_str.replace('```json', '').replace('```', '')
                # Try to clean up common JSON formatting issues
                json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                json_str = ' '.join(json_str.split())  # Normalize whitespace
                return json_str

            # Try to extract and parse JSON
            json_str = clean_json_str(content)
            if json_str:
                try:
                    return json.loads(json_str), response
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {str(e)}")
                    print(f"Problematic JSON string: {json_str[:200]}...")
                    
            # Fallback to regex pattern if needed
            import re
            json_pattern = r'\[\s*\{[^]]*\}\s*\]'
            match = re.search(json_pattern, content, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    return json.loads(json_str), response
                except json.JSONDecodeError as e:
                    print(f"Error parsing regex match: {str(e)}")
                    
            print(f"Could not extract valid JSON from response: {content[:200]}...")
            return None, response
        except Exception as e:
            print(f"Error generating {content_type}: {str(e)}")
            return None, None
            
    def _update_initial_context_tokens(self, response):
        """Update token count information for initial context"""
        if hasattr(response, 'usage_metadata'):
            metadata = response.usage_metadata
            
            # Update initial context tokens
            st.session_state.token_counts['initial_context'] = {
                'prompt_tokens': metadata.prompt_token_count,
                'output_tokens': metadata.candidates_token_count,
                'total_tokens': metadata.total_token_count
            }
            
            # Reset query tokens when context changes
            st.session_state.token_counts['query_tokens'] = {
                'prompt_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            }
            
            # Update cumulative totals
            st.session_state.token_counts['prompt_token_count'] = metadata.prompt_token_count
            st.session_state.token_counts['candidates_token_count'] = metadata.candidates_token_count
            st.session_state.token_counts['total_token_count'] = metadata.total_token_count
            
            # Store current operation metrics
            st.session_state.token_counts['current_operation'] = {
                'name': st.session_state.token_counts['last_operation'],
                'prompt_tokens': metadata.prompt_token_count,
                'output_tokens': metadata.candidates_token_count,
                'total_tokens': metadata.total_token_count
            }

    def get_model_info(self) -> Optional[Any]:
        """Get model information including token limits"""
        try:
            if not self.model_info:
                self.model_info = genai.get_model("models/gemini-2.0-flash")
            return self.model_info
        except Exception as e:
            print(f"Error getting model info: {str(e)}")
            return None
            
    def cleanup(self):
        """Clean up resources when app is closed"""
        # Release any resources if needed
        self.chat = None
        self.processed_frames = None
        self.is_initialized = False
        print("GeminiHandler resources cleaned up")
