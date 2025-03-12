import os
import streamlit as st
from dotenv import load_dotenv
from utils.video_processor import VideoProcessor
from utils.transcript_handler import TranscriptHandler
from utils.gemini_handler import GeminiHandler
import asyncio
import time
import base64
from PIL import Image
import io
import random
import tempfile

# Configure Streamlit page
st.set_page_config(
    page_title="Video Learning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Initialize handlers in session state for persistence
if 'handlers' not in st.session_state:
    st.session_state.handlers = {
        'video_processor': VideoProcessor(),
        'transcript_handler': TranscriptHandler(),
        'gemini_handler': GeminiHandler()
    }

# Use handlers from session state
video_processor = st.session_state.handlers['video_processor']
transcript_handler = st.session_state.handlers['transcript_handler']
gemini_handler = st.session_state.handlers['gemini_handler']

# Re-initialize gemini_handler if last segment changed
if ('last_segment' not in st.session_state or 
    st.session_state.get('current_segment') != st.session_state.get('last_segment')):
    st.session_state.handlers['gemini_handler'] = GeminiHandler()
    gemini_handler = st.session_state.handlers['gemini_handler']
    if st.session_state.get('current_segment'):
        st.session_state.last_segment = st.session_state.current_segment

# Custom CSS for better UI
st.markdown("""
<style>
    .token-info-container {
        position: fixed;
        top: 70px;
        right: 20px;
        background-color: #333;
        color: white;
        border-radius: 5px;
        padding: 8px 12px;
        width: auto;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        z-index: 1000;
        font-size: 0.8rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .token-simple-title {
        font-weight: bold;
        margin-bottom: 3px;
        font-size: 0.85rem;
    }
    .token-simple-metric {
        display: inline-block;
        margin-right: 8px;
        padding: 2px 5px;
        background-color: #444;
        border-radius: 3px;
    }
    .token-simple-value {
        font-weight: bold;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'video_info' not in st.session_state:
    st.session_state.video_info = None
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'transcript_text' not in st.session_state:
    st.session_state.transcript_text = None
if 'current_frames' not in st.session_state:
    st.session_state.current_frames = []
if 'current_segment' not in st.session_state:
    st.session_state.current_segment = None
if 'flashcards' not in st.session_state:
    st.session_state.flashcards = []
if 'quiz' not in st.session_state:
    st.session_state.quiz = None
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0
if 'user_answers' not in st.session_state:
    st.session_state.user_answers = {}
if 'video_source' not in st.session_state:
    st.session_state.video_source = None
if 'local_video_path' not in st.session_state:
    st.session_state.local_video_path = None
if 'token_counts' not in st.session_state:
    st.session_state.token_counts = {
        'prompt_token_count': 0,
        'candidates_token_count': 0,
        'total_token_count': 0,
        'last_operation': '',
        'initial_context': {
            'prompt_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        },
        'query_tokens': {
            'prompt_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        },
        'context_window': {
            'input_limit': 1000000,  # Default for Gemini 2.0 Flash
            'output_limit': 16000,    # Default for Gemini 2.0 Flash
        }
    }
if 'segment_interval' not in st.session_state:
    st.session_state.segment_interval = 5  # Default 5 minutes (in minutes)
if 'segments' not in st.session_state:
    st.session_state.segments = []

def format_timestamp(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def get_segment_options(duration, interval_minutes):
    segment_length = interval_minutes * 60
    segments = []
    current_time = 0
    while current_time < duration:
        end_time = min(current_time + segment_length, duration)
        segment_label = f"{format_timestamp(current_time)} - {format_timestamp(end_time)}"
        segments.append({
            "label": segment_label,
            "start": current_time,
            "end": end_time
        })
        current_time = end_time
    return segments

def update_token_counts(response):
    """Update token count information from the Gemini API response with separate tracking"""
    if hasattr(response, 'usage_metadata'):
        metadata = response.usage_metadata
        
        # Initialize token tracking if not exists
        if 'initial_context' not in st.session_state.token_counts:
            st.session_state.token_counts['initial_context'] = {
                'prompt_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            }
        if 'query_tokens' not in st.session_state.token_counts:
            st.session_state.token_counts['query_tokens'] = {
                'prompt_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            }

        # Track tokens based on operation type
        if st.session_state.token_counts['last_operation'] == 'Initialize Context':
            # Store initial context tokens
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
        else:
            # Add only new query tokens (excluding context)
            st.session_state.token_counts['query_tokens']['prompt_tokens'] += metadata.prompt_token_count
            st.session_state.token_counts['query_tokens']['output_tokens'] += metadata.candidates_token_count
            st.session_state.token_counts['query_tokens']['total_tokens'] += metadata.total_token_count

        # Update cumulative totals
        st.session_state.token_counts['prompt_token_count'] = (
            st.session_state.token_counts['initial_context']['prompt_tokens'] +
            st.session_state.token_counts['query_tokens']['prompt_tokens']
        )
        st.session_state.token_counts['candidates_token_count'] = (
            st.session_state.token_counts['initial_context']['output_tokens'] +
            st.session_state.token_counts['query_tokens']['output_tokens']
        )
        st.session_state.token_counts['total_token_count'] = (
            st.session_state.token_counts['initial_context']['total_tokens'] +
            st.session_state.token_counts['query_tokens']['total_tokens']
        )
        
        # Store current operation metrics
        st.session_state.token_counts['current_operation'] = {
            'name': st.session_state.token_counts['last_operation'],
            'prompt_tokens': metadata.prompt_token_count,
            'output_tokens': metadata.candidates_token_count,
            'total_tokens': metadata.total_token_count
        }
    
    # Update context window limits
    model_info = gemini_handler.get_model_info()
    if model_info:
        st.session_state.token_counts['context_window']['input_limit'] = model_info.input_token_limit
        st.session_state.token_counts['context_window']['output_limit'] = model_info.output_token_limit

def parse_transcript_file(file_content):
    try:
        lines = file_content.strip().split("\n")
        parsed_transcript = []
        current_time = 0
        for line in lines:
            if not line.strip():
                continue
            duration = 5
            entry = {
                'text': line.strip(),
                'start': current_time,
                'duration': duration
            }
            parsed_transcript.append(entry)
            current_time += duration
        return parsed_transcript
    except Exception as e:
        st.error(f"Error parsing transcript file: {str(e)}")
        return []

def get_transcript_text_for_segment(transcript_text, start, end):
    if not transcript_text:
        return ""
    lines = transcript_text.strip().split('\n')
    if hasattr(st.session_state, 'video_info') and st.session_state.video_info:
        total_duration = st.session_state.video_info['duration']
        if total_duration > 0 and len(lines) > 0:
            time_per_line = total_duration / len(lines)
            start_idx = int(start / time_per_line)
            end_idx = int(end / time_per_line) + 1
            segment_lines = lines[start_idx:end_idx]
            return " ".join(segment_lines)
    if len(lines) > 20:
        segment_percentage = (end - start) / st.session_state.video_info['duration']
        num_lines = max(10, int(len(lines) * segment_percentage))
        return " ".join(lines[:num_lines])
    return transcript_text

def display_token_info():
    """Display detailed token count information"""
    token_info = st.session_state.token_counts
    token_info_html = f"""
    <div class="token-info-container">
        <div class="token-simple-title">Token Usage</div>
        <span class="token-simple-metric">Context: <span class="token-simple-value">{token_info['initial_context'].get('total_tokens', 0):,}</span></span>
        <span class="token-simple-metric">Queries: <span class="token-simple-value">{token_info['query_tokens'].get('total_tokens', 0):,}</span></span>
        <span class="token-simple-metric">Total: <span class="token-simple-value">{token_info['total_token_count']:,}</span></span>
    </div>
    """
    st.markdown(token_info_html, unsafe_allow_html=True)

# Sidebar for video URL input and learning tools
with st.sidebar:
    st.image("https://fonts.gstatic.com/s/i/short-term/release/materialsymbolsoutlined/school/default/48px.svg")
    st.title("Learning Tools")

    # Video Input Section
    st.header("📹 Video Input")
    video_source = st.radio(
        "Select video source:",
        options=["YouTube", "Local MP4"],
        index=0 if st.session_state.video_source == "youtube" or st.session_state.video_source is None else 1
    )

    if video_source == "YouTube":
        st.session_state.video_source = "youtube"
        video_url = st.text_input("Enter YouTube URL")
        
        if st.button("Load YouTube Video", use_container_width=True):
            with st.spinner("Processing YouTube video..."):
                try:
                    video_info = video_processor.download_video(video_url)
                    st.session_state.video_info = video_info

                    video_id = transcript_handler.extract_video_id(video_url)
                    available_transcripts = transcript_handler.get_available_transcripts(video_id)
                    st.session_state.available_transcripts = available_transcripts

                    st.session_state.segments = get_segment_options(
                        video_info['duration'],
                        st.session_state.segment_interval
                    )
                    
                    st.session_state.local_video_path = None
                    st.session_state.transcript_text = None

                    st.success("YouTube video loaded successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.video_info = None
        
        # YouTube Transcript Selection
        if st.session_state.video_source == "youtube" and hasattr(st.session_state, 'available_transcripts') and st.session_state.available_transcripts:
            st.header("📝 YouTube Transcript")
            transcript_options = list(st.session_state.available_transcripts.keys())
            selected_option = st.selectbox(
                "Select language:",
                options=transcript_options
            )

            if st.button("Load Transcript", use_container_width=True):
                try:
                    lang_code = st.session_state.available_transcripts[selected_option]['code']
                    video_id = transcript_handler.extract_video_id(video_url)
                    full_transcript = transcript_handler.get_transcript(video_id, [lang_code])

                    if st.session_state.current_segment:
                        segment = st.session_state.current_segment
                        st.session_state.transcript = transcript_handler.get_transcript_for_chunk(
                            full_transcript,
                            segment["start"],
                            segment["end"]
                        )
                    else:
                        st.session_state.transcript = full_transcript
                    st.success(f"Transcript loaded!")
                except Exception as e:
                    st.error("Failed to load transcript")
                    st.session_state.transcript = None
                    
    else:  # Local MP4
        st.session_state.video_source = "local"
        uploaded_video = st.file_uploader("Upload MP4 video file", type=["mp4"])
        uploaded_transcript = st.file_uploader("Upload transcript text file", type=["txt"])
        
        if uploaded_video is not None and uploaded_transcript is not None:
            if st.button("Load Local Video", use_container_width=True):
                with st.spinner("Processing local video..."):
                    try:
                        temp_dir = tempfile.mkdtemp()
                        temp_video_path = os.path.join(temp_dir, "uploaded_video.mp4")
                        
                        with open(temp_video_path, "wb") as f:
                            f.write(uploaded_video.getbuffer())
                        
                        video_info = video_processor.process_local_video(temp_video_path, uploaded_video.name)
                        st.session_state.video_info = video_info
                        st.session_state.local_video_path = temp_video_path
                        
                        transcript_text = uploaded_transcript.getvalue().decode("utf-8")
                        st.session_state.transcript_text = transcript_text
                        
                        parsed_transcript = parse_transcript_file(transcript_text)
                        st.session_state.transcript = parsed_transcript
                        
                        st.session_state.segments = get_segment_options(
                            video_info['duration'],
                            st.session_state.segment_interval
                        )
                        
                        st.success("Local video and transcript loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading local video: {str(e)}")
                        st.session_state.video_info = None
                        st.session_state.local_video_path = None
                        st.session_state.transcript_text = None

    # Segment Settings
    st.subheader("⏰ Segment Settings")
    segment_interval = st.slider(
        "Segment Interval (minutes):",
        min_value=1,
        max_value=25,
        value=st.session_state.segment_interval,
        step=1
    )
    if segment_interval != st.session_state.segment_interval:
        st.session_state.segment_interval = segment_interval
        if st.session_state.video_info:
            st.session_state.segments = get_segment_options(
                st.session_state.video_info['duration'],
                st.session_state.segment_interval
            )
    
    # Segment Selection
    if st.session_state.video_info:
        st.header("🎯 Video Segment")
        segments = st.session_state.segments
        segment_labels = [s["label"] for s in segments]
        selected_segment_idx = st.selectbox(
            "Select video segment:",
            range(len(segment_labels)),
            format_func=lambda x: segment_labels[x]
        )

        if st.button("Load Segment", use_container_width=True):
            with st.spinner("Loading segment..."):
                selected_segment = segments[selected_segment_idx]
                if st.session_state.video_source == "youtube":
                    frames = video_processor.extract_frames(
                        st.session_state.video_info['url'],
                        selected_segment["start"],
                        selected_segment["end"]
                    )
                else:  # local video
                    frames = video_processor.extract_frames_from_local(
                        st.session_state.local_video_path,
                        selected_segment["start"],
                        selected_segment["end"]
                    )
                    
                st.session_state.current_frames = frames
                st.session_state.current_segment = selected_segment
                
                if st.session_state.video_source == "youtube" and st.session_state.transcript:
                    st.session_state.transcript = transcript_handler.get_transcript_for_chunk(
                        st.session_state.transcript,
                        selected_segment["start"],
                        selected_segment["end"]
                    )
                elif st.session_state.video_source == "local" and st.session_state.transcript_text:
                    segment_transcript = get_transcript_text_for_segment(
                        st.session_state.transcript_text,
                        selected_segment["start"],
                        selected_segment["end"]
                    )
                    st.session_state.transcript = segment_transcript
                    
                st.success(f"Loaded segment {selected_segment['label']}")

        # Learning Tools
        if st.session_state.current_frames:
            st.header("📚 Learning Tools")

            if st.button("Generate Flashcards", use_container_width=True):
                with st.spinner("Generating flashcards..."):
                    try:
                        flashcards_response = asyncio.run(gemini_handler.generate_flashcards(
                            st.session_state.transcript,
                            st.session_state.current_frames
                        ))
                        
                        if flashcards_response:
                            flashcards, response = flashcards_response
                            st.session_state.flashcards = flashcards
                            update_token_counts(response)
                            st.session_state.token_counts['last_operation'] = 'Generate Flashcards'
                            st.success("Flashcards generated!")
                        else:
                            st.warning("Could not generate flashcards. Please try again.")
                    except Exception as e:
                        st.error(f"Error generating flashcards: {str(e)}")

            if st.button("Generate Quiz", use_container_width=True):
                with st.spinner("Generating quiz..."):
                    try:
                        quiz_response = asyncio.run(gemini_handler.generate_quiz(
                            st.session_state.transcript,
                            st.session_state.current_frames
                        ))
                        
                        if quiz_response:
                            quiz, response = quiz_response
                            st.session_state.quiz = quiz
                            st.session_state.quiz_score = 0
                            st.session_state.user_answers = {}  # Reset user answers
                            update_token_counts(response)
                            st.session_state.token_counts['last_operation'] = 'Generate Quiz'
                            st.success("Quiz generated!")
                        else:
                            st.warning("Could not generate quiz. Please try again.")
                    except Exception as e:
                        st.error(f"Error generating quiz: {str(e)}")

# Main content area
if st.session_state.video_info:
    st.header("📺 Video Player")
    video_title = st.session_state.video_info['title']
    st.markdown(f"### {video_title}")

    if st.session_state.current_segment:
        segment = st.session_state.current_segment
        st.markdown(f"**Current Segment:** {segment['label']}")

    if st.session_state.video_source == "youtube":
        st.video(st.session_state.video_info["url"])
    else:  # local video
        with open(st.session_state.local_video_path, "rb") as video_file:
            video_bytes = video_file.read()
        st.video(video_bytes)

    # Learning Materials Section
    col1, col2 = st.columns(2)

    # Flashcards Column
    with col1:
        st.header("📝 Flashcards")
        if st.session_state.flashcards:
            for i, card in enumerate(st.session_state.flashcards):
                with st.expander(f"Card {i+1}: {card['question'][:50]}...", expanded=False):
                    st.write("**Question:**", card['question'])
                    st.write("**Answer:**", card['answer'])
        else:
            st.info("Select a segment and generate flashcards to study!")

    # Quiz Column
    with col2:
        st.header("📋 Quiz")
        if st.session_state.quiz:
            for i, question in enumerate(st.session_state.quiz):
                st.subheader(f"Q{i+1}: {question['question']}")
                options = question['options']
                random.shuffle(options)
                user_answer = st.radio(
                    "Select your answer:",
                    options,
                    key=f"quiz_{i}"
                )

                if st.button("Submit", key=f"submit_{i}"):
                    st.session_state.user_answers[i] = user_answer
                    if user_answer == question['correct_answer']:
                        st.success("Correct! ✅")
                    else:
                        st.error(f"Incorrect. The correct answer is: {question['correct_answer']}")

            if len(st.session_state.user_answers) == len(st.session_state.quiz):
                correct_count = sum(st.session_state.user_answers.get(i) == q['correct_answer'] for i, q in enumerate(st.session_state.quiz))
                st.metric("Quiz Score", f"{correct_count}/{len(st.session_state.quiz)}")

                with st.expander("Review Answers"):
                    for i, question in enumerate(st.session_state.quiz):
                        user_answer = st.session_state.user_answers.get(i, "Not Answered")
                        correct_answer = question['correct_answer']
                        is_correct = user_answer == correct_answer

                        st.write(f"**Q{i+1}: {question['question']}**")
                        st.write(f"Your answer: {user_answer}", help="Correct" if is_correct else "Incorrect")
                        if not is_correct:
                            st.write(f"Correct answer: {correct_answer}")
        else:
            st.info("Select a segment and generate a quiz to test your knowledge!")

    # Chat Section
    st.header("💭 Chat with AI")
    if st.session_state.current_segment:
        placeholder_text = f"Ask about the video content in segment {st.session_state.current_segment['label']}..."
    else:
        placeholder_text = "Please select a video segment first..."

    # Create columns for the chat input area
    input_col1, input_col2 = st.columns([4, 1])

    with input_col1:
        user_input = st.text_area("Your message:", key="chat_input", height=100)

    with input_col2:
        uploaded_image = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="chat_image")
        uploaded_file = st.file_uploader("Upload File", type=['pdf', 'txt', 'doc', 'docx'], key="chat_file")

    # Add a submit button
    if st.button("Send", use_container_width=True):
        if not st.session_state.current_segment:
            st.warning("Please select a video segment first!")
            st.stop()

        if not user_input and not uploaded_image and not uploaded_file:
            st.warning("Please enter a message or upload a file!")
            st.stop()

        # Prepare the message content
        message_content = {
            "text": user_input,
            "image": None,
            "file": None
        }

        # Handle image upload
        if uploaded_image:
            try:
                # Read and process the image
                image = Image.open(uploaded_image)
                # Convert to RGB if necessary
                if image.mode != "RGB":
                    image = image.convert("RGB")
                # Resize if too large (e.g., max 800px width)
                max_width = 800
                if image.width > max_width:
                    ratio = max_width / image.width
                    new_size = (max_width, int(image.height * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                # Convert to base64 for storage
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                message_content["image"] = img_str
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.stop()

        # Handle file upload
        if uploaded_file:
            try:
                # Read and store file content
                file_content = uploaded_file.read()
                if uploaded_file.type.startswith('text/'):
                    # For text files, store the content as string
                    file_content = file_content.decode('utf-8')
                else:
                    # For binary files, store as base64
                    file_content = base64.b64encode(file_content).decode()
                message_content["file"] = {
                    "name": uploaded_file.name,
                    "type": uploaded_file.type,
                    "content": file_content
                }
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.stop()

        # Add message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": message_content
        })

        with st.spinner("Analyzing content..."):
            try:
                # Prepare the context with additional content
                context = {
                    "frames": st.session_state.current_frames,
                    "transcript": st.session_state.transcript,
                    "uploaded_image": message_content["image"],
                    "uploaded_file": message_content["file"]
                }

                response_data = asyncio.run(gemini_handler.generate_response(
                    user_input,
                    context["frames"],
                    context["transcript"],
                    uploaded_image=context["uploaded_image"],
                    uploaded_file=context["uploaded_file"]
                ))
                
                if response_data:
                    response_text, raw_response = response_data
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": {"text": response_text}
                    })
                    update_token_counts(raw_response)
                    st.session_state.token_counts['last_operation'] = 'Chat Response'
                else:
                    st.warning("Could not generate response. Please try again.")
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

    # Display chat history with support for images and files
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], dict):
                # Display text content
                if "text" in message["content"] and message["content"]["text"]:
                    st.write(message["content"]["text"])

                # Display image if present
                if "image" in message["content"] and message["content"]["image"]:
                    try:
                        image_bytes = base64.b64decode(message["content"]["image"])
                        st.image(image_bytes, caption="Uploaded Image", use_container_width=True)
                    except Exception as e:
                        st.error(f"Error displaying image: {str(e)}")

                # Display file if present
                if "file" in message["content"] and message["content"]["file"]:
                    file_info = message["content"]["file"]
                    if file_info["type"].startswith('text/'):
                        with st.expander(f"📎 {file_info['name']}"):
                            st.text(file_info["content"])
                    else:
                        st.write(f"📎 Uploaded file: {file_info['name']}")
            else:
                # Handle legacy message format
                st.write(message["content"])

# Display token info in the corner
display_token_info()

# Cleanup temporary files when the app is closed
def cleanup():
    video_processor.cleanup()
