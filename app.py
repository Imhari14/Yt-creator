import os
import streamlit as st
from dotenv import load_dotenv
from utils.video_processor import VideoProcessor
from utils.transcript_handler import TranscriptHandler
from utils.gemini_handler import GeminiHandler
import asyncio
import time
import random
import tempfile

# Configure Streamlit page
st.set_page_config(
    page_title="Video Learning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Simple token info container styling - positioned at top */
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


# Load environment variables
load_dotenv()

# Initialize handlers
video_processor = VideoProcessor()
transcript_handler = TranscriptHandler()
gemini_handler = GeminiHandler()

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
if 'user_answers' not in st.session_state: # User answers
    st.session_state.user_answers = {}
if 'video_source' not in st.session_state:
    st.session_state.video_source = None  # "youtube" or "local"
if 'local_video_path' not in st.session_state:
    st.session_state.local_video_path = None
# Initialize token count tracking
if 'token_counts' not in st.session_state:
    st.session_state.token_counts = {
        'prompt_token_count': 0,
        'candidates_token_count': 0,
        'total_token_count': 0,
        'last_operation': '',
        'context_window': {
            'input_limit': 1000000,  # Default for Gemini 1.5 Flash
            'output_limit': 8000,    # Default for Gemini 1.5 Flash
        }
    }
# Initialize segment interval
if 'segment_interval' not in st.session_state:
    st.session_state.segment_interval = 5  # Default 5 minutes (in minutes)
if 'segments' not in st.session_state:
    st.session_state.segments = []


def format_timestamp(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def get_segment_options(duration, interval_minutes):
    # Create segments based on user-selected interval
    segment_length = interval_minutes * 60  # Convert minutes to seconds
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
    """Update token count information from the Gemini API response - with cumulative tracking"""
    if hasattr(response, 'usage_metadata'):
        metadata = response.usage_metadata
        # Add current usage to cumulative totals
        st.session_state.token_counts['prompt_token_count'] += metadata.prompt_token_count
        st.session_state.token_counts['candidates_token_count'] += metadata.candidates_token_count
        st.session_state.token_counts['total_token_count'] += metadata.total_token_count
        
        # Store current operation metrics
        st.session_state.token_counts['current_operation'] = {
            'name': st.session_state.token_counts['last_operation'],
            'prompt_tokens': metadata.prompt_token_count,
            'output_tokens': metadata.candidates_token_count,
            'total_tokens': metadata.total_token_count
        }
    
    # Get model info to update context window limits
    model_info = gemini_handler.get_model_info()
    if model_info:
        st.session_state.token_counts['context_window']['input_limit'] = model_info.input_token_limit
        st.session_state.token_counts['context_window']['output_limit'] = model_info.output_token_limit

def parse_transcript_file(file_content):
    """Parse transcript text file and convert to a format similar to YouTube transcripts"""
    try:
        lines = file_content.strip().split("\n")
        parsed_transcript = []
        
        # Simple parsing of text file - assuming each line has content
        current_time = 0  # Start at 0 seconds
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Create a transcript entry with estimated timing
            # Assuming each line takes about 5 seconds (adjustable)
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
    """Extract transcript text for a specific segment from the full transcript text"""
    if not transcript_text:
        return ""
    
    # Split transcript into lines for processing
    lines = transcript_text.strip().split('\n')
    segment_text = []
    
    # Since we don't have timing information in plain text files,
    # we'll assume even distribution across the video duration
    if hasattr(st.session_state, 'video_info') and st.session_state.video_info:
        total_duration = st.session_state.video_info['duration']
        # Calculate time per line
        if total_duration > 0 and len(lines) > 0:
            time_per_line = total_duration / len(lines)
            
            # Calculate line indices for the segment
            start_idx = int(start / time_per_line)
            end_idx = int(end / time_per_line) + 1
            
            # Get lines for the segment
            segment_lines = lines[start_idx:end_idx]
            segment_text = " ".join(segment_lines)
            return segment_text
    
    # If we can't calculate properly, return a portion of the transcript
    if len(lines) > 20:
        # Return a subset of the transcript if it's very large
        segment_percentage = (end - start) / st.session_state.video_info['duration']
        num_lines = max(10, int(len(lines) * segment_percentage))
        return " ".join(lines[:num_lines])
    else:
        # For short transcripts, return everything
        return transcript_text

# Update the token info display function to handle the case where 'current_operation' doesn't exist yet
def display_token_info():
    """Display token count information with a simple, clean design at the top"""
    token_info = st.session_state.token_counts
    
    token_info_html = f"""
    <div class="token-info-container">
        <div class="token-simple-title">Token Usage</div>
        <span class="token-simple-metric">Input: <span class="token-simple-value">{token_info['prompt_token_count']:,}</span></span>
        <span class="token-simple-metric">Output: <span class="token-simple-value">{token_info['candidates_token_count']:,}</span></span>
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
    
    # Video source selection - YouTube or Local MP4
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
                    # Get video info and streaming URL
                    video_info = video_processor.download_video(video_url)
                    st.session_state.video_info = video_info

                    # Get available transcripts
                    video_id = transcript_handler.extract_video_id(video_url)
                    available_transcripts = transcript_handler.get_available_transcripts(video_id)
                    st.session_state.available_transcripts = available_transcripts

                    # Create segment options based on user-defined interval
                    st.session_state.segments = get_segment_options(
                        video_info['duration'],
                        st.session_state.segment_interval
                    )
                    
                    # Clear any previous local video path
                    st.session_state.local_video_path = None
                    st.session_state.transcript_text = None

                    st.success("YouTube video loaded successfully!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.video_info = None
        
        # Transcript Selection for YouTube
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

                    # Filter transcript for current segment if one is selected
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
        
        # Local video file upload
        uploaded_video = st.file_uploader("Upload MP4 video file", type=["mp4"])
        
        # Transcript file upload
        uploaded_transcript = st.file_uploader("Upload transcript text file", type=["txt"])
        
        if uploaded_video is not None and uploaded_transcript is not None:
            if st.button("Load Local Video", use_container_width=True):
                with st.spinner("Processing local video..."):
                    try:
                        # Save uploaded video to temporary file
                        temp_dir = tempfile.mkdtemp()
                        temp_video_path = os.path.join(temp_dir, "uploaded_video.mp4")
                        
                        with open(temp_video_path, "wb") as f:
                            f.write(uploaded_video.getbuffer())
                        
                        # Process the video
                        video_info = video_processor.process_local_video(temp_video_path, uploaded_video.name)
                        st.session_state.video_info = video_info
                        st.session_state.local_video_path = temp_video_path
                        
                        # Process transcript
                        transcript_text = uploaded_transcript.getvalue().decode("utf-8")
                        st.session_state.transcript_text = transcript_text
                        
                        # Parse transcript to a format similar to YouTube transcripts
                        parsed_transcript = parse_transcript_file(transcript_text)
                        st.session_state.transcript = parsed_transcript
                        
                        # Create segment options based on user-defined interval
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

    # Segment Interval Selection
    st.subheader("⏰ Segment Settings")
    segment_interval = st.slider(
        "Segment Interval (minutes):",
        min_value=1,
        max_value=15,
        value=st.session_state.segment_interval,
        step=1
    )
        # Update session state if interval changed
    if segment_interval != st.session_state.segment_interval:
        st.session_state.segment_interval = segment_interval
        # Recalculate segments if video is already loaded
        if st.session_state.video_info:
            st.session_state.segments = get_segment_options(
                st.session_state.video_info['duration'],
                st.session_state.segment_interval
            )

    # Segment Selection
    if st.session_state.video_info:
        st.header("🎯 Video Segment")

        # Segment selector
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
                # Extract frames for selected segment
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
                
                # Update transcript for this segment
                if st.session_state.video_source == "youtube" and st.session_state.transcript:
                    # For YouTube, we already have the transcript with time information
                    st.session_state.transcript = transcript_handler.get_transcript_for_chunk(
                        st.session_state.transcript,
                        selected_segment["start"],
                        selected_segment["end"]
                    )
                elif st.session_state.video_source == "local" and st.session_state.transcript_text:
                    # For local video, extract portion of transcript based on timing estimation
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
                            # Update token counts
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
                            # Update token counts
                            update_token_counts(response)
                            st.session_state.token_counts['last_operation'] = 'Generate Quiz'
                            st.success("Quiz generated!")
                        else:
                            st.warning("Could not generate quiz. Please try again.")
                    except Exception as e:
                        st.error(f"Error generating quiz: {str(e)}")

# Main content area
if st.session_state.video_info:
    # Video Player Section
    st.header("📺 Video Player")
    video_title = st.session_state.video_info['title']
    st.markdown(f"### {video_title}")

    # Show current segment info if selected
    if st.session_state.current_segment:
        segment = st.session_state.current_segment
        st.markdown(f"**Current Segment:** {segment['label']}")

    # Display video based on source
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
                    st.write("**Question:**", card['question'])  # Display the full question
                    st.write("**Answer:**", card['answer'])
        else:
            st.info("Select a segment and generate flashcards to study!")

    # Quiz Column
    with col2:
        st.header("📋 Quiz")
        if st.session_state.quiz:
            for i, question in enumerate(st.session_state.quiz):
                st.subheader(f"Q{i+1}: {question['question']}")
                # Randomize option order
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
            #Show results
            if len(st.session_state.user_answers) == len(st.session_state.quiz):
                correct_count = sum(st.session_state.user_answers.get(i) == q['correct_answer'] for i, q in enumerate(st.session_state.quiz))
                st.metric("Quiz Score", f"{correct_count}/{len(st.session_state.quiz)}")

                # Review Section
                with st.expander("Review Answers"):
                    for i, question in enumerate(st.session_state.quiz):
                        user_answer = st.session_state.user_answers.get(i, "Not Answered")
                        correct_answer = question['correct_answer']
                        is_correct = user_answer == correct_answer

                        st.write(f"**Q{i+1}: {question['question']}**")
                        st.write(f"Your answer: {user_answer}", help= "Correct" if is_correct else "Incorrect")
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

    user_input = st.chat_input(placeholder_text)

    if user_input:
        if not st.session_state.current_segment:
            st.warning("Please select a video segment first!")
            st.stop()

        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("Analyzing video content..."):
            try:
                response_data = asyncio.run(gemini_handler.generate_response(
                    user_input,
                    st.session_state.current_frames,
                    st.session_state.transcript
                ))
                
                if response_data:
                    response_text, raw_response = response_data
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                    # Update token counts
                    update_token_counts(raw_response)
                    st.session_state.token_counts['last_operation'] = 'Chat Response'
                else:
                    st.warning("Could not generate response. Please try again.")
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            #st.markdown(f"<div class='{message['role']}-message'>{message['content']}</div>", unsafe_allow_html=True) #Another option

else:
    st.info("👈 Enter a YouTube URL or upload a local MP4 file in the sidebar to get started!")

# Display token info in the corner
display_token_info()

# Cleanup temporary files when the app is closed
def cleanup():
    video_processor.cleanup()