import os
import streamlit as st
from dotenv import load_dotenv
from utils.video_processor import VideoProcessor
from utils.transcript_handler import TranscriptHandler
from utils.gemini_handler import GeminiHandler
import asyncio
import time
import random  # <--- IMPORT RANDOM HERE


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
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .video-container {
        position: relative;
        padding-bottom: 56.25%; /* 16:9 aspect ratio */
        height: 0;
        overflow: hidden;
        max-width: 100%;
    }
    .video-container iframe,
    .video-container video {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        border-radius: 10px; /* Rounded corners for video */
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .sidebar .stButton > button {
        background-color: #2196F3;
    }
    .sidebar .stButton > button:hover {
        background-color: #1976D2;
    }
    .segment-selector {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    /* Flashcard and Quiz Expander Styling */
    .stExpander {
        border: 1px solid #ccc;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .stExpanderHeader {
        font-weight: bold;
        background-color: #f0f2f6;
        padding: 10px;
    }
    .stExpanderContent {
        padding: 10px;
    }
    /* Chat message styling */
    .user-message {
        background-color: #dcf8c6;  /* Light green for user messages */
        padding: 8px 12px;
        border-radius: 10px;
        margin-bottom: 8px;
        display: inline-block; /* Allows the bubble to fit content */
        max-width: 80%; /* Limit width */
     }

    .assistant-message {
        background-color: #f0f0f0;  /* Light gray for assistant messages */
        padding: 8px 12px;
        border-radius: 10px;
        margin-bottom: 8px;
        display: inline-block;
        max-width: 80%;
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
if 'current_frames' not in st.session_state:
    st.session_state.current_frames = None
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


def format_timestamp(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def get_segment_options(duration):
    # Create segments of 5 minutes
    segment_length = 5 * 60  # 5 minutes in seconds
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

# Sidebar for video URL input and learning tools
with st.sidebar:
    st.image("https://fonts.gstatic.com/s/i/short-term/release/materialsymbolsoutlined/school/default/48px.svg")
    st.title("Learning Tools")

    # Video Input Section
    st.header("📹 Video Input")
    video_url = st.text_input("Enter YouTube URL")

    if st.button("Load Video", use_container_width=True):
        with st.spinner("Processing video..."):
            try:
                # Get video info and streaming URL
                video_info = video_processor.download_video(video_url)
                st.session_state.video_info = video_info

                # Get available transcripts
                video_id = transcript_handler.extract_video_id(video_url)
                available_transcripts = transcript_handler.get_available_transcripts(video_id)
                st.session_state.available_transcripts = available_transcripts

                # Create segment options
                st.session_state.segments = get_segment_options(video_info['duration'])

                st.success("Video loaded successfully!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.video_info = None

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
                frames = video_processor.extract_frames(
                    st.session_state.video_info['url'],
                    selected_segment["start"],
                    selected_segment["end"]
                )
                st.session_state.current_frames = frames
                st.session_state.current_segment = selected_segment
                st.success(f"Loaded segment {selected_segment['label']}")

        # Transcript Selection
        if hasattr(st.session_state, 'available_transcripts') and st.session_state.available_transcripts:
            st.header("📝 Transcript")
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

        # Learning Tools
        if st.session_state.current_frames:
            st.header("📚 Learning Tools")

            if st.button("Generate Flashcards", use_container_width=True):
                with st.spinner("Generating flashcards..."):
                    try:
                        flashcards = asyncio.run(gemini_handler.generate_flashcards(
                            st.session_state.transcript,
                            st.session_state.current_frames
                        ))
                        if flashcards:
                            st.session_state.flashcards = flashcards
                            st.success("Flashcards generated!")
                        else:
                            st.warning("Could not generate flashcards. Please try again.")
                    except Exception as e:
                        st.error(f"Error generating flashcards: {str(e)}")

            if st.button("Generate Quiz", use_container_width=True):
                with st.spinner("Generating quiz..."):
                    try:
                        quiz = asyncio.run(gemini_handler.generate_quiz(
                            st.session_state.transcript,
                            st.session_state.current_frames
                        ))
                        if quiz:
                            st.session_state.quiz = quiz
                            st.session_state.quiz_score = 0
                            st.session_state.user_answers = {}  # Reset user answers
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

    st.video(st.session_state.video_info["url"])

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
                response = asyncio.run(gemini_handler.generate_response(
                    user_input,
                    st.session_state.current_frames,
                    st.session_state.transcript
                ))
                if response:
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                else:
                    st.warning("Could not generate response. Please try again.")
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            #st.markdown(f"<div class='{message['role']}-message'>{message['content']}</div>", unsafe_allow_html=True) #Another option

else:
    st.info("👈 Enter a YouTube URL in the sidebar to get started!")

# Cleanup temporary files when the app is closed
def cleanup():
    video_processor.cleanup()