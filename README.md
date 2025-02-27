# YouTube Video Learning Assistant 🎓

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)
![Gemini AI](https://img.shields.io/badge/Gemini-1.5--flash-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Last Updated](https://img.shields.io/badge/Last%20Updated-2025--02--27-brightgreen.svg)

An intelligent learning platform that transforms YouTube videos into interactive study materials using Google's Gemini AI.

## 🌟 Features

- **🎬 YouTube Video Integration**: Load any YouTube video with a simple URL
- **📚 Segmented Learning**: Study videos in manageable 5-minute segments
- **📝 Transcript Support**: Access video transcripts in multiple languages
- **🧠 AI-Generated Study Materials**:
  - **Flashcards**: Generate question-answer pairs from video content
  - **Interactive Quizzes**: Test your knowledge with multiple-choice questions
  - **AI Chat Assistant**: Ask questions about specific video segments
- **🖼️ Visual Analysis**: AI processes both visual frames and audio transcript
- **📊 Progress Tracking**: Score yourself on quizzes and review your answers

## 📋 System Requirements

- Python 3.x
- Google Gemini API Key

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Imhari14/Yt-video.git
   cd Yt-video
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your Gemini API key:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## 🚀 Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

### Step-by-Step Guide:

1. **Enter YouTube URL** in the sidebar and click "Load Video"
2. **Select a Video Segment** (5-minute chunks) and click "Load Segment"
3. **Load Transcript** in your preferred language
4. Use the **Learning Tools**:
   - Generate Flashcards for studying
   - Generate Quiz to test yourself
   - Chat with AI about the video content

## 🧩 Project Structure

```
Yt-video/
├── app.py                 # Main Streamlit application
├── .env                   # Environment variables (API keys)
├── requirements.txt       # Project dependencies
├── utils/
│   ├── gemini_handler.py  # Google Gemini AI integration
│   ├── transcript_handler.py  # YouTube transcript processing
│   └── video_processor.py # Video downloading and processing
└── downloads/             # Temporary storage for video files
```

## 🛠️ Core Components

### Video Processor
- Downloads YouTube videos using yt-dlp
- Extracts video frames for AI analysis
- Creates manageable time segments

### Transcript Handler
- Extracts and processes YouTube video transcripts
- Supports multiple languages and translations
- Segments transcripts to match video chunks

### Gemini Handler
- Integrates with Google's Gemini 1.5 Flash model
- Processes visual frames along with transcript text
- Generates flashcards, quizzes, and chat responses

## 📚 Dependencies

- `streamlit`: Web application framework
- `google-generativeai`: Google Gemini AI integration
- `yt-dlp`: YouTube video downloading
- `youtube_transcript_api`: Transcript extraction
- `opencv-python`: Video frame processing
- `pillow`: Image handling for AI model
- `python-dotenv`: Environment variable management

## 🔍 How It Works

1. **Video Processing**:
   - The app downloads video information and streaming URL
   - Video is segmented into 5-minute chunks
   - Key frames are extracted for AI analysis

2. **Transcript Analysis**:
   - Available language options are detected
   - Transcript is fetched and segmented to match video chunks

3. **AI-Powered Learning**:
   - Gemini 1.5 Flash AI model analyzes both visual content and transcript
   - Generates contextual flashcards based on key concepts
   - Creates multiple-choice quizzes with automated scoring
   - Provides conversational responses about video content

## 👨‍💻 Author

- [@Imhari14](https://github.com/Imhari14)

## 📄 License

[MIT License](LICENSE)

---

**Last Updated:** February 27, 2025
