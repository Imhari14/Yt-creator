# Video Learning Assistant 🎓

An intelligent video learning platform built with Streamlit and Google's Gemini AI that transforms video content into interactive learning experiences.

## 🌟 Features

- **Multi-Source Video Support**
  - YouTube video integration
  - Local MP4 file upload capability
  - Transcript support (YouTube auto-captions & manual uploads)

- **Smart Video Segmentation**
  - Customizable segment intervals (1-15 minutes)
  - Frame extraction and analysis
  - Segment-based learning focus

- **AI-Powered Learning Tools**
  - Interactive flashcard generation
  - Automated quiz creation
  - Intelligent Q&A chat interface
  - Real-time video content analysis

- **Learning Progress Tracking**
  - Quiz scoring and performance metrics
  - Answer review system
  - Progress monitoring

## 🚀 Tech Stack

- Python 3.x
- Streamlit
- Google Gemini AI
- YouTube Data API
- OpenCV (for video processing)
- dotenv (for environment management)

## 📋 Prerequisites

1. Python 3.x installed
2. Google Gemini API key
3. Required Python packages:
   ```bash
   streamlit
   python-dotenv
   google-generativeai
   pytube
   opencv-python
   youtube-transcript-api
   ```

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Imhari14/Yt-creator.git
   cd Yt-creator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory and add:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## 🎯 Usage

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Choose your video source:
   - Enter a YouTube URL, or
   - Upload a local MP4 file with transcript

3. Configure learning settings:
   - Select segment interval
   - Choose video segments
   - Generate learning materials

4. Interact with learning tools:
   - Study with AI-generated flashcards
   - Take auto-generated quizzes
   - Use the AI chat for deeper understanding

## 🎮 Features In Detail

### Video Processing
- Supports both YouTube and local MP4 videos
- Automatic frame extraction
- Transcript handling for better context

### Learning Segments
- Customizable segment duration
- Focused learning on specific video parts
- Interactive segment selection

### AI Learning Tools
- **Flashcards**: AI-generated question-answer pairs
- **Quizzes**: Auto-generated multiple-choice questions
- **Chat Interface**: Context-aware AI responses

### Progress Tracking
- Quiz performance metrics
- Answer review system
- Session-based learning progress

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is currently unlicensed. Consider adding a license to protect your work.

## 👤 Author

- GitHub: [@Imhari14](https://github.com/Imhari14)

## 🙏 Acknowledgments

- Google Gemini AI for powering the intelligent features
- Streamlit for the interactive web interface
- YouTube Data API for video integration
