# 🎓 Study Buddy - Your AI Learning Companion 

> Turn any YouTube video into your personal tutor! 🚀


## ✨ What's This?

Study Buddy transforms YouTube videos into interactive study sessions! 

### 🎯 Features

- 📺 **Video Processing**: YouTube & local MP4s
- 🤖 **Gemini AI**: Advanced AI explanations
- 🎯 **Custom Segments**: Digestible video chunks
- 💭 **Interactive Chat**: Ask questions about the video
- 📝 **Flashcards**: Auto-generated study materials
- 📋 **Quizzes**: Test your knowledge
- 🌍 **Multi-Language**: Learn in your preferred language

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Google Gemini API key

### Installation

```bash
# 1️⃣ Clone the repo
git clone https://github.com/Imhari14/Yt-creator.git
cd Yt-creator

# 2️⃣ Set up a virtual environment
python -m venv venv
source venv/bin/activate  # Activate the virtual environment

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Set up environment variables
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# 5️⃣ Launch the app
streamlit run app.py
```

## 💡 How to Use

### Pick Your Video Source

```python
# YouTube URL
youtube_url = "https://www.youtube.com/watch?v=your_video_id"

# Local MP4 File
video_path = "path/to/your/video.mp4"
transcript_path = "path/to/your/transcript.txt"
```

### Learning Features

#### Generate Flashcards 📝
```python
flashcards = [
    {"question": "What is the main concept at 2:15?", "answer": "Key principle of machine learning algorithms"}
]
```

#### Take a Quiz 📋
```python
quiz = {
    "question": "What is the primary advantage of deep learning?",
    "options": ["Automatic feature extraction", "Lower computational cost"],
    "correct_answer": "Automatic feature extraction"
}
```

#### Chat with AI 💭
```python
user_query = "Explain the concept at 5:30 in simpler terms?"
ai_response = "Gradient descent is like going down a hill to find the lowest point."
```

## 🛠️ Built With

- [Streamlit](https://streamlit.io/)
- [Google Gemini AI](https://deepmind.google/technologies/gemini/)
- [Python](https://www.python.org/)

## 🤝 Contributing

1. **Issues & Features**
```bash
1. Go to "Issues"
2. Click "New Issue"
3. Use the bug report template
```

2. **Pull Requests**
```bash
1. Fork the repo
2. Create your feature branch
git checkout -b feature/AmazingFeature

3. Commit your changes
git commit -m 'Add some AmazingFeature'

4. Push to your branch
git push origin feature/AmazingFeature

5. Open a Pull Request
```

## 📝 License

```text
MIT License

© 2025 Imhari14
```

## 🙌 Shoutouts

- [Streamlit Team](https://streamlit.io/)
- [Google Gemini AI Team](https://deepmind.google/technologies/gemini/)
- [Open Source Community](https://github.com/)

---

<div align="center">
Made with ❤️ by [Imhari14](https://github.com/Imhari14)
</div>
