from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from typing import List, Dict, Optional
import re

class TranscriptHandler:
    @staticmethod
    def extract_video_id(url: str) -> str:
        """
        Extracts video ID from YouTube URL
        """
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and shared URLs
            r'youtu\.be\/([0-9A-Za-z_-]{11})',   # Short URLs
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        raise ValueError("Invalid YouTube URL")

    def get_available_transcripts(self, video_id: str) -> Dict[str, Dict]:
        """
        Gets available transcripts for a video
        """
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            available_transcripts = {}

            # Get manual transcripts
            for transcript in transcript_list._manually_created_transcripts.values():
                key = f"{transcript.language} (Manual)"
                available_transcripts[key] = {
                    'code': transcript.language_code,
                    'language': transcript.language,
                    'is_generated': False,
                    'is_translatable': transcript.is_translatable
                }

            # Get generated transcripts
            for transcript in transcript_list._generated_transcripts.values():
                key = f"{transcript.language} (Auto-generated)"
                available_transcripts[key] = {
                    'code': transcript.language_code,
                    'language': transcript.language,
                    'is_generated': True,
                    'is_translatable': transcript.is_translatable
                }

            # Add translation languages if any transcript is translatable
            translatable = None
            for transcript_info in available_transcripts.values():
                if transcript_info['is_translatable']:
                    try:
                        translatable = transcript_list.find_transcript([transcript_info['code']])
                        break
                    except:
                        continue

            if translatable and hasattr(translatable, 'translation_languages'):
                for lang in translatable.translation_languages:
                    key = f"{lang['language']} (Translated)"
                    if key not in available_transcripts:
                        available_transcripts[key] = {
                            'code': lang['language_code'],
                            'language': lang['language'],
                            'is_generated': True,
                            'is_translatable': False
                        }

            return available_transcripts

        except TranscriptsDisabled:
            print(f"Transcripts are disabled for video ID: {video_id}")
            return {}
        except Exception as e:
            print(f"Error getting transcripts: {str(e)}")
            return {}

    def get_transcript(self, video_id: str, languages: List[str]) -> Optional[List[Dict]]:
        """
        Gets transcript for specified languages
        Returns list of dictionaries containing text and timestamps, or None if no transcript found
        """
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            for lang_code in languages:
                try:
                    # Try to get direct transcript
                    try:
                        transcript = transcript_list.find_transcript([lang_code])
                        return transcript.fetch()
                    except NoTranscriptFound:
                        # Try to find a translatable transcript
                        for t in list(transcript_list._manually_created_transcripts.values()) + \
                                list(transcript_list._generated_transcripts.values()):
                            if t.is_translatable:
                                try:
                                    translated = t.translate(lang_code)
                                    return translated.fetch()
                                except:
                                    continue
                except:
                    continue

            return None

        except Exception as e:
            print(f"Error getting transcript: {str(e)}")
            return None

    def get_transcript_for_chunk(self, transcript: List[Dict], start: int, end: int) -> str:
        """
        Gets transcript text for specified time chunk
        """
        if not transcript:
            return ""

        chunk_transcript = []
        for entry in transcript:
            if start <= entry['start'] <= end:
                chunk_transcript.append(entry['text'])
        return " ".join(chunk_transcript)

    def get_timestamp_text(self, transcript: List[Dict], timestamp: float, window: int = 30) -> str:
        """
        Gets transcript text around specified timestamp with given window (in seconds)
        """
        if not transcript:
            return ""

        context_transcript = []
        for entry in transcript:
            if timestamp - window <= entry['start'] <= timestamp + window:
                context_transcript.append(entry['text'])
        return " ".join(context_transcript)