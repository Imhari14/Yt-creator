import os
import cv2
import yt_dlp
import numpy as np
from typing import List, Tuple
import shutil

class VideoProcessor:
    def __init__(self):
        self.ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': 'downloads/%(id)s.%(ext)s',
            'quiet': True,
        }

    def download_video(self, url: str) -> dict:
        """
        Gets video info and direct streaming URL from YouTube
        """
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # Extract video info without downloading
                info = ydl.extract_info(url, download=False)
                
                # Get the best format with both video and audio
                formats = info.get('formats', [])
                best_format = None
                for f in formats:
                    if f.get('acodec') != 'none' and f.get('vcodec') != 'none':
                        if not best_format or f.get('tbr', 0) > best_format.get('tbr', 0):
                            best_format = f

                if not best_format:
                    raise Exception("No suitable video format found")

                return {
                    'url': best_format['url'],
                    'title': info.get('title', ''),
                    'thumbnail': info.get('thumbnail', ''),
                    'duration': info.get('duration', 0)
                }
        except Exception as e:
            raise Exception(f"Error processing video: {str(e)}")

    def extract_frames(self, video_url: str, chunk_start: int = 0, chunk_end: int = 300) -> List[Tuple[np.ndarray, float]]:
        """
        Extracts frames from video at 0.5 fps (every 2 seconds) from chunk_start to chunk_end seconds
        Returns list of tuples containing frame and its timestamp
        """
        frames = []
        cap = cv2.VideoCapture(video_url)
        
        # Set video position to chunk start
        cap.set(cv2.CAP_PROP_POS_MSEC, chunk_start * 1000)
        
        # Extract frames at 2-second intervals (0.5 fps)
        current_time = chunk_start
        while current_time <= chunk_end:
            # Set position to current time
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
            
            ret, frame = cap.read()
            if not ret:
                break
                
            # Get actual timestamp in seconds
            actual_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            frames.append((frame, actual_timestamp))
            
            # Move to next position (2 seconds later)
            current_time += 2
        
        # Release video capture
        cap.release()
        return frames

    def get_chunks(self, duration: int, chunk_size: int = 300) -> List[Tuple[int, int]]:
        """
        Divides video duration into chunks of specified size (in seconds)
        Returns list of tuples containing start and end times of chunks
        """
        chunks = []
        start = 0
        while start < duration:
            end = min(start + chunk_size, duration)
            chunks.append((start, end))
            start = end
        return chunks

    def format_duration(self, seconds: int) -> str:
        """
        Formats duration in seconds to HH:MM:SS
        """
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def cleanup(self):
        """
        Removes downloaded files
        """
        try:
            shutil.rmtree('downloads', ignore_errors=True)
        except Exception as e:
            print(f"Error cleaning up files: {str(e)}")
