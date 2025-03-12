import os
import cv2
import yt_dlp
import numpy as np
from typing import List, Tuple
import shutil
import tempfile

class VideoProcessor:
    def __init__(self):
        self.ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': 'downloads/%(id)s.%(ext)s',
            'quiet': True,
        }
        self.temp_files = []

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

    def process_local_video(self, video_path: str, title: str = None) -> dict:
        """
        Process a local MP4 file
        Returns video info similar to YouTube format
        """
        try:
            if not os.path.exists(video_path):
                raise Exception(f"Video file not found: {video_path}")
            
            # Open the video to get properties
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            
            # Release video capture
            cap.release()
            
            # Keep track of this file for cleanup
            self.temp_files.append(video_path)
            
            # Use filename as title if not provided
            if not title:
                title = os.path.basename(video_path)
            
            return {
                'url': video_path,  # Local path serves as URL
                'title': title,
                'thumbnail': '',     # No thumbnail for local videos
                'duration': duration
            }
        except Exception as e:
            raise Exception(f"Error processing local video: {str(e)}")

    def extract_frames(self, video_url: str, chunk_start: int = 0, chunk_end: int = 300) -> List[Tuple[np.ndarray, float]]:
        """
        Extracts frames from video at 0.5 fps (every 2 seconds)
        Returns list of tuples containing frame and its timestamp
        Works with both YouTube URLs and local file paths
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
        print(f"Extracted {len(frames)} frames from video segment ({chunk_start}s to {chunk_end}s)")
        return frames

    def extract_frames_from_local(self, video_path: str, chunk_start: int = 0, chunk_end: int = 300) -> List[Tuple[np.ndarray, float]]:
        """
        A wrapper method for extract_frames when used with local files
        Uses the same implementation as extract_frames but is kept for API clarity
        """
        return self.extract_frames(video_path, chunk_start, chunk_end)

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
        Removes downloaded files and temporary files
        """
        try:
            # Remove YouTube download directory
            shutil.rmtree('downloads', ignore_errors=True)
            
            # Remove any temporary files we've created
            for temp_file in self.temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
            # Clean up any parent temp directories
            for temp_file in self.temp_files:
                temp_dir = os.path.dirname(temp_file)
                if os.path.exists(temp_dir) and temp_dir.startswith(tempfile.gettempdir()):
                    try:
                        os.rmdir(temp_dir)  # Will only remove if empty
                    except:
                        pass  # Ignore if directory not empty
                        
            # Reset list of temp files
            self.temp_files = []
        except Exception as e:
            print(f"Error cleaning up files: {str(e)}")
