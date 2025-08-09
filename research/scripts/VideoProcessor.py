import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import subprocess

class VideoProcessor:
    """
    A utility class for video processing operations.
    
    This class provides methods for various video processing tasks such as
    frame similarity comparison, frame extraction, and other video manipulation
    operations.
    """
    
    @staticmethod
    def are_frames_similar(frame1: np.ndarray, frame2: np.ndarray, threshold: float = 0.9) -> bool:
        """
        Determines if two video frames are visually similar using Structural Similarity Index (SSIM).
        
        This method converts the input frames to grayscale and compares them using SSIM.
        SSIM measures the similarity between two images based on luminance, contrast, and structure.
        
        Args:
            frame1 (np.ndarray): First frame to compare (numpy array in BGR format with shape (height, width, 3))
            frame2 (np.ndarray): Second frame to compare (numpy array in BGR format with shape (height, width, 3))
            threshold (float, optional): Similarity threshold between 0 and 1.
                                       Higher values require frames to be more similar.
                                       Defaults to 0.9.
        
        Returns:
            bool: True if frames are similar (similarity >= threshold), False otherwise
            
        Raises:
            ValueError: If frames have different dimensions or if threshold is not between 0 and 1
            TypeError: If frames are not valid numpy arrays
        """
        # Input validation
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
            
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Compute SSIM between two frames
        similarity, _ = ssim(gray1, gray2, full=True)
        
        return similarity >= threshold

    # @staticmethod
    # def convert_avi_to_mp4(avi_file_path: str, output_name: str, overwrite: bool = False) -> bool:
    #     """Convert AVI video to MP4 format using ffmpeg."""
    #     if not overwrite and os.path.exists(output_name):
    #         return False
    #     elif overwrite:
    #         os.remove(output_name + ".mp4")
    #     os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = avi_file_path, output = output_name), )
    #     return True
    
    @staticmethod
    def convert_avi_to_mp4(avi_file_path: str, output_name: str, overwrite: bool = False) -> bool:
        """Convert AVI video to MP4 format using ffmpeg."""
        output_file = output_name + ".mp4"
        if not overwrite and os.path.exists(output_file):
            return False
        elif overwrite and os.path.exists(output_file):
            os.remove(output_file)
        
        command = [
            "ffmpeg",
            "-i", avi_file_path,
            "-ac", "2",
            "-b:v", "2000k",
            "-c:a", "aac",
            "-c:v", "libx264",
            "-b:a", "160k",
            "-vprofile", "high",
            "-bf", "0",
            "-strict", "experimental",
            "-f", "mp4",
            output_file
        ]
        
        try:
            subprocess.check_output(command, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print("Error converting video:")
            print(e.output.decode())
            return False
        
        return True