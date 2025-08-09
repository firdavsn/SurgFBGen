# video_captioner.py

from abc import ABC, abstractmethod
from openai import OpenAI
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import base64
from src.VideoProcessor import VideoProcessor
class VideoCaption:
    """
    Class to store video captions and configuration parameters.
    
    Attributes:
        frame_captions (list[str]): List of captions for individual frames
        combined_caption (str): Overall caption combining all frame captions
        frame_interval (int): Number of frames to skip between extractions
        fps (int): Frames per second of the video
        model_config (dict): Model-specific configuration parameters
    """
    
    def __init__(self, fps: int = 30, frame_interval: int = 30):
        """
        Initialize caption object with configuration
        
        Args:
            fps (int): Frames per second of the video (default 30)
            frame_interval (int): Number of frames to skip between extractions (default 30,
                                meaning 1 frame captured per second at 30fps)
        """
        self.frame_captions = []
        self.combined_caption = ""
        self.fps = fps
        self.frame_interval = frame_interval
        self.model_config = {}    # Empty dict for model-specific params
        
    def add_frame_caption(self, caption: str):
        """
        Add a caption for an individual frame
        
        Args:
            caption (str): Caption text for a single frame
        """
        self.frame_captions.append(caption)
        
    def set_combined_caption(self, caption: str):
        """
        Set the combined caption for all frames
        
        Args:
            caption (str): Combined caption text
        """
        self.combined_caption = caption
        
    def get_frame_captions(self) -> list[str]:
        """
        Get list of individual frame captions
        
        Returns:
            list[str]: List of frame captions
        """
        return self.frame_captions
        
    def get_combined_caption(self) -> str:
        """
        Get the combined caption
        
        Returns:
            str: Combined caption text
        """
        return self.combined_caption




class VideoCaptionerConfig:
    def __init__(self, openai_api_key: str, min_frame_similarity: float = 0.9, max_tokens_per_frame_generation: int = 100, target_fps: float = None):
        """
        Configuration for VideoCaptioner.
        
        Args:
            openai_api_key (str): API key for OpenAI services.
            min_frame_similarity (float): Threshold for frame similarity comparison.
        """
        self.openai_api_key = openai_api_key
        self.min_frame_similarity = min_frame_similarity
        self.max_tokens_per_frame_generation = max_tokens_per_frame_generation
        self.target_fps = target_fps

class VideoCaptioner(ABC):
    """
    Base class for video captioning.

    This class provides a common interface for captioning videos using various approaches.
    Specific implementations (e.g., using GPT-4o, LLaVa, or other methods) should inherit
    from this class and implement the abstract methods.

    Attributes:
        video_source (str): The path or identifier of the video to be captioned.
    """

    def __init__(self, config: VideoCaptionerConfig):
        """
        Initializes the VideoCaptioner with a video source.

        Args:
            video_source (str): Path or identifier for the video.
        """
        self.config = config

    @abstractmethod
    def caption_video(self, video_source: str) -> str:
        """
        Generates a caption for the video.

        Subclasses should implement this method to process the video,
        extract frames (or use an alternative strategy), generate captions for the frames,
        and then combine them into a final video caption.

        Returns:
            str: The caption or description for the video.
        """
        pass

    @abstractmethod
    def extract_frames(self, video_source: str) -> list:
        """
        Extracts frames from the video.

        Subclasses should implement this method to extract key frames from the video
        that will be used for generating captions.

        Returns:
            list: A list of frames (could be image objects, file paths, etc.).
        """
        pass

    def combine_frame_captions(self, frame_captions: list) -> str:
        """
        Combines individual frame captions into a single caption for the video.

        This default implementation simply joins the captions with spaces, but subclasses
        can override this method if a more sophisticated combination strategy is needed.

        Args:
            frame_captions (list): A list of captions generated for individual frames.

        Returns:
            str: The combined video caption.
        """
        return " ".join(frame_captions)


class GPT4oCaptioner(VideoCaptioner):
    def __init__(self, config: VideoCaptionerConfig):
        """
        Initializes the GPT4oCaptioner with the given configuration.
        
        Args:
            config (VideoCaptionerConfig): Configuration object containing necessary parameters.
        """
        super().__init__(config)
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.model = "gpt-4o"
        self.min_frame_similarity = config.min_frame_similarity
        self.max_tokens_per_frame_generation = config.max_tokens_per_frame_generation
        self.target_fps = config.target_fps
        
    def caption_video(self, video_source: str, context: dict = None) -> "VideoCaption":
        """
        Processes the video by extracting frames, generating individual frame captions,
        and then combines those into one final overall caption by prompting the model again.
        The provided context dictionary can include both system and user instructions.
        
        Args:
            video_source (str): Path or identifier for the video.
            context (dict, optional): A dictionary with keys "system" and "user" to provide additional
                                      context for the model prompts.
            
        Returns:
            VideoCaption: An object containing individual frame captions and the combined caption.
        """
        # Extract frames from video
        frames = self.extract_frames(video_source)
        
        # Create a VideoCaption object to store frame captions and the combined caption
        video_caption_obj = VideoCaption(fps=30, frame_interval=30)
        
        # Generate captions for each frame and store them
        for frame in frames:
            caption = self.generate_frame_caption(frame, context)
            video_caption_obj.add_frame_caption(caption)
            
        # Generate a final combined caption using the list of frame captions
        combined_caption = self.generate_combined_caption(video_caption_obj.get_frame_captions(), context)
        video_caption_obj.set_combined_caption(combined_caption)
        
        return video_caption_obj
    
    def generate_frame_caption(self, frame: np.ndarray, context: dict = None) -> str:
        """
        Generates a caption for a single frame using GPT-4o.
        Context can be added to the prompt via a dictionary containing keys "system" and "user".
        
        Args:
            frame (np.ndarray): The frame to caption.
            context (dict, optional): A dictionary with keys "system" and "user" to provide additional
                                      context for the caption prompt.
            
        Returns:
            str: The caption for the frame.
        """
        # Convert frame to base64 string for API
        _, img_encoded = cv2.imencode('.jpg', frame)
        base64_image = base64.b64encode(img_encoded).decode('utf-8')
        
        messages = []
        if context is not None and "system" in context:
            messages.append({"role": "system", "content": context["system"]})
            
        # Build the user prompt
        user_prompt = "Describe what is happening in this image in a concise sentence."
        if context is not None and "user" in context:
            user_prompt = f"{context['user']} {user_prompt}"
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        })
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens_per_frame_generation
            )
            caption = response.choices[0].message.content.strip()
            return caption
        except Exception as e:
            print(f"Error generating caption: {str(e)}")
            return "Error generating caption"
    
    def generate_combined_caption(self, frame_captions: list[str], context: dict = None) -> str:
        """
        Generates a final, overall caption for the video by prompting the model 
        using the list of individual frame captions and additional context if provided.
        
        Args:
            frame_captions (list[str]): List of captions generated for individual frames.
            context (dict, optional): A dictionary with keys "system" and "user" to provide additional
                                      context for the combination prompt.
            
        Returns:
            str: The final combined caption for the video.
        """
        prompt = "Using the following frame captions, generate a concise and coherent overall caption for the video:\n" + "\n".join(frame_captions)
        
        messages = []
        if context is not None and "system" in context:
            messages.append({"role": "system", "content": context["system"]})
        
        user_prompt = prompt
        if context is not None and "user" in context:
            user_prompt = f"{context['user']} {prompt}"
            
        messages.append({"role": "user", "content": user_prompt})
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens_per_frame_generation
            )
            combined_caption = response.choices[0].message.content.strip()
            return combined_caption
        except Exception as e:
            print("Error generating combined caption:", str(e))
            return "Error generating combined caption"
    
    def extract_frames(self, video_source: str) -> list:
        """
        Extracts frames from the video file.

        Args:
            video_source (str): Path to the video file.
            target_fps (float, optional): Target frames per second to extract. 
                                          If None, extracts all frames. Defaults to None.

        Returns:
            list: List of video frames as numpy arrays.
        """
        video = cv2.VideoCapture(video_source)
        
        if not video.isOpened():
            raise ValueError(f"Error: Could not open video file {video_source}")
            
        # Get video's original FPS
        original_fps = video.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame interval if target_fps is specified
        frame_interval = 1
        if self.target_fps is not None and self.target_fps < original_fps:
            frame_interval = int(round(original_fps / self.target_fps))
        
        stored_frames = []
        frame_count = 0
        while True:
            success, frame = video.read()
            if not success:
                break
                
            if frame_count % frame_interval == 0:
                if not stored_frames:
                    stored_frames.append(frame)
                else:
                    last_stored_frame = stored_frames[-1]
                    if not VideoProcessor.are_frames_similar(last_stored_frame, frame, self.min_frame_similarity):
                        stored_frames.append(frame)
            frame_count += 1
                
        video.release()
        return stored_frames

# Example usage (for testing purposes):
if __name__ == "__main__":
    
    # Example usage (for testing purposes):
    config = VideoCaptionerConfig(
        openai_api_key="your_openai_api_key_here",
        min_frame_similarity=0.85  # Set your desired threshold here
    )
    captioner = GPT4oCaptioner(config)
    video_path = "path/to/your/video.mp4"
    caption_output = captioner.caption_video(video_path)
    print(caption_output)
