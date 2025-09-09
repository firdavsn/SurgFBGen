import os
import torch
from torchvision.transforms.functional import resize as resize_tensor
import cv2
import numpy as np
from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML
from transformers import pipeline
from PIL import Image
from cotracker.predictor import CoTrackerPredictor
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin_min
import h5py
from tqdm import tqdm

def detect_edges(
    image: np.ndarray, 
    min_val: int = 10, 
    max_val: int = 40, 
    bottom_padding: int = 0,
    top_padding: int = 0,
    left_padding: int = 0,
    right_padding: int = 0
) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] == 3:
        image = image[0]
    
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    processed_image = image.copy()
    h, w = processed_image.shape[:2]

    edges = cv2.Canny(processed_image, min_val, max_val)

    if bottom_padding > 0:
        padding = min(bottom_padding, h)
        edges[h-padding:h, :] = 0
    if top_padding > 0:
        padding = min(top_padding, h)
        edges[0:padding, :] = 0
    if left_padding > 0:
        padding = min(left_padding, w)
        edges[:, 0:padding] = 0
    if right_padding > 0:
        padding = min(right_padding, w)
        edges[:, w-padding:w] = 0
    
    return edges

def dilate_edges_with_circles(
    edges: np.ndarray, 
    radius: int
) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    return dilated_edges

def fill_gaps(edges_image: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(edges_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_area = np.zeros_like(edges_image)
    cv2.drawContours(filled_area, contours, -1, (255), thickness=cv2.FILLED)
    final_image = cv2.bitwise_or(filled_area, edges_image)
    return final_image

def resize_image(image_tensor, size):
    return resize_tensor(image_tensor, size, antialias=True)

def resize_video(video_tensor, size):
    resized_frames = []
    for frame in video_tensor:
        resized_frame = resize_image(frame, size)
        resized_frames.append(resized_frame)
    return torch.stack(resized_frames)

def mask_video_by_depth(video_tensor, depth_tensor, threshold=128):
    masked_video = video_tensor.clone()
    mask = depth_tensor < threshold
    masked_video[mask] = 0
    return masked_video

def solidify_depth(
    depth_tensor,
    mode: str = 'threshold', # 'threshold' or 'edges'
    via_edges_params: dict = {
        "min_val": 10,
        "max_val": 40,
        "bottom_padding": 10,
        "top_padding": 0,
        "left_padding": 0,
        "right_padding": 0,
        'dilate_radius': 5
    },
    via_threshold_params: dict = {
        'threshold': 30
    },
):
    if mode == 'edges':
        new_depth_tensor = []
        for depth in depth_tensor:
            edges = detect_edges(
                depth.cpu().numpy(), 
                min_val=via_edges_params['min_val'], 
                max_val=via_edges_params['max_val'], 
                bottom_padding=via_edges_params['bottom_padding'],
                top_padding=via_edges_params['top_padding'],
                left_padding=via_edges_params['left_padding'],
                right_padding=via_edges_params['right_padding']
            )
            new_depth = dilate_edges_with_circles(edges, via_edges_params['dilate_radius'])
            new_depth = fill_gaps(new_depth)
            new_depth_tensor.append(new_depth)
        new_depth_tensor = torch.Tensor(np.stack(new_depth_tensor))
    elif mode == 'threshold':
        new_depth_tensor = depth_tensor.clone()
        new_depth_tensor[new_depth_tensor < via_threshold_params['threshold']] = 0
        new_depth_tensor[new_depth_tensor >= via_threshold_params['threshold']] = 255
    else:
        raise ValueError("Invalid mode. Choose 'threshold' or 'edges'.")
    return new_depth_tensor


def dim_by_distance_from_gray(image, max_dim_factor=1.0):
    image_float = image.astype(np.float64)

    gray_point = np.array([128.0, 128.0, 128.0])
    distances = np.sqrt(np.sum((image_float - gray_point)**2, axis=2))

    max_dist = np.sqrt(3 * 128**2)
    normalized_distances = distances / max_dist
    
    normalized_distances = np.clip(normalized_distances, 0, 1)

    dim_amount = normalized_distances * max_dim_factor
    scaling_factors = 1.0 - dim_amount

    dimmed_image_float = image_float * scaling_factors[:, :, np.newaxis]

    return dimmed_image_float.astype(np.uint8)

def depth_video(
    video: torch.Tensor,    # [T, C, H, W]
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
) -> torch.Tensor:  # [T, C, H, W]
    frames = np.array(video.permute(0, 2, 3, 1).cpu().numpy(), dtype=np.uint8) # [T, H, W, C]
    depths = []
    for i in range(len(frames)):
        frame = frames[i]
        frame = dim_by_distance_from_gray(frame, max_dim_factor=1)
        depth = pipe(Image.fromarray(frame).convert("RGB"))["depth"]
        depths.append(depth)
    depths = torch.Tensor(np.array(depths))
    return depths

def determine_track_distances(pred_tracks):
    displacements = pred_tracks[:, 1:, :, :] - pred_tracks[:, :-1, :, :]
    distances = torch.linalg.norm(displacements, dim=-1)
    total_track_distances = distances.sum(dim=1)
    return total_track_distances

def filter_tracks_by_clusters(
    tracks: torch.Tensor,
    n_clusters: int = 5
) -> torch.Tensor:
    
    if not isinstance(tracks, torch.Tensor) or tracks.dim() != 4 or tracks.shape[0] != 1:
        raise ValueError("Input 'tracks' must be a torch.Tensor with shape [1, F, N, 2].")

    initial_dots = tracks[0, 0, :, :].cpu().numpy()

    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clustering.fit_predict(initial_dots)

    centroids = np.array([initial_dots[cluster_labels == i].mean(axis=0) for i in range(n_clusters)])

    closest_point_indices, _ = pairwise_distances_argmin_min(
        centroids, 
        initial_dots
    )

    indices_tensor = torch.tensor(
        closest_point_indices, 
        dtype=torch.long, 
        device=tracks.device
    )

    filtered_tracks = tracks[:, :, indices_tensor, :]

    return filtered_tracks


def track_instruments(
    video_path: str,
    cotracker_model: CoTrackerPredictor,
    filter_instrument: bool = True,
    filter_instrument_num_tracks: int = 30
):
    video = read_video_from_path(video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
    print(f"Loaded video from {os.path.basename(video_path)} with shape {video.shape}")
    
    depths = depth_video(video)
    masks = solidify_depth(
        depths,
        mode='edges',
        via_edges_params={
            "min_val": 10,
            "max_val": 40,
            "bottom_padding": 10,
            "top_padding": 0,
            "left_padding": 0,
            "right_padding": 0,
            'dilate_radius': 10
        },
        via_threshold_params={'threshold': 20}
    )

    if torch.cuda.is_available():
        cotracker_model = cotracker_model.cuda()
        to_pred_video = video.cuda().float()[None]

    pred_tracks, pred_visibility = cotracker_model(to_pred_video, grid_size=20)

    if filter_instrument:
        done = False
        initial_idx = 0
        while not done:
            initial_frame_mask = masks[initial_idx].cpu().numpy()
            initial_points = pred_tracks[:, initial_idx, :, :]

            x_indices = initial_points[:, :, 0].clamp(0, initial_frame_mask.shape[1] - 1).long()
            y_indices = initial_points[:, :, 1].clamp(0, initial_frame_mask.shape[0] - 1).long()

            mask_values_at_points = torch.Tensor(initial_frame_mask).cuda()[y_indices.squeeze(0), x_indices.squeeze(0)]
            tracks_to_include = (mask_values_at_points == 255).unsqueeze(0)
            mask_1d = tracks_to_include.squeeze(0)
            filtered_tracks = pred_tracks[:, :, mask_1d, :]
            num_tracks = filtered_tracks.shape[2]
            if num_tracks < 5:
                initial_idx += 1
                if initial_idx >= masks.shape[0]:
                    done = True
                    filtered_tracks = pred_tracks
                    print("Could not find a frame with enough tracks on the mask. Using all tracks.")
            else:
                done = True
        total_track_distances = determine_track_distances(filtered_tracks)

        k = int(filtered_tracks.shape[2] / 1.5)
        _, top_indices = torch.topk(total_track_distances.squeeze(), k)
        filtered_tracks = filtered_tracks[:, :, top_indices, :]
        
        n_clusters = min(filter_instrument_num_tracks, filtered_tracks.shape[2])
        final_tracks = filter_tracks_by_clusters(filtered_tracks, n_clusters=n_clusters)
        
        return final_tracks, depths, masks
    else:
        return pred_tracks, depths, masks


def main(
    clips_dir: str = '~/surgery/clips_with_wiggle/fb_clips_wiggle',
    output_h5_path: str = '~/surgery/surgical_fb_generation/SurgFBGen/outputs/instrument_tracks/instrument_tracks.h5',
    cotracker_checkpoint_path: str = '~/surgery/surgical_fb_generation/SurgFBGen/checkpoints/cotracker3.pth',
    overwrite: bool = False,
    filter_instrument: bool = True,
    filter_instrument_num_tracks: int = 30
):
    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)

    cotracker_model = CoTrackerPredictor(checkpoint=cotracker_checkpoint_path)

    if overwrite:
        h5 = h5py.File(output_h5_path, 'w')
        processed_files = []
    else:
        h5 = h5py.File(output_h5_path, 'a')
        processed_files = list(h5.keys())
    
    file2path = {
        f: os.path.join(clips_dir, f) 
        for f in os.listdir(clips_dir) if f.endswith('.mp4') or f.endswith('.avi')
    }
    files = list(file2path.keys())
    
    for file in tqdm(files):
        h5 = h5py.File(output_h5_path, 'a')
        if file in processed_files:
            print(f"Skipping {file}, already processed.")
            continue
        
        clip_path = file2path[file]
        if not os.path.isfile(clip_path):
            continue
        
        # print(f"Processing {file}...")
        tracks, depths, masks = track_instruments(
            video_path=clip_path,
            cotracker_model=cotracker_model,
            filter_instrument=filter_instrument,
            filter_instrument_num_tracks=filter_instrument_num_tracks
        )
        
        data = {
            'cvid': file,
            'clip_path': clip_path,
            'tracks': tracks.cpu().numpy(),
            'depths': depths.cpu().numpy(),
            'masks': masks.cpu().numpy()
        }
        h5.create_dataset(file, data=tracks.cpu().numpy())
        h5.close()


if __name__ == "__main__":
    main(
        output_h5_path='~/surgery/surgical_fb_generation/SurgFBGen/outputs/instrument_tracks/instrument_tracks-num_tracks=1.h5',
        filter_instrument=True,
        overwrite=True,
        filter_instrument_num_tracks=1
    )
