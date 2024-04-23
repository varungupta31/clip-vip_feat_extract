import av
import torch
from torch.nn import functional as F
import numpy as np
from easydict import EasyDict as edict
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPProcessor, CLIPTokenizerFast
from clipvip.CLIP_VIP import CLIPModel, clip_loss
from transformers import AutoProcessor
import os
from glob import glob
import json
from natsort import natsorted
import time
from tqdm import tqdm

extraCfg = edict({
    "type": "ViP",
    "temporal_size": 12,
    "if_use_temporal_embed": 1,
    "logit_scale_init_value": 4.60,
    "add_cls_num": 3,
    "return_pooled": False,
    "patch_pooled": False, #The representation of each patch will be returned if False.
})

clipconfig = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
clipconfig.vision_additional_config = extraCfg

checkpoint = torch.load("/ssd_scratch/cvit/varun/pretrain_clipvip_base_32.pt")
cleanDict = { key.replace("clipmodel.", "") : value for key, value in checkpoint.items() }
model =  CLIPModel(config=clipconfig)
model.load_state_dict(cleanDict)

tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch16")

def extract_text_features(texts, tokenizer, model):    
    tokens = tokenizer(texts, padding=True, return_tensors="pt")
    textOutput = model.get_text_features(**tokens)
    return textOutput



def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
#     '''
#     Sample a given number of frame indices from the video.
#     Args:
#         clip_len (`int`): Total number of frames to sample.
#         frame_sample_rate (`int`): Sample every n-th frame.
#         seg_len (`int`): Maximum allowed index of sample's last frame.
#     Returns:
#         indices (`List[int]`): List of sampled frame indices
#     '''

#     end_idx = seg_len
#     start_idx = 0
#     indices = np.linspace(start_idx, end_idx, num=clip_len)
#     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
#     return indices

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len+1)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def video_loader(video_path, processor, model):

    container = av.open(video_path)
    fcount = container.streams.video[0].frames

    #This decides how many frames are to be sampled. The original paper has set it to 12 (always).
    clip_len = 12
    
    indices = sample_frame_indices(clip_len=clip_len, frame_sample_rate=fcount//clip_len, seg_len=fcount)
    video = read_video_pyav(container, indices)
    pixel_values = processor(videos=list(video), return_tensors="pt").pixel_values
    B, N, C, H, W = pixel_values.shape
    vv = pixel_values.reshape(-1, C, H, W)
    inputs = {
        "if_norm": True,
        "pixel_values": pixel_values}
    
    with torch.no_grad():
        video_features = model.get_image_features(**inputs)
    
    return video_features



output_dir = r'/ssd_scratch/cvit/varun/charades_clipvip_feats'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# msrvtt_annotations_path = r'/home2/varungupta/clip-vip_feat_extract/jupyter_stuff/msrvtt_annotations_formatted.json'
# msrvtt_annotations_path = r'/home2/varungupta/clip-vip_feat_extract/mini.json'
charades_videos_path = r'/ssd_scratch/cvit/varun/MSRVTT/videos/all'

videos_list = natsorted(glob(charades_videos_path + '/*.mp4'))[:10]
# msrvtt_annotations = json.load(open(msrvtt_annotations_path, 'r'))


#for each video, we create a npy file.
# all_videos = sorted(list(msrvtt_annotations.keys()), reverse=True)

for video in tqdm(videos_list):
    vid_name = video.split('/')[-1].split('.')[0]
    if os.path.exists(os.path.join(output_dir, video + '.npy')):
        continue
    st = time.time()
    # video_data = {}
    # contents = msrvtt_annotations[video]
    
    # video_path = os.path.join(msrvtt_videos_path, video + '.mp4')

    vid_feat = video_loader(video, processor, model)
    vid_feat = vid_feat.detach().cpu().numpy()
    print(vid_feat.shape)
        
    np.save(os.path.join(output_dir, vid_name + '.npy'), vid_feat)
