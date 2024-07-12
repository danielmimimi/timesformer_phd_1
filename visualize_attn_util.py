#################################################
### Functions used to compute and visualize the learnt Space-Time attention in TimeSformer ###
#################################################

from pathlib import Path
import random
from timesformer.models.vit import *
from timesformer.datasets import utils as utils
from timesformer.config.defaults import get_cfg
from einops import rearrange, repeat, reduce
import cv2
# from google.colab.patches import cv2_imshow
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import matplotlib.pyplot as plt

DEFAULT_MEAN = [0.45, 0.45, 0.45]
DEFAULT_STD = [0.225, 0.225, 0.225]
# convert video path to input tensor for model
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(DEFAULT_MEAN,DEFAULT_STD),
    transforms.Resize(224),
    transforms.CenterCrop(224),
])

# convert the video path to input for cv2_imshow()
transform_plot = transforms.Compose([
    lambda p: cv2.imread(str(p),cv2.IMREAD_COLOR),
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    lambda x: rearrange(x*255, 'c h w -> h w c').numpy()
])


def get_frames(path_to_video, num_frames=8,frame_distance=1,label=None):
  "return a list of paths to the frames of sampled from the video"
  path_to_frames = [f for f in path_to_video if Path(f).suffix == '.png']
  path_to_frames.sort(key=lambda f: int(f.with_suffix('').name[-5:]))
  assert num_frames <= len(path_to_frames), "num_frames can't exceed the number of frames extracted from videos"
  if len(path_to_frames) == num_frames:
    return(path_to_frames)
  else:
    video_length = len(path_to_frames)
    # seg_size = float(video_length - 1) / num_frames
    seq = []
    for i in range(num_frames):
      start_point = random.randint(0, video_length - num_frames*frame_distance)
      seq = [start_point + i*frame_distance for i in range(num_frames)]
    return ([path_to_frames[p] for p in seq])

def create_video_input(path_to_video,frame_distance,label):
  "create the input tensor for TimeSformer model"
  path_to_frames = get_frames(path_to_video,frame_distance=frame_distance,label=label)
  frames = [transform(cv2.imread(str(p), cv2.IMREAD_COLOR)) for p in path_to_frames]
  frames = torch.stack(frames, dim=0)
  frames = rearrange(frames, 't c h w -> c t h w')
  frames = frames.unsqueeze(dim=0)
  return(frames)

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def create_masks(masks_in, np_imgs):
  masks = []
  for mask, img in zip(masks_in, np_imgs):
    mask= cv2.resize(mask, (img.shape[1], img.shape[0]))
    mask = show_mask_on_image(img, mask)
    masks.append(mask)
  return(masks)


def drop_low_attention(attention,discard_ratio=0.9):
    flat = attention.view(attention.size(0), -1)
    _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
    indices = indices[indices != 0]
    flat[0, indices] = 0
    return flat


def fuse_attention_heads(attention,head_fusion,head_dim=1):
    if head_fusion == "mean":
        attention_heads_fused = attention.mean(axis=1)
    elif head_fusion == "max":
        attention_heads_fused = attention.max(axis=1)[0]
    elif head_fusion == "min":
        attention_heads_fused = attention.min(axis=1)[0]
    else:
        raise "Attention head fusion type Not supported"
    return attention_heads_fused


def combine_divided_attention(attn_t, attn_s,head_fuse_method='mean'):
  """Returns ts,s,t"""
  ## time attention
  # average time attention weights across heads
  #attn_t = attn_t.mean(dim = 1)
  attn_t = fuse_attention_heads(attn_t,head_fuse_method)
  # add cls_token to attn_t as an identity matrix since it only attends to itself 
  I = torch.eye(attn_t.size(-1)).unsqueeze(0)
  attn_t = torch.cat([I,attn_t], 0)
  # adding identity matrix to account for skipped connection 
  attn_t = attn_t +  torch.eye(attn_t.size(-1))[None,...]
  # renormalize
  attn_t = attn_t / attn_t.sum(-1)[...,None]

  ## space attention
  # average across heads
  #attn_s = attn_s.mean(dim = 1)
  attn_s = fuse_attention_heads(attn_s,head_fuse_method)
  # adding residual and renormalize 
  attn_s = attn_s +  torch.eye(attn_s.size(-1))[None,...]
  attn_s = attn_s / attn_s.sum(-1)[...,None]
  ## combine the space and time attention
  attn_ts = einsum('tpk, ktq -> ptkq', attn_s, attn_t)
  ## average the cls_token attention across the frames
  # splice out the attention for cls_token
  attn_cls = attn_ts[0,:,:,:]
  # average the cls_token attention and repeat across the frames
  attn_cls_a = attn_cls.mean(dim=0)
  attn_cls_a = repeat(attn_cls_a, 'p t -> j p t', j = 8)
  # add it back
  attn_ts = torch.cat([attn_cls_a.unsqueeze(0),attn_ts[1:,:,:,:]],0)
  return(attn_ts,attn_s.mean(dim=-1).transpose(1,0),attn_t.mean(dim=-1))

class DividedAttentionRollout():
  def __init__(self, model,device, **kwargs):
    self.model = model
    self.device = device
    self.hooks = []

  def get_attn_t(self, module, input, output):
    self.time_attentions.append(output.detach().cpu())
  def get_attn_s(self, module, input, output):
    self.space_attentions.append(output.detach().cpu())

  def remove_hooks(self): 
    for h in self.hooks: h.remove()
    
  def __call__(self, input_tensor):
    if len(input_tensor.shape) == 4:
      input_tensor = input_tensor.unsqueeze(dim=0)
    # input_tensor = create_video_input(path_to_video,frame_distance,label)
    
    self.model.zero_grad()
    self.time_attentions = []
    self.space_attentions = []
    self.attentions = []
    self.attentions_min = []
    self.attentions_max = []
    for name, m in self.model.named_modules():
      if 'temporal_attn.attn_drop' in name:
        self.hooks.append(m.register_forward_hook(self.get_attn_t))
      elif 'attn.attn_drop' in name:
        self.hooks.append(m.register_forward_hook(self.get_attn_s))
    preds = self.model(input_tensor.to(self.device))
    preds = preds.detach().cpu()
    for h in self.hooks: h.remove()
    for attn_t,attn_s in zip(self.time_attentions, self.space_attentions):
      self.attentions.append(combine_divided_attention(attn_t,attn_s))
    for attn_t,attn_s in zip(self.time_attentions, self.space_attentions):
      self.attentions_min.append(combine_divided_attention(attn_t,attn_s,head_fuse_method='min'))
    for attn_t,attn_s in zip(self.time_attentions, self.space_attentions):
      self.attentions_max.append(combine_divided_attention(attn_t,attn_s,head_fuse_method='max'))

    def generate_mask(attention,index=0):
        p,t = attention[0][index].shape[0], attention[0][index].shape[1]
        result = torch.eye(p*t)
        for att in attention:
          if index == 0:
              att = rearrange(att[index], 'p1 t1 p2 t2 -> (p1 t1) (p2 t2)')  # Shape [1576, 1576]
          else:
              att = att[index].flatten().unsqueeze(1)  # Shape [1576, 1]
              att = att.expand(-1, result.shape[1])  # Shape [1576, 1576]


          result = torch.matmul(att, result)
        if index == 0:
            mask = rearrange(result, '(p1 t1) (p2 t2) -> p1 t1 p2 t2', p1 = p, p2=p)
            mask = mask.mean(dim=1) # mean over time?
            mask = mask[0,1:,:]
            width = int(mask.size(0)**0.5)
            mask = rearrange(mask, '(h w) t -> h w t', w = width).numpy()
            mask = mask / np.max(mask)
        else:
            mask = rearrange(result[0], '(p1 t1) -> p1 t1', p1=p, t1=t)
        #mask = rearrange(result, '(p1 t1) (p2 t2) -> p1 t1 p2 t2', p1 = p, p2=p)
        # mask = mask.mean(dim=1) # mean over time?
        # mask = mask[0,1:,:]
        # width = int(mask.size(0)**0.5)
        # mask = rearrange(mask, '(h w) t -> h w t', w = width).numpy()
        # mask = mask / np.max(mask)
        return mask
    # mask = generate_mask(self.attentions,1)
    return (generate_mask(self.attentions),generate_mask(self.attentions_min),generate_mask(self.attentions_max)),preds

