# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import sys

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from transformers import AutoProcessor, AutoTokenizer, CLIPModel

toPIL = T.ToPILImage()
import torch.nn as nn
from einops import rearrange


class ResNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.model_name = cfg["model_name"]

        self.clip = CLIPModel.from_pretrained(self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def get_text_embeddings(self, texts):
        """Input is list of texts."""
        text_inputs = self.tokenizer(texts, padding=True, return_tensors="pt")
        text_features = self.clip.get_text_features(**text_inputs)
        return text_features

    def get_image_embeddings(self, frames_batch):
        """Input is list of list of images."""
        self.batch_size = len(frames_batch)
        vid_embs = []
        for frames in frames_batch:
            image_inputs = self.processor(images=frames, return_tensors="pt")
            frame_embeddings = self.clip.get_image_features(**image_inputs)
            frame_embeddings = rearrange(frame_embeddings, "(b n) d -> b n d", b=len(frames_batch))
            # Normalize, mean aggregate and return normalized video_embeddings
            frame_embeddings = frame_embeddings / frame_embeddings.norm(dim=-1, keepdim=True)
            video_embeddings = frame_embeddings.mean(dim=1)
            video_embeddings = video_embeddings / video_embeddings.norm(dim=-1, keepdim=True)
            vid_embs.append(video_embeddings)
        return torch.cat(vid_embs, dim=0)
