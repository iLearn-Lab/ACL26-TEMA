"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import warnings
import tiktoken
import re
import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.models.blip_models.blip import BlipBase
from lavis.models.blip_models.blip_outputs import BlipOutputFeatures
from lavis.models.med import XBertEncoder
from lavis.models.vit import VisionTransformerEncoder
from torch import nn
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from typing import Tuple, Optional, Union


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        device = x.device
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape

        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            mask = ~mask
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            mask = mask.unsqueeze(3)
            attention = attention.masked_fill(mask, -float("inf"))  # batch_size, L, L, head

        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)

class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

   

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=True):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.MLP = nn.Sequential(
                        nn.LayerNorm(in_channels * 2),
                        nn.Linear(in_channels * 2, in_channels),
                        nn.GELU(),
                        nn.Linear(in_channels, out_channels)
                        )


    def forward(self, x, text_embed):
        text_embed_ = torch.cat([x, text_embed], dim=-1)
        batch = x.shape[0]
        chanel = x.shape[1] * 2
        gamma = self.MLP(text_embed_)
        x = gamma * x + (1-gamma) * text_embed
        #x =  x + text_embed
        return x


@registry.register_model("BlipCir")
class BlipCir(BlipBase):
    """
    Class for BLIP feature extractor.

    Supported model types:
        - base: BLIP base model with pre-trained weights from capfilt by BLIP large model.

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip_feature_extractor", "base")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip_feature_extractor_base.yaml",
        "large": "configs/models/blip_cir_large.yaml",
    }

    def __init__(self, image_encoder, text_encoder, embed_dim, cfg, max_txt_len=512, q=1):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder = image_encoder
        self.text_encoder = text_encoder
        self.loss_T = nn.Parameter(torch.tensor([10.]))
        # creating projection layers for ITC
        text_width = text_encoder.config.hidden_size
        vision_width = image_encoder.vision_width

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.max_txt_len = max_txt_len
        self.cosine_criterion = nn.CosineEmbeddingLoss()
        self.criterion_target = torch.as_tensor([1])

        self.transformer1 = Transformer(dim_self=embed_dim, num_heads=8, dim_ref=embed_dim,num_layers=1)
        self.transformer2 = Transformer(dim_self=embed_dim, num_heads=8, dim_ref=embed_dim,num_layers=1)
        self.ini_ground_img = nn.Parameter(torch.zeros(1, q, embed_dim)).cuda()
        self.ini_ground_txt = nn.Parameter(torch.zeros(1, q, embed_dim)).cuda()
        self.q = q
        self.affine = FeatureWiseAffine(embed_dim, embed_dim, use_affine_level=True)

        

    
    def forward(self, samples):
        image = samples["image"]
        target = samples["target"]
        caption = samples["text_input"]
        # summ = samples["summ"]

        ###============== reference text fusion ===================###
        with torch.no_grad():
            image_embeds = self.visual_encoder.forward_features(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
            img_local = self.vision_proj(image_embeds[:, 1:, :])
            img_global = self.vision_proj(image_embeds[:,0,:])
        bsz, dim = img_global.size()

        
        
        text = self.tokenizer(caption, return_tensors="pt", padding=True, max_length=self.max_txt_len).to(
            self.device
        )

        text.input_ids[:, 0] = self.tokenizer.enc_token_id
        
        
        text_obj = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True, mode = 'text')
        text_local = self.text_proj(text_obj.last_hidden_state[:,1:,:])
        text_global = self.text_proj(text_obj.last_hidden_state[:,0,:])
        
        with torch.no_grad():
            summ = self.tokenizer(summ, return_tensors="pt", padding=True, max_length=self.max_txt_len).to(
                self.device)
            summ.input_ids[:, 0] = self.tokenizer.enc_token_id
            summ_obj = self.text_encoder(summ.input_ids, attention_mask = summ.attention_mask, return_dict = True, mode = 'text')
            summ_feats = F.normalize(self.text_proj(summ_obj.last_hidden_state[:,0,:]), dim=-1).unsqueeze(1)


        img_median = img_global.unsqueeze(1)

        img_ground_tokens = self.ini_ground_img.expand(bsz, self.q, dim)
        img_ground_tokens = self.transformer1(torch.cat([img_ground_tokens, img_median, img_local], dim=1))[:, :self.q, :]
        text_ground_tokens = self.ini_ground_txt.expand(bsz, self.q, dim)
        # text_ground_tokens = self.transformer2(torch.cat([text_ground_tokens, summ_feats, text_local], dim=1))[:, :self.q, :]
        text_ground_tokens = self.transformer2(torch.cat([text_ground_tokens, text_local], dim=1))[:, :self.q, :]

        img_tokens = torch.cat([img_global.unsqueeze(1), img_ground_tokens], dim=1)
        text_tokens = torch.cat([text_global.unsqueeze(1), text_ground_tokens], dim=1)
        
        """
        fusion_output = self.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, 0, :]), dim=-1
        )
        """
        
        fusion_feats = self.affine(img_tokens, text_tokens)

        
        with torch.no_grad():
            target_embeds = self.visual_encoder.forward_features(target)
            target_local = self.vision_proj(target_embeds[:, 1:, :])
            target_global = self.vision_proj(target_embeds[:,0,:])
            
            #target_features = target_global
            #target_feats = F.normalize(target_features, dim=-1)

        target_ground_tokens = self.transformer1(torch.cat([target_global.unsqueeze(1).expand(bsz, self.q, dim), target_local], dim=1))[:, :self.q, :]
        target_tokens = torch.cat([target_global.unsqueeze(1), target_ground_tokens], dim=1)

        fusion_feats = F.normalize(torch.mean(fusion_feats, dim=1), p=2, dim=-1)
        target_feats = F.normalize(torch.mean(target_tokens, dim=1), p=2, dim=-1)
        
        
        #sim_i2t = fusion_feats @ target_feats.t()

        bs = image.size(0)
        targets = torch.linspace(0,  bs - 1, bs, dtype=int).to(
            image.device
        )

        #sim_i2t = sim_i2t / self.temp

        #loss_itc = F.cross_entropy(sim_i2t, targets)

        
        loss_bbc = self.info_nce(fusion_feats, target_feats)
        loss_ortho = self.orthogonal_regularization(img_ground_tokens) + self.orthogonal_regularization(text_ground_tokens)
        loss_summ =  self.cosine_loss(summ_feats.squeeze(1), F.normalize(torch.mean(text_ground_tokens, dim=1), p=2, dim=-1))
        

        return { 'loss_bbc': loss_bbc, 'loss_ortho': loss_ortho, 'loss_summ': loss_summ}
        #return { 'loss_bbc': loss_bbc,  'loss_summ': 0.95 * loss_summ}
        # return {'loss_itc': 0}        

    





    #@torch.no_grad()
    def extract_retrieval_compose(self, image, caption):
        ###============== reference text fusion ===================###

        image_embeds = self.visual_encoder.forward_features(image)

        img_local = self.vision_proj(image_embeds[:, 1:, :])
        img_global = self.vision_proj(image_embeds[:,0,:])
        bsz, dim = img_global.size()

        
        
        text = self.tokenizer(caption, return_tensors="pt", padding=True, max_length=self.max_txt_len).to(
            self.device
        )

        text.input_ids[:, 0] = self.tokenizer.enc_token_id
        
        text_obj = self.text_encoder(text.input_ids, attention_mask = text.attention_mask, return_dict = True, mode = 'text')
        text_local = self.text_proj(text_obj.last_hidden_state[:,1:,:])
        text_global = self.text_proj(text_obj.last_hidden_state[:,0,:])
        
        
        # summ = self.tokenizer(summ, return_tensors="pt", padding=True, max_length=self.max_txt_len).to(
        #    self.device)
        # summ.input_ids[:, 0] = self.tokenizer.enc_token_id
        # summ_obj = self.text_encoder(summ.input_ids, attention_mask = summ.attention_mask, return_dict = True, mode = 'text')
        # summ_feats = F.normalize(self.text_proj(summ_obj.last_hidden_state[:,0,:]), dim=-1).unsqueeze(1)
        
        # attention_mask = select_top_channels(summ_feats, img_local, 400)
        # img_local_selected = attention_mask
        # img_median = torch.mean(img_local_selected, dim=1).unsqueeze(1)
        
        img_ground_tokens = self.ini_ground_img.expand(bsz, self.q, dim)
        img_ground_tokens = self.transformer1(torch.cat([img_ground_tokens, img_local], dim=1))[:, :self.q, :]
        text_ground_tokens = self.ini_ground_txt.expand(bsz, self.q, dim)
        text_ground_tokens = self.transformer2(torch.cat([text_ground_tokens, text_local], dim=1))[:, :self.q, :]

        img_tokens = torch.cat([img_global.unsqueeze(1), img_ground_tokens], dim=1)
        text_tokens = torch.cat([text_global.unsqueeze(1), text_ground_tokens], dim=1)
        
        fusion_feats = self.affine(img_tokens, text_tokens)
        fusion_feats = F.normalize(torch.mean(fusion_feats, dim=1), p=2, dim=-1)
        # print("fus:",fusion_feats.shape)
        return fusion_feats#.unsqueeze(1).unsqueeze(1)

    #@torch.no_grad()
    def extract_retrieval_target(self, img):
        
        target_embeds = self.visual_encoder.forward_features(img)
        target_local = self.vision_proj(target_embeds[:, 1:, :])
        target_global = self.vision_proj(target_embeds[:,0,:])
        
        target_ground_tokens = self.transformer1(torch.cat([target_global.unsqueeze(1).expand(-1, self.q, -1), target_local], dim=1))[:, :self.q, :]
        target_tokens = torch.cat([target_global.unsqueeze(1), target_ground_tokens], dim=1)
#
        target_feats = F.normalize(torch.mean(target_tokens, dim=1), p=2, dim=-1)
        
        #target_embeds = self.visual_encoder.forward_features(img)[:,0,:]
##
        #target_features = self.vision_proj(target_embeds)
        #target_feats = F.normalize(target_features, dim=-1)
        # print("tar:",target_feats.shape)
        return target_feats#.permute(0, 2, 1)




    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.

        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".

        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.

        Examples:
        ```python
            >>> from PIL import Image
            >>> from lavis.models import load_model_and_preprocess
            >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB")
            >>> caption = "a large fountain spewing water into the air"
            >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_feature_extractor", is_eval=True)
            >>> image = vis_processors["eval"](raw_image).unsqueeze(0)
            >>> text_input = txt_processors["eval"](caption)

            >>> sample = {"image": image, "text_input": [text_input]}

            >>> features_multimodal = model.extract_features(sample)
            >>> features_multimodal.keys()
            odict_keys(['image_embeds', 'multimodal_embeds'])
            >>> features_multimodal.image_embeds.shape
            torch.Size([1, 197, 768])
            >>> features_multimodal.multimodal_embeds.shape
            torch.Size([1, 12, 768])

            >>> features_text = model.extract_features(sample, mode="text")
            >>> features_text.keys()
            odict_keys(['text_embeds', 'text_features'])
            >>> features_text.text_embeds.shape
            torch.Size([1, 12, 768])
            >>> features_text.text_features.shape
            torch.Size([1, 12, 256])

            >>> features_image = model.extract_features(sample, mode="image")
            >>> features_image.keys()
            odict_keys(['image_embeds', 'image_features'])
            >>> features_image.image_embeds.shape
            torch.Size([1, 197, 768])
            >>> features_image.image_features.shape
            torch.Size([1, 197, 256])
        ```
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return image features
            image_embeds = self.visual_encoder.forward_features(image)

            image_features = self.vision_proj(image_embeds)
            image_features = F.normalize(image_features, dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            # return text features
            text_output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
                mode="text",
            )
            text_embeds = text_output.last_hidden_state

            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel features
            image_embeds = self.visual_encoder.forward_features(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            text.input_ids[:, 0] = self.tokenizer.enc_token_id

            output = self.text_encoder(
                text.input_ids,
                attention_mask=text.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg=None):
        # set from_pretrained=True to load weights for 'bert-base-uncased'
        image_encoder = VisionTransformerEncoder.from_config(cfg)
        text_encoder = XBertEncoder.from_config(cfg)

        embed_dim = cfg.get("embed_dim", 256)
        max_txt_len = cfg.get("max_txt_len", 512)

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            embed_dim=embed_dim,
            max_txt_len=max_txt_len,
            cfg = cfg
        )

        # load pre-trained weights
        pretrain_path = cfg.get("pretrained", None)
        if pretrain_path is not None:
            msg = model.load_from_pretrained(url_or_filename=pretrain_path)
        else:
            warnings.warn("No pretrained weights are loaded.")

        return model
    
    def info_nce(self, query, target):
        x = torch.mm(query, target.T)
        labels = torch.arange(query.shape[0]).long().cuda()
        return F.cross_entropy(x * self.loss_T, labels)
    
        
    def orthogonal_regularization(self, templates):
        # batch_size, length, dim
        batch_size, length, dim = templates.size()
        device = templates.device
        norm_templates = F.normalize(templates, p=2, dim=-1)
        # (B,L,D) * (B,D,L)
        cosine_score = torch.matmul(norm_templates, norm_templates.permute(0,2,1).contiguous()) # batch_size, length, length 
        eye_matrix = torch.eye(length).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        l2_loss = torch.nn.MSELoss()
        return l2_loss(cosine_score, eye_matrix)
    
    def cosine_loss(self, X1, X2):
        cosine_loss = self.cosine_criterion(X2, X1, self.criterion_target.cuda())
        return cosine_loss
    
    

def split_caption(s):
    head = s[0]
    s_set = set(s[1:])
    ss = []
    ss.append(head)
    for i in s_set:
        ss.append(i)
    return ss

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def trun_text(text: str):
    if isinstance(text, list):
        result_list = []
        for single_text in text:
            tokens = re.split(r'\d+\.\s*', single_text)

            tokens = [t for t in tokens if t.strip()]
            result_list.append(tokens)
        return result_list
    
    else:
        result_list = re.split(r'\d+\.\s*', text)
        result_list = [t for t in result_list if t.strip()]
        return result_list

def select_top_channels(A, B, top_k=300):
    A = A.expand(-1, B.size(1), -1)  # (32, 576, 256)

    A_norm = F.normalize(A, p=2, dim=-1)  # (32, 576, 256)
    B_norm = F.normalize(B, p=2, dim=-1)  # (32, 576, 256)

    similarity = torch.sum(A_norm * B_norm, dim=-1)  # (32, 576)

    _, indices = torch.topk(similarity, k=top_k, dim=-1)  # (32, 300)

    batch_size = B.size(0)
    batch_indices = torch.arange(batch_size).view(-1, 1).expand(-1, top_k)  # (32, 300)
    selected_B = B[batch_indices, indices]  # (32, 300, 256)
    
    return selected_B

"""
def select_channels_with_threshold(A, B, threshold=0.1):
    A = A.expand(-1, B.size(1), -1)  # (32, 576, 256)
    
    A_norm = F.normalize(A, p=2, dim=-1)  # (32, 576, 256)
    B_norm = F.normalize(B, p=2, dim=-1)  # (32, 576, 256)
    
    similarity = torch.sum(A_norm * B_norm, dim=-1)  # (32, 576)

    mask = (similarity > threshold).float()[..., None]  # (32, 576)
    
    return mask
"""
