import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from .avcil_layers import LSCLinear, SplitLSCLinear


class AVAudioVisualBackbone(nn.Module):
    """
    Audio-Visual feature extraction backbone
    For the design phylosophy of FACIL, the network is initialized with model (def __init__(self, model, remove_existing_head=False))
    That's why we design the backbone separately, rather than integrating it directly into the main network class like AVCIL. 

    Inputs: tuple (visual_features, audio_features)
        - visual_features: [B, num_frames, visual_embed_dim] ([B, T, D_v])
        - audio_features: [B, audio_embed_dim] ([B, D_a])
    Outputs:
        - fused_features: [B, embed_dim] ([B, D_f])
    """
    def __init__(self, visual_embed_dim=768, audio_embed_dim=768, embed_dim=768, num_segments=8):
        super().__init__()
        self.visual_embed_dim = visual_embed_dim
        self.audio_embed_dim = audio_embed_dim
        self.embed_dim = embed_dim
        self.num_segments = num_segments

        # AVCIL Style projection layers
        self.audio_proj = nn.Linear(audio_embed_dim, embed_dim)
        self.visual_proj = nn.Linear(visual_embed_dim, embed_dim)
        self.attn_audio_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_visual_proj = nn.Linear(embed_dim, embed_dim)

        # no dropout or batch norm layers, follow the original AVCIL design. But still can easily add them if needed.

        # head_var is a attribute that tells LLL_Net (the network wrapper for Long-Tailed-CIL): "Where is the classification head in this model?"
        # This is designed for delete the original head when adding CIL heads. Since AVCIL uses a single head design, we can just set head_var to 'head' and maintain a single head in the main network class.
        self.head_var = 'head'

    def audio_visual_attention(self, audio_features, visual_features):

        proj_audio_features = torch.tanh(self.attn_audio_proj(audio_features))
        proj_visual_features = torch.tanh(self.attn_visual_proj(visual_features))

        # (BS, 8, 14*14, 768)
        spatial_score = torch.einsum("ijkd,id->ijkd", [proj_visual_features, proj_audio_features])
        spatial_attn_score = F.softmax(spatial_score, dim=2)

        # (BS, 8, 768)
        spatial_attned_proj_visual_features = torch.sum(spatial_attn_score * proj_visual_features, dim=2)

        # (BS, 8, 768)
        temporal_score = torch.einsum("ijd,id->ijd", [spatial_attned_proj_visual_features, proj_audio_features])
        temporal_attn_score = F.softmax(temporal_score, dim=1)

        return spatial_attn_score, temporal_attn_score

    def forward(self, x, return_intermediates=False):
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError("AVAudioVisualBackbone expects x=(visual, audio)")

        visual, audio = x
        # If we want to singly use visual/audio modality, we can simply set the other modality into zeros, instead of changing the backbone design.
        if visual is None:
            raise ValueError("input frames are None when modality contains visual")
        if audio is None:
            raise ValueError("input audio are None when modality contains audio")

        # Align with avcil dimentsion. Normally we just need the first if branch.
        if visual.dim() == 3:
            visual = visual.view(visual.shape[0], self.num_segments, -1, 768)
        elif visual.dim() == 4:
            pass
        else:
            raise ValueError(f"Unexpected visual shape: {tuple(visual.shape)}")

        spatial_attn_score, temporal_attn_score = self.audio_visual_attention(audio, visual)
        visual_pooled_feature = torch.sum(spatial_attn_score * visual, dim=2)          # [B,8,768]
        visual_pooled_feature = torch.sum(temporal_attn_score * visual_pooled_feature, dim=1)  # [B,768]

        audio_feature = F.relu(self.audio_proj(audio))
        visual_feature = F.relu(self.visual_proj(visual_pooled_feature))
        audio_visual_features = visual_feature + audio_feature

        if return_intermediates:
            return (
                audio_visual_features,
                visual_pooled_feature,
                audio_feature,
                visual_feature,
                spatial_attn_score,
                temporal_attn_score,
            )
        return audio_visual_features


class AVCILNet(nn.Module):
    '''
    The structure aligns with Long-Tailed-CIL/FACIL design philosophy:
    - Single-head expandable classifier (functionally aligned with AVCIL incremental_classifier)
    - Provides add_head / heads / task_cls / task_offset / freeze_* / get_copy interfaces
    '''
    def __init__(self, backbone, use_lsc=False):
        super().__init__()
        self.backbone = backbone
        self.use_lsc = use_lsc

        self.classifier = None
        self.num_classes = 0

        # Align with LLL_Net design for head management. Since AVCIL uses single head, we can just maintain one head and expose it via heads[0].
        self.model = backbone
        self.head_var = "heads"
        self.heads = nn.ModuleList()     # 始终只放一个 head（单头设计）
        self.task_cls = torch.tensor([], dtype=torch.long)
        self.task_offset = torch.tensor([], dtype=torch.long)
        self.schedule_step = [80, 120]

    def _build_classifier(self, num_classes):
        if self.use_lsc:
            return LSCLinear(768, num_classes)
        return nn.Linear(768, num_classes)

    def incremental_classifier(self, numclass):
        # For the first step add head
        if self.num_classes == 0:
            self.classifier = self._build_classifier(numclass)
            self.num_classes = numclass
            return

        old_classifier = self.classifier
        old_out = old_classifier.out_features
        in_features = old_classifier.in_features

        new_classifier = self._build_classifier(numclass)

        # torch.no_grad() to avoid tracking in autograd, since this is just weight copying, not a real forward pass. It's more safe
        with torch.no_grad():
            if self.use_lsc:
                K = old_classifier.K
                new_classifier.weight.data[:K * old_out] = old_classifier.weight.data[:K * old_out]
            else:
                new_classifier.weight.data[:old_out] = old_classifier.weight.data[:old_out]
                new_classifier.bias.data[:old_out] = old_classifier.bias.data[:old_out]

        self.classifier = new_classifier
        self.num_classes = numclass

    def add_head(self, num_outputs):
        """
        适配 FACIL 的 add_head(task_ncls):
        单头模式下，扩展 total classes。
        """
        if num_outputs <= 0:
            return
        new_total = self.num_classes + int(num_outputs)
        self.incremental_classifier(new_total)

        # use head[0] to maintain the single head, for compatibility with LLL_Net design. 
        self.heads = nn.ModuleList([self.classifier])

        # in multi-head design, task_cls is like [10,10,10], while in single-head design, we can just maintain the total class number in task_cls[0], for example [30], and keep task_offset[0] as 0.
        self.task_cls = torch.tensor([self.num_classes], dtype=torch.long)
        self.task_offset = torch.tensor([0], dtype=torch.long)

    def forward(
        self,
        x,
        return_features=False,
        out_logits=True,
        out_features=False,
        out_features_norm=False,
        out_feature_before_fusion=False,
        out_attn_score=False,
        AFC_train_out=False
    ):
        """
        同时兼容:
        - FACIL: forward(x, return_features)
        - AVCIL: out_* / AFC_train_out 风格
        """
        (
            audio_visual_features,
            visual_pooled_feature,
            audio_feature,
            visual_feature,
            spatial_attn_score,
            temporal_attn_score,
        ) = self.backbone(x, return_intermediates=True)

        # size of logits: [B, num_classes]
        logits = self.classifier(audio_visual_features)
        outputs = ()

        # FACIL style
        if return_features:
            return logits, audio_visual_features

        # AVCIL style, this part is almost the same as the original AVCIL code.
        if AFC_train_out:
            audio_feature.retain_grad()
            visual_feature.retain_grad()
            visual_pooled_feature.retain_grad()
            outputs += (logits, visual_pooled_feature, audio_feature, visual_feature)
            return outputs

        if out_logits:
            outputs += (logits,)
        if out_features:
            if out_features_norm:
                outputs += (F.normalize(audio_visual_features),)
            else:
                outputs += (audio_visual_features,)
        if out_feature_before_fusion:
            outputs += (F.normalize(audio_feature), F.normalize(visual_feature))
        if out_attn_score:
            outputs += (spatial_attn_score, temporal_attn_score)

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def forward_with_attention(self, x, return_attn=False):
        logits, features = self.forward(x, return_features=True)
        if not return_attn:
            return logits

        _, _, _, _, spatial_attn_score, temporal_attn_score = self.backbone(
            x, return_intermediates=True
        )
        return logits, features, spatial_attn_score, temporal_attn_score

    def get_copy(self):
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        self.load_state_dict(deepcopy(state_dict))

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()


class AVCILNetWrapper(nn.Module):
    """
    将单头 logits 转成 FACIL 常见的 list[Tensor] 形式，减少 approach 改动。
    """
    def __init__(self, avnet):
        super().__init__()
        self.avnet = avnet
        self.heads = nn.ModuleList([])
        self.task_cls = []
        self.task_offset = []

    def add_head(self, num_outputs):
        self.avnet.add_head(num_outputs)
        self.update_meta()

    def update_meta(self):
        self.heads = self.avnet.heads
        self.task_cls = self.avnet.task_cls.tolist() if torch.is_tensor(self.avnet.task_cls) else self.avnet.task_cls
        self.task_offset = self.avnet.task_offset.tolist() if torch.is_tensor(self.avnet.task_offset) else self.avnet.task_offset

    def forward(self, x, return_features=False):
        if return_features:
            logits, features = self.avnet.forward(x, return_features=True)
            return [logits], features
        logits = self.avnet.forward(x, return_features=False)
        return [logits]

    def freeze_backbone(self):
        self.avnet.freeze_backbone()

    def freeze_all(self):
        self.avnet.freeze_all()

    def freeze_bn(self):
        self.avnet.freeze_bn()

    def get_copy(self):
        return self.avnet.get_copy()

    def set_state_dict(self, state_dict):
        self.avnet.set_state_dict(state_dict)


# ---- factory functions required by main_incremental.py ----
# We provide factory functions to create the backbone, main net, and optional wrapper, just like other networks (resnet series).
# We also have choice to use the classes directly, like LeNet or VggNet. (VggNet also has a factory function, but not using it. LeNet doesn't have a factory function.) 
# In conclusion, using factory functions is a more consistent design.
def avcil_backbone(pretrained=False, **kwargs):
    return AVAudioVisualBackbone(**kwargs)


def avcil_net(pretrained=False, **kwargs):
    # default: use single-head AVCIL net directly
    backbone = kwargs.pop('backbone', None)
    if backbone is None:
        backbone = AVAudioVisualBackbone()
    use_lsc = kwargs.pop('use_lsc', False)
    return AVCILNet(backbone=backbone, use_lsc=use_lsc)


def avcil_wrapper(pretrained=False, **kwargs):
    # optional wrapper version
    avnet = kwargs.pop('avnet', None)
    if avnet is None:
        backbone = kwargs.pop('backbone', None)
        if backbone is None:
            backbone = AVAudioVisualBackbone()
        use_lsc = kwargs.pop('use_lsc', False)
        avnet = AVCILNet(backbone=backbone, use_lsc=use_lsc)
    return AVCILNetWrapper(avnet)