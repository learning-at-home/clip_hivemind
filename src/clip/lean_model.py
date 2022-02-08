import numpy as np
import torch
from torch import nn

from lib.models.transformer import LeanTransformer, LeanTransformerConfig
from lib.modules import LeanSelfAttention, LeanFFN


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, output_dim: int, config: LeanTransformerConfig):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        width = config.hidden_size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = LeanTransformer(config)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        # no need to transpose: LeanTransformer already accepts NLD
        x = self.transformer(x).last_hidden_state

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x
    
class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 vision_config: LeanTransformerConfig,
                 text_config: LeanTransformerConfig
                 ):
        super().__init__()

        self.context_length = context_length

        self.visual = VisualTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            output_dim=embed_dim,
            config=vision_config,
        )

        self.transformer = LeanTransformer(text_config)
        transformer_width = self.transformer.config.hidden_size

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        proj_std = (self.transformer.config.hidden_size ** -0.5) * ((2 * self.transformer.config.num_hidden_layers) ** -0.5)
        attn_std = self.transformer.config.hidden_size ** -0.5
        fc_std = (2 * self.transformer.config.hidden_size) ** -0.5
        for group in self.transformer.layer_groups:
            for block in group.layers:
                assert isinstance(block.attention, LeanSelfAttention)
                nn.init.normal_(block.attention.dense_qkv.weight, std=attn_std)
                nn.init.normal_(block.attention.dense_out.weight, std=proj_std)

                assert isinstance(block.ffn, LeanFFN)
                nn.init.normal_(block.ffn.dense_i2h.weight, std=fc_std)
                nn.init.normal_(block.ffn.dense_h2o.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.config.hidden_size ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)

        # no need to transpose: LeanTransformer already accepts NLD
        x = self.transformer(x).last_hidden_state

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features, text_features, self.logit_scale.exp()


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    raise NotImplementedError()

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()


class SimpleCLIP(CLIP):
    """A thin wrapper over CLIP that makes it backwards-compatible w.r.t. the parameters of the original OpenCLIP"""
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: int,
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 gradient_checkpointing: bool = False,
                 reversible: bool = True,  # enabled manually, not present in config
                 ):
        vision_config = LeanTransformerConfig(
            hidden_size=vision_width, intermediate_size=vision_width * 4,
            num_hidden_layers=vision_layers, num_attention_heads=vision_width // 64,
            position_embedding_type="absolute", # not using "rotary" because it's meant for 1-dimensional tasks
            max_position_embeddings=image_resolution ** 2 // vision_patch_size ** 2,
            reversible=reversible,
        )
        text_config = LeanTransformerConfig(
            hidden_size=transformer_width, intermediate_size=transformer_width * 4,
            num_hidden_layers=transformer_layers, num_attention_heads=transformer_heads,
            position_embedding_type="absolute",  # you *should* use rotary here after we're done testing!
            max_position_embeddings=context_length,  # remove this after switching to rotary
            reversible=reversible,
        )
        super().__init__(
            embed_dim, image_resolution, vision_patch_size, context_length, vocab_size, vision_config, text_config
        )
        for param in self.parameters():
            param.grad = torch.zeros_like(param)
        if gradient_checkpointing:
            self.transformer._get_sequential().gradient_checkpointing = True
            self.visual.transformer._get_sequential().gradient_checkpointing = True
