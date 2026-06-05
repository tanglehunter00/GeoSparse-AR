import torch

from .unetr import UNETR
from .base_model import BaseModel


class Model_Config():
    def __int__(self):
        pass


def build_model(args, roi_size):
    in_channels = args.in_channels
    n_class = args.out_channels

    model = UNETR(
        in_channels=in_channels,
        out_channels=n_class,
        img_size=roi_size,
        patch_size=[args.patch_size, args.patch_size, args.patch_size]
    )

    config = Model_Config()
    config.pos_type = args.pos_type
    config.img_size = roi_size
    config.patch_size = [args.patch_size, args.patch_size, args.patch_size]
    config.hidden_size = 768
    config.intermediate_size = 3072
    config.num_attention_heads = 12
    config.num_key_value_heads = 12
    config.num_hidden_layers = 12
    encoder = BaseModel(config)
    model_size = sum(t.numel() for t in encoder.parameters())
    print('model_size', model_size)

    if args.pretrain_path is not None:
        ck = torch.load(args.pretrain_path, map_location="cpu")
        if isinstance(ck, dict) and "state_dict" in ck:
            model_dict = ck["state_dict"]
        else:
            model_dict = ck if isinstance(ck, dict) else {}
        pretrained_state = {}
        for k, v in model_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            if nk.startswith("_orig_mod."):
                nk = nk[len("_orig_mod.") :]
            if nk.startswith("model."):
                pretrained_state[nk[len("model.") :]] = v
        incomp = encoder.load_state_dict(pretrained_state, strict=False)
        nm = incomp.missing_keys if hasattr(incomp, "missing_keys") else incomp[0]
        nu = incomp.unexpected_keys if hasattr(incomp, "unexpected_keys") else incomp[1]
        print("Loaded ViT pretrained keys:", len(pretrained_state), "missing:", len(nm), "unexpected:", len(nu))

    model.vit = encoder
    return model

    
