import torch

from .ssl_encoder import PretrainMatchedEncoder
from .unetr import UNETR


class Model_Config:
    pass


def _normalize_pretrain_state_dict(raw: dict) -> dict:
    out = {}
    for k, v in raw.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        out[nk] = v
    return out


def _checkpoint_has_gasp_modules(state: dict) -> bool:
    return any(k.startswith("geo_prior_gen.") for k in state)


def load_gasp_pretrained_encoder(encoder: PretrainMatchedEncoder, ckpt_path: str, *, strict: bool = True) -> None:
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ck, dict) and "state_dict" in ck:
        raw = ck["state_dict"]
    elif isinstance(ck, dict):
        raw = ck
    else:
        raise TypeError(f"不支持的 checkpoint 类型: {type(ck)}")

    pretrained_state = _normalize_pretrain_state_dict(raw)
    if not _checkpoint_has_gasp_modules(pretrained_state):
        raise RuntimeError(
            "checkpoint 中未找到 geo_prior_gen.* 权重。"
            "请使用 GeoPriorGen3DOn=1 且 HybridSparseAttnOn=1 训练得到的 full pretrain 权重，"
            "勿使用 baseline / 开关关闭的 ssl_checkpoint。"
        )

    incomp = encoder.load_state_dict(pretrained_state, strict=strict)
    missing = getattr(incomp, "missing_keys", None) or incomp[0]
    unexpected = getattr(incomp, "unexpected_keys", None) or incomp[1]
    print(
        "GASP encoder load:",
        f"keys={len(pretrained_state)}",
        f"missing={len(missing)}",
        f"unexpected={len(unexpected)}",
    )
    if missing:
        print("  missing (head):", missing[:8], "..." if len(missing) > 8 else "")
    if unexpected:
        print("  unexpected:", unexpected[:8], "..." if len(unexpected) > 8 else "")
    if strict and (missing or unexpected):
        raise RuntimeError("GASP 预训练权重与编码器结构未严格对齐，请检查 checkpoint 与 newFineTune 版本。")


def build_model(args, roi_size):
    model = UNETR(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        img_size=roi_size,
        patch_size=[args.patch_size, args.patch_size, args.patch_size],
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

    encoder = PretrainMatchedEncoder(config)
    n_enc = sum(p.numel() for p in encoder.parameters())
    print("GASP encoder params (M):", n_enc / 1e6)

    if args.pretrain_path is not None:
        strict_load = not getattr(args, "pretrain_load_nonstrict", False)
        load_gasp_pretrained_encoder(encoder, args.pretrain_path, strict=strict_load)

    model.vit = encoder
    return model
