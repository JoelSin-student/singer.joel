# Utility helpers
#
#
#
#

def print_config(params):
    print("<< Final Configuration Settings >>", flush=True)
    print("[Model Parameters]")
    print(f"  model_mode: {params.get('model_mode', 'n/a')}")
    print(f"  d_model: {params.get('d_model', 'n/a')}")
    print(f"  n_head: {params.get('n_head', 'n/a')}")
    print(f"  num_encoder_layer: {params.get('num_encoder_layer', 'n/a')}")
    print(f"  dropout: {params.get('dropout', 'n/a')}")
    print(f"  sequence_size: {params.get('sequence_len', 'n/a')}")
    print(f"  use_gradient_data: {params.get('use_gradient_data', False)}")
    print("\n[Training Parameters]")
    print(f"  Epochs: {params.get('num_epoch', 'n/a')}")
    print(f"  Batch Size: {params.get('batch_size', 'n/a')}")
    print("\n[Optimization Parameters]")
    print(f"  Learning Rate: {params.get('learning_rate', 'n/a')}")
    print(f"  Weight Decay: {params.get('weight_decay', 'n/a')}")
    print("\n[Loss Function Parameters]")
    print(f"  Loss Alpha: {params.get('loss_alpha', 'mse-only')}")
    print(f"  Loss Beta: {params.get('loss_beta', 'mse-only')}")
    print("\n[Other Settings]")
    print(f"  Input Dimension: {params.get('input_dim', 'n/a')}")
    print(f"  Output Dimension: {params.get('output_dim', 'n/a')}")
    print(f"  Number of Joints: {params.get('num_joints', 'n/a')}")
    print(f"  Number of Dimensions: {params.get('num_dims', 'n/a')}")
    print("---" * 20)


def format_ablation_tag(abl_id):
    if abl_id is None:
        return ""

    tag = str(abl_id).strip()
    if not tag:
        return ""

    return tag if tag.startswith("abl_id_") else f"abl_id_{tag}"


def resolve_ablation_id(config, section_name):
    section = config.get(section_name, {})
    if isinstance(section, dict):
        abl_id = section.get("abl_id", None)
        if abl_id is not None and str(abl_id).strip():
            return abl_id
    abl_id = config.get("abl_id", None)
    if abl_id is not None and str(abl_id).strip():
        return abl_id
    return None


def join_nonempty(*parts):
    values = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if text:
            values.append(text)
    return "_".join(values)