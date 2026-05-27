import torch
import numpy as np
import argparse
import os
from pathlib import Path

from .models.v3 import load_model
# from .models.v3_clip_embed import load_model
# from .models.v6 import load_model
# from .models.v35 import load_model
#from .models.v5 import load_model
# from .models.v4 import load_model


def make_dummy(batch=1, T=77, L=7, device="cuda", float_dtype=torch.float32,
               clip_embed=False):
    z_long = lambda *shape: torch.zeros(*
                                        shape, dtype=torch.long,  device=device)
    z_float = lambda *shape: torch.zeros(*
                                         shape, dtype=float_dtype, device=device)

    # clip_embed models use float embeddings; one-hot models use int64 token IDs
    tokens = z_float(batch, T) if clip_embed else z_long(batch, T)
    token_mask = z_float(batch, T)

    lora_ids = z_long(batch, L)
    lora_w = z_float(batch, L)

    cfg = z_float(batch, 1)
    n_loras = z_float(batch, 1)

    sampler_id = z_long(batch, 1)
    steps_log = z_float(batch, 1)
    steps_bucket = z_long(batch, 1)

    upscaler_id = z_long(batch, 1)
    up_has = z_float(batch, 1)
    up_steps = z_float(batch, 1)
    denoise = z_float(batch, 1)

    model_id = z_long(batch, 1)

    return (tokens, token_mask,
            lora_ids, lora_w,
            cfg, n_loras,
            sampler_id, steps_log, steps_bucket,
            upscaler_id, up_has, up_steps, denoise, model_id)


def export_onnx_fp32(out_path, model_path, device="cuda"):
    # dummy = make_dummy(batch=2, T=768, L=7, device=device,
    #                    float_dtype=torch.float32, clip_embed=True)
    dummy = make_dummy(batch=2, T=81, L=7, device=device,
                       float_dtype=torch.float32, clip_embed=False)
    model = load_model(model_path)
    model.eval().to(device)

    input_names = [
        "tokens", "token_mask",
        "lora_ids", "lora_w",
        "cfg", "n_loras",
        "sampler_id", "steps_log", "steps_bucket",
        "upscaler_id", "up_has", "up_steps", "denoise", "model_id"
    ]
    output_names = ["output"]

    # Use legacy dynamic_axes when dynamo=False
    dyn_axes = {n: {0: "batch"} for n in input_names}
    # dyn_axes["token_mask"] = {1: "seq_len"}
    # dyn_axes["tokens"][1] = "seq_len"
    # dyn_axes["token_mask"][1] = "seq_len"
    # dyn_axes["lora_ids"][1] = "n_loras"
    # dyn_axes["lora_w"][1] = "n_loras"
    # Model outputs logits shaped [B], so mark dim 0 as batch
    dyn_axes["output"] = {0: "batch"}

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            out_path,
            opset_version=18,
            input_names=input_names,
            output_names=output_names,
            # dynamic_shapes=dyn,
            dynamic_axes=dyn_axes,

            do_constant_folding=True,
            export_params=True,
            keep_initializers_as_inputs=False,
            dynamo=True
        )
    print(f"✔ ONNX (FP32) saved to {out_path}")


def convert_onnx_to_fp16(in_path, out_path=None, keep_io_types=False):
    try:
        import onnx
        from onnxconverter_common import float16
    except Exception as e:
        raise SystemExit(
            "This step requires 'onnx' and 'onnxconverter-common'. Install with:\n"
            "  pip install onnx onnxconverter-common\n"
            f"Import error: {e}"
        )
    model = onnx.load(in_path)
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
    )
    if out_path is None:
        base, ext = os.path.splitext(in_path)
        out_path = base + "_fp16" + ext
    onnx.save(model_fp16, out_path)
    print(f"✔ ONNX (FP16) saved to {out_path}")
    return out_path


def fix_cast_attr_to_match_output(in_path, out_path=None):
    """
    After FP16 conversion, make every Cast node's `to` attribute match the
    actually inferred dtype of its output (e.g., set to FLOAT16 if the graph
    now treats that tensor as fp16). Also re-infer shapes/types and check.
    """
    import os
    import onnx
    from onnx import TensorProto, shape_inference, checker

    model = onnx.load(in_path)

    # Infer shapes/types so we can read the current dtypes of tensors
    model = shape_inference.infer_shapes(model)
    g = model.graph

    # Build a map: value_name -> inferred elem_type
    dtype = {}

    def add_vi(vi):
        tt = vi.type.tensor_type
        if tt.elem_type != 0:
            dtype[vi.name] = tt.elem_type

    for vi in list(g.input):
        add_vi(vi)
    for vi in list(g.output):
        add_vi(vi)
    for vi in list(g.value_info):
        add_vi(vi)

    changed = False
    for n in g.node:
        if n.op_type != "Cast" or not n.output:
            continue
        out_name = n.output[0]
        # Find the Cast's current `to` attr
        to_attr = next((a for a in n.attribute if a.name == "to"), None)
        if to_attr is None:
            continue

        # What dtype does the graph currently think this output is?
        inferred = dtype.get(out_name, None)
        if inferred is None:
            # If not inferred, try a heuristic: if the Cast input is BOOL and
            # the model was globally converted to fp16, prefer FLOAT16.
            in_name = n.input[0] if n.input else None
            in_dtype = dtype.get(in_name, None)
            if in_dtype == TensorProto.BOOL:
                inferred = TensorProto.FLOAT16

        if inferred is not None and to_attr.i != inferred:
            to_attr.i = int(inferred)
            changed = True

    if changed:
        model = shape_inference.infer_shapes(model)
        checker.check_model(model)

    onnx.save(model, in_path)
    print(f"✔ Cast .to attributes aligned with inferred dtypes → {in_path}")
    return in_path


def keep_selected_inputs_fp32(in_path, out_path=None, input_names=None):
    """
    Keep selected inputs as FP32 at the graph boundary and insert Cast(to=FLOAT16)     right before their first consumer so internals remain FP16.
    """
    import os
    import onnx
    from onnx import helper, TensorProto, shape_inference, checker

    if not input_names:
        return in_path

    m = onnx.load(in_path)
    g = m.graph

    # Track existing names to avoid collisions
    existing = {vi.name for vi in list(
        g.input)+list(g.output)+list(g.value_info)}
    for n in g.node:
        existing.update(n.input)
        existing.update(n.output)

    def unique_name(base):
        name = base
        i = 1
        while name in existing:
            i += 1
            name = f"{base}_{i}"
        existing.add(name)
        return name

    # map for graph inputs
    name_to_vi = {vi.name: vi for vi in g.input}

    for keep in (input_names or []):
        vi = name_to_vi.get(keep)
        if vi is None:
            print(f"… skip '{keep}': not a graph input")
            continue

        # enly float inputs are relevant
        elem = vi.type.tensor_type.elem_type
        if elem not in (TensorProto.FLOAT, TensorProto.FLOAT16):
            continue

        # Ensure API boundary is FP32
        vi.type.tensor_type.elem_type = TensorProto.FLOAT

        # eTake a snapshot of current nodes to find consumers BEFORE inserting anything
        original_nodes = list(g.node)

        consumers = []
        for idx, node in enumerate(original_nodes):
            for inp_idx, inp in enumerate(node.input):
                if inp == keep:
                    consumers.append((idx, inp_idx))

        # If no consumers, still insert a Cast at the very start (harmless)
        cast_out = unique_name(f"{keep}_fp16in")
        cast_node = helper.make_node(
            "Cast",
            inputs=[keep],
            outputs=[cast_out],
            name=f"Cast_In_{keep}",
            to=TensorProto.FLOAT16,
        )

        insert_idx = min((i for i, _ in consumers), default=0)
        g.node.insert(insert_idx, cast_node)

        # eewire ONLY the original consumer nodes (the snapshot), not the newly inserted Cast
        for node, (node_idx, inp_idx) in zip(
            (original_nodes[i] for i, _ in consumers), consumers
        ):
            node.input[inp_idx] = cast_out

    # eeinfer and validate
    m = shape_inference.infer_shapes(m)
    checker.check_model(m)

    onnx.save(m, in_path)
    print(
        f"✔ Kept selected inputs FP32 and inserted Cast→FP16 near consumers → {in_path}")
    return in_path


def convert_v3(out_path, model_path, device="cuda", fp16=False, keep_io_types=False,
               target_outputs=None):
    export_onnx_fp32(out_path, model_path, device=device)

    if not fp16:
        return

    fp16_path = convert_onnx_to_fp16(out_path, None, keep_io_types=False)

    if keep_io_types:
        fp16_path = keep_selected_inputs_fp32(fp16_path, None, input_names=[
            "cfg", "n_loras", "steps_log", "up_has", "up_steps", "denoise", "token_mask", "lora_w", "model_id"])

    fp16_path = fix_cast_attr_to_match_output(fp16_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PyTorch model to ONNX format")
    parser.add_argument("--model_path", type=str,
                        default="best_model.pt", help="(unused by v3)")
    parser.add_argument("--output_path", type=str,
                        default="best_model.onnx", help="Path for ONNX model output")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--fp16", action="store_true",
                        help="After export, convert the ONNX to FP16")
    parser.add_argument("--keep_io_types", action="store_true",
                        help="Keep model I/O as FP32 while internals are FP16 (adds final Cast automatically)")
    parser.add_argument("--target_outputs", type=str, nargs="*",
                        help="Optional list of output names to enforce FP32 Cast on (defaults to all outputs)")
    args = parser.parse_args()

    convert_v3(
        args.output_path,
        args.model_path,
        device=args.device,
        fp16=args.fp16,
        keep_io_types=args.keep_io_types,
        target_outputs=args.target_outputs,
    )
