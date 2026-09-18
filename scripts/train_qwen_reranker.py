from __future__ import annotations

import argparse
from importlib.metadata import version
import math
from pathlib import Path
import platform
import random
import tempfile
import time

import torch
import torch.nn.functional as functional
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from neural_data import INSTRUCTION, MODEL, REVISION, digest, dump, load, tokenize_groups, verify_dataset


def collate_tokens(tokenizer, groups: list[dict], device):
    sequences = [sequence for group in groups for sequence in group["tokens"]]
    batch = tokenizer.pad({"input_ids": sequences, "attention_mask": [[1] * len(row) for row in sequences]},
                          padding=True, return_tensors="pt")
    return {name: values.to(device) for name, values in batch.items()}


def score_tokens(model, tokenizer, groups: list[dict], device, yes_id: int, no_id: int):
    batch = collate_tokens(tokenizer, groups, device)
    logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
    scores = logits[:, yes_id] - logits[:, no_id]
    return list(scores.split([len(group["tokens"]) for group in groups]))


def ranking_loss(scores: list[torch.Tensor]):
    return torch.stack([functional.cross_entropy(row.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=row.device))
                        for row in scores]).mean()


def metrics_from_scores(scores: list[list[float]]) -> dict:
    if not scores or any(not all(math.isfinite(value) for value in row) for row in scores):
        raise ValueError("Empty or nonfinite scores")
    ranks = [1 + sum(value >= row[0] for value in row[1:]) for row in scores]
    return {"groups": len(ranks), "silver_top1": sum(rank == 1 for rank in ranks) / len(ranks),
            "silver_mrr": sum(1 / rank for rank in ranks) / len(ranks),
            "tie_policy": "tied negatives precede the silver positive"}


@torch.no_grad()
def evaluate(model, tokenizer, groups, device, yes_id, no_id):
    was_training = model.training
    model.eval()
    scores, losses = [], []
    for group in groups:
        values = score_tokens(model, tokenizer, [group], device, yes_id, no_id)
        losses.append(float(ranking_loss(values)))
        scores.append(values[0].cpu().tolist())
    model.train(was_training)
    return {**metrics_from_scores(scores), "mean_listwise_loss": sum(losses) / len(losses)}


def optimizer_parameters(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def save_checkpoint(path, model, tokenizer, optimizer, scheduler, state, config):
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as temporary:
        staged = Path(temporary)
        model.save_pretrained(staged)
        tokenizer.save_pretrained(staged)
        torch.save({
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }, staged / "training_state.pt")
        dump(staged / "run_config.json", config)
        dump(staged / "progress.json", state)
        staged.rename(path)


def restore_training(path, optimizer, scheduler):
    checkpoint = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    torch.set_rng_state(checkpoint["torch_rng"])
    random.setstate(checkpoint["python_rng"])
    if checkpoint["cuda_rng"]:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])


def train_loop(model, tokenizer, groups, dev, optimizer, scheduler, config, output, yes_id, no_id,
               resume=None, stop_after_steps=None, save_steps=100):
    device = next(model.parameters()).device
    state = {"epoch": 0, "next_group": 0, "step": 0}
    if resume is not None:
        state = load(resume / "progress.json")
        restore_training(resume, optimizer, scheduler)
    if stop_after_steps is not None and state["step"] >= stop_after_steps:
        return state
    step_durations = []
    model.train()
    effective_groups = config["batch_groups"] * config["gradient_accumulation"]
    for epoch in range(state["epoch"], config["epochs"]):
        order = torch.randperm(len(groups), generator=torch.Generator().manual_seed(config["seed"] + epoch)).tolist()
        first = state["next_group"] if epoch == state["epoch"] else 0
        for start in range(first, len(order), effective_groups):
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            selected = [groups[index] for index in order[start:start + effective_groups]]
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for micro_start in range(0, len(selected), config["batch_groups"]):
                micro = selected[micro_start:micro_start + config["batch_groups"]]
                scores = score_tokens(model, tokenizer, micro, device, yes_id, no_id)
                loss = ranking_loss(scores) * (len(micro) / len(selected))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                total_loss += loss.detach().item()
            grad_norm = torch.nn.utils.clip_grad_norm_(optimizer_parameters(model), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            duration = time.perf_counter() - started
            step_durations.append(duration)
            next_group = min(start + len(selected), len(order))
            state = {"epoch": epoch + int(next_group == len(order)),
                     "next_group": 0 if next_group == len(order) else next_group, "step": state["step"] + 1}
            if state["step"] % 10 == 0 or len(step_durations) == 1:
                print(f"step={state['step']} loss={total_loss:.5f} grad={float(grad_norm):.4f} seconds={duration:.3f}", flush=True)
            stopping = stop_after_steps is not None and state["step"] >= stop_after_steps
            if state["step"] % save_steps == 0 or stopping or next_group == len(order):
                checkpoint = output / f"checkpoint-{state['step']:06d}"
                save_checkpoint(checkpoint, model, tokenizer, optimizer, scheduler, state, config)
                dump(output / "latest.json", {"checkpoint": checkpoint.name})
                measured = step_durations[2:] or step_durations
                mean_seconds = sum(measured) / len(measured)
                dump(output / "throughput.json", {
                    "measured_steps_this_invocation": len(step_durations), "mean_seconds_per_optimizer_step": mean_seconds,
                    "planned_optimizer_steps": config["total_steps"],
                    "projected_training_hours_excluding_eval_io": mean_seconds * config["total_steps"] / 3600,
                    "remaining_training_hours_excluding_eval_io": mean_seconds * (config["total_steps"] - state["step"]) / 3600,
                    "device": str(device), "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else None,
                    "estimate_warning": "observed throughput, not a guarantee; CPU smoke cannot estimate H100 speed",
                })
            if stopping:
                return state
        metrics = evaluate(model, tokenizer, dev, device, yes_id, no_id)
        dump(output / f"epoch-{epoch + 1}-development.json", metrics)
        print(f"epoch={epoch + 1} development={metrics}", flush=True)
    dump(output / "training_complete.json", {"status": "complete", **state})
    return state


def main():
    parser = argparse.ArgumentParser(description="Fixed-query Qwen reranker LoRA with optimizer/RNG resume")
    parser.add_argument("--data", type=Path, required=True,
                        help="Private schema-v2 candidate directory; see docs/DATA_SCHEMA.md")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--negatives", type=int, default=3)
    parser.add_argument("--batch-groups", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--eval-groups", type=int, default=256)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.negatives, args.batch_groups, args.gradient_accumulation,
           args.lora_rank, args.eval_groups, args.save_steps, args.max_length) < 1 or args.learning_rate <= 0:
        raise ValueError("Training options must be positive")
    if args.stop_after_steps is not None and args.stop_after_steps < 1:
        raise ValueError("stop-after-steps must be positive")
    if args.model != MODEL and not args.revision:
        raise ValueError("Alternative models require an explicit --revision commit hash")
    revision = args.revision or REVISION
    manifest = verify_dataset(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, padding_side="left", trust_remote_code=False)
    yes_id, no_id = [tokenizer.convert_tokens_to_ids(token) for token in ("yes", "no")]
    if tokenizer.encode("yes", add_special_tokens=False) != [yes_id] or tokenizer.encode("no", add_special_tokens=False) != [no_id]:
        raise ValueError("Expected single-token yes/no labels")
    train, train_stats = tokenize_groups(tokenizer, args.data / "train.jsonl", args.max_length, args.negatives)
    dev, dev_stats = tokenize_groups(tokenizer, args.data / "dev.jsonl", args.max_length)
    random.Random(args.seed).shuffle(dev)
    dev = dev[:args.eval_groups]
    config = {
        "schema_version": 2, "model": args.model, "revision": revision, "instruction": INSTRUCTION,
        "data_manifest_sha256": digest(args.data / "manifest.json"), "epochs": args.epochs,
        "max_length": args.max_length, "negatives": args.negatives, "batch_groups": args.batch_groups,
        "gradient_accumulation": args.gradient_accumulation, "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank, "seed": args.seed, "eval_groups": args.eval_groups,
        "train_groups": len(train), "dev_query_ids": [row["query_id"] for row in dev],
        "total_steps": args.epochs * math.ceil(len(train) / (args.batch_groups * args.gradient_accumulation)),
        "code_sha256": {name: digest(Path(__file__).parent / name) for name in ("neural_data.py", "train_qwen_reranker.py")},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                    **{name: version(name) for name in ("transformers", "peft", "accelerate", "tokenizers")}},
    }
    output = args.output.resolve()
    if args.resume:
        if load(args.resume / "run_config.json") != config or load(output / "run_config.json") != config:
            raise ValueError("Resume configuration/data/code differs; use exactly the original settings")
        if not (args.resume / "training_state.pt").exists():
            raise ValueError("Incomplete checkpoint")
    elif output.exists():
        raise FileExistsError("Output exists. Use --resume or a new output directory.")
    else:
        output.mkdir(parents=True)
        dump(output / "run_config.json", config)
        dump(output / "token_profile.json", {"train": train_stats, "dev": dev_stats, "manifest": manifest,
                                            "selected_dev_groups": len(dev)})
    print(f"train={len(train)} dev_selected={len(dev)} updates={config['total_steps']}; zero truncation", flush=True)
    if args.profile_only:
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Real-model training requires a BF16-capable CUDA GPU; profile-only works on CPU")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, torch_dtype=torch.bfloat16,
                                               attn_implementation="sdpa", trust_remote_code=False)
    model.to("cuda:0")
    model.config.use_cache = False
    if args.resume:
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", r=args.lora_rank, lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05, bias="none", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(optimizer_parameters(model), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(config["total_steps"] * 0.05), config["total_steps"])
    if not args.resume:
        baseline = evaluate(model, tokenizer, dev, torch.device("cuda:0"), yes_id, no_id)
        dump(output / "frozen_development.json", baseline)
        print(f"Frozen baseline: {baseline}", flush=True)
    train_loop(model, tokenizer, train, dev, optimizer, scheduler, config, output, yes_id, no_id,
               args.resume, args.stop_after_steps, args.save_steps)


if __name__ == "__main__":
    main()
