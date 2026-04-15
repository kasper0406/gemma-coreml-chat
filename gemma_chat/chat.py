"""Terminal chat for Gemma4-E2B (Textual UI): CoreML or JAX reference.

Usage:
    uv run gemma-chat --model gemma4-e2b.mlpackage
    uv run gemma-chat --model gemma4-e2b.mlpackage --decode-only
    uv run gemma-chat --backend jax
    uv run python -m gemma_chat.chat --help

Commands during chat:
    /quit  or  /exit   — exit the app
    /reset             — clear conversation history
"""

from __future__ import annotations

import argparse
import sys

import coremltools as ct

from gemma_chat.config import E2B_CONFIG, HF_MODEL_ID
from gemma_chat.generate import load_coreml_model, load_tokenizer
from gemma_chat.weight_mapper import load_params


# Gemma4 uses <|turn> / <turn|> markers. Use the tokenizer's apply_chat_template.


def format_chat_prompt(
    history: list[dict[str, str]],
    tokenizer,
    system_prompt: str | None = None,
) -> str:
    """Render conversation history using the tokenizer's chat template.

    Each entry in history is {"role": "user"|"assistant"|"model", "content": "..."}.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for turn in history:
        role = "assistant" if turn["role"] == "model" else turn["role"]
        messages.append({"role": role, "content": turn["content"]})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


HELP_TEXT = """
Commands:
  /reset    — clear conversation history
  /quit     — exit
  /help     — show this message
"""


def parse_compute_units(value: str) -> ct.ComputeUnit:
        mapping = {
                "all": ct.ComputeUnit.ALL,
                "cpu-only": ct.ComputeUnit.CPU_ONLY,
        }
        return mapping[value]


def main() -> None:
    from gemma_chat.tui_app import ChatRuntimeConfig, run_chat_tui

    parser = argparse.ArgumentParser(
        description="Chat with Gemma4-E2B (CoreML or JAX)."
    )
    parser.add_argument(
        "--backend",
        choices=("coreml", "jax"),
        default="coreml",
        help="Inference backend (default: coreml)",
    )
    parser.add_argument(
        "--model",
        default="gemma4-e2b.mlpackage",
        help=(
            "Path to multifunction .mlpackage, or a directory containing "
            "decode.mlpackage + prefill.mlpackage (default: gemma4-e2b.mlpackage)"
        ),
    )
    parser.add_argument(
        "--compute-units",
        choices=("all", "cpu-only"),
        default="all",
        help=(
            "CoreML compute target: all for best runtime performance, "
            "cpu-only for much faster first-load compilation"
        ),
    )
    parser.add_argument(
        "--decode-only",
        action="store_true",
        help="Skip batched prefill; use token-by-token decode only (slower but less RAM)",
    )
    parser.add_argument(
        "--model-id",
        default=HF_MODEL_ID,
        help=f"HuggingFace model ID for tokenizer / JAX weights (default: {HF_MODEL_ID})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per turn (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (0 = greedy, default: 1.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling probability (default: 0.9)",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Optional system prompt",
    )
    args = parser.parse_args()

    if args.backend == "coreml":
        compute_units = parse_compute_units(args.compute_units)
        print("Loading tokenizer…", flush=True)
        tokenizer = load_tokenizer(args.model_id)

        # Pre-load decode model (always needed); prefill loaded only if not --decode-only.
        print(
            f"Loading CoreML decode function from {args.model!r} "
            f"(compute_units={args.compute_units})…",
            flush=True,
        )
        decode_model = load_coreml_model(
            args.model,
            compute_units=compute_units,
            function_name="decode",
        )
        prefill_model = None
        if not args.decode_only:
            print(
                f"Loading CoreML prefill function from {args.model!r} "
                f"(compute_units={args.compute_units})…",
                flush=True,
            )
            prefill_model = load_coreml_model(
                args.model,
                compute_units=compute_units,
                function_name="prefill",
            )

        cfg = ChatRuntimeConfig(
            backend="coreml",
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            system_prompt=args.system,
            model_path=args.model,
            decode_only=args.decode_only,
            decode_model=decode_model,
            prefill_model=prefill_model,
        )
    else:
        jax_cfg = E2B_CONFIG
        print("Loading tokenizer…", flush=True)
        tokenizer = load_tokenizer(args.model_id)
        print(
            f"Loading JAX weights ({args.model_id})…",
            flush=True,
        )
        params = load_params(model_id=args.model_id, config=jax_cfg)
        cfg = ChatRuntimeConfig(
            backend="jax",
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            system_prompt=args.system,
            jax_params=params,
            jax_cfg=jax_cfg,
        )

    try:
        run_chat_tui(cfg)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
