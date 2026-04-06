"""A/B comparison: full-sequence .mlpackage vs KV decode path.

Runs the same tokenized prompt through both stacks (default: greedy decoding
so differences are easier to interpret). Use this to see whether regressions
come from the decode export vs the full model.

Example::

    uv run gemma-compare-inference \\
        --full-model gemma4-e2b-v40.mlpackage \\
        --decode-model gemma4-e2b-decode.mlpackage \\
        --prompt "Hi there"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gemma_chat.config import HF_MODEL_ID, MLPACKAGE_PATH, MAX_SEQ_LEN
from gemma_chat.generate import (
    generate_text,
    generate_text_kvcached,
    load_coreml_model,
    load_tokenizer,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare text generation: full-sequence CoreML model vs "
            "KV-cached decode model (same tokenizer)."
        )
    )
    parser.add_argument(
        "--full-model",
        type=Path,
        default=None,
        help=f"Full forward .mlpackage (default: {MLPACKAGE_PATH})",
    )
    parser.add_argument(
        "--decode-model",
        type=Path,
        required=True,
        help="Decode-step .mlpackage from gemma-export-decode",
    )
    parser.add_argument(
        "--prompt",
        default="Hi there, who are you?",
        help="Test prompt string",
    )
    parser.add_argument(
        "--model-id",
        default=HF_MODEL_ID,
        help=f"Tokenizer HF id (default: {HF_MODEL_ID})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=48,
        help="Max new tokens per path (default: 48)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 = greedy (default); >0 for sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p when temperature > 0 (default: 0.9)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=MAX_SEQ_LEN,
        help=f"Max sequence length (default: {MAX_SEQ_LEN})",
    )
    args = parser.parse_args()

    full_path = args.full_model or Path(MLPACKAGE_PATH)
    if not full_path.exists():
        print(f"Full model not found: {full_path}", file=sys.stderr)
        sys.exit(1)
    if not args.decode_model.exists():
        print(f"Decode model not found: {args.decode_model}", file=sys.stderr)
        sys.exit(1)

    tokenizer = load_tokenizer(args.model_id)
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)

    print("Loading full-sequence model…", flush=True)
    full_m = load_coreml_model(full_path)
    print("Loading decode model…", flush=True)
    decode_m = load_coreml_model(args.decode_model)

    print("\n--- Full-sequence model ---\n", flush=True)
    text_full = generate_text(
        args.prompt,
        cml_model=full_m,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_seq_len=args.max_seq_len,
    )
    print(text_full)

    print("\n--- KV decode model (slow prefill) ---\n", flush=True)
    text_kv = generate_text_kvcached(
        args.prompt,
        tokenizer=tokenizer,
        decode_path=None,
        decode_model=decode_m,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_seq_len=args.max_seq_len,
    )
    print(text_kv)
    print(f"\n(prompt tokens: {len(prompt_ids)})", flush=True)


if __name__ == "__main__":
    main()
