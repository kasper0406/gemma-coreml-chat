"""Prompt generation for benchmarking — realistic and synthetic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BenchmarkPrompt:
    """A prompt ready for benchmarking."""

    name: str
    token_ids: list[int]
    description: str

    @property
    def length(self) -> int:
        return len(self.token_ids)


# ---------------------------------------------------------------------------
# Realistic prompt templates (user turn only — template wraps them)
# ---------------------------------------------------------------------------

_SHORT_PROMPT = "What is the capital of France?"

_MEDIUM_PROMPT = (
    "Explain the differences between classical and operant conditioning in "
    "psychology. Include examples of each, and discuss how these learning "
    "theories have been applied in educational settings. Be thorough but "
    "concise in your response."
)

_LONG_PROMPT = (
    "You are a helpful research assistant. I need you to analyze the "
    "following passage and answer detailed questions about it.\n\n"
    + "The history of artificial intelligence began in antiquity, with myths, "
    "stories and rumors of artificial beings endowed with intelligence or "
    "consciousness by master craftsmen. The seeds of modern AI were planted "
    "by philosophers who attempted to describe the process of human thinking "
    "as the mechanical manipulation of symbols. This work culminated in the "
    "invention of the programmable digital computer in the 1940s, a machine "
    "based on the abstract essence of mathematical reasoning. This device and "
    "the ideas behind it inspired a handful of scientists to begin seriously "
    "discussing the possibility of building an electronic brain. The field of "
    "AI research was founded at a workshop held on the campus of Dartmouth "
    "College during the summer of 1956. Those who attended would become the "
    "leaders of AI research for decades. Many of them predicted that a machine "
    "as intelligent as a human being would exist in no more than a generation, "
    "and they were given millions of dollars to make this vision come true. "
    "Eventually, it became obvious that commercial developers and researchers "
    "had grossly underestimated the difficulty of the project. In 1974, in "
    "response to the criticism from James Lighthill and ongoing pressure from "
    "congress, the U.S. and British governments cut off exploratory research "
    "in AI. The next few years would later be called an AI winter, a period "
    "when obtaining funding for AI projects was difficult. In the early 1980s, "
    "AI research was revived by the commercial success of expert systems, a "
    "form of AI program that simulated the knowledge and analytical skills of "
    "human experts. By 1985, the market for AI had reached over a billion "
    "dollars. At the same time, Japan's fifth generation computer project "
    "inspired the U.S and British governments to restore funding for academic "
    "research. However, beginning with the collapse of the Lisp Machine "
    "market in 1987, AI once again fell into disrepute, and a second, longer "
    "lasting winter began. " * 3  # repeat to get a longer passage
    + "\n\nBased on the passage above:\n"
    "1. What event founded the field of AI research?\n"
    "2. What caused the first AI winter?\n"
    "3. What revived AI research in the 1980s?\n"
    "4. Summarize the cyclical pattern of AI funding described in the text."
)

_REALISTIC_PROMPTS = [
    ("short_qa", _SHORT_PROMPT, "Short Q&A (~50 tokens)"),
    ("medium_explanation", _MEDIUM_PROMPT, "Medium explanation request (~100 tokens)"),
    ("long_comprehension", _LONG_PROMPT, "Long reading comprehension (~1000+ tokens)"),
]


def _tokenize_chat(
    tokenizer, user_text: str, max_length: int | None = None,
) -> list[int]:
    """Apply chat template and tokenize.  Optionally truncate to max_length."""
    messages = [{"role": "user", "content": user_text}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if max_length is not None:
        ids = ids[:max_length]
    return ids


def realistic_prompts(tokenizer) -> list[BenchmarkPrompt]:
    """Return realistic prompts tokenized with the chat template."""
    prompts = []
    for name, text, desc in _REALISTIC_PROMPTS:
        ids = _tokenize_chat(tokenizer, text)
        prompts.append(BenchmarkPrompt(
            name=name,
            token_ids=ids,
            description=f"{desc} ({len(ids)} tokens)",
        ))
    return prompts


def synthetic_prompt(
    tokenizer, target_length: int, *, filler_token: int | None = None,
) -> BenchmarkPrompt:
    """Build a synthetic prompt padded to exactly ``target_length`` tokens.

    Uses the chat template for a short user message, then fills the remaining
    space with a repeating filler token so the total length hits the target.
    """
    # Start with a minimal user message to get the template overhead.
    base_ids = _tokenize_chat(tokenizer, "Repeat after me:")
    if filler_token is None:
        # Use a common word token (e.g. "hello")
        hello_ids = tokenizer.encode("hello", add_special_tokens=False)
        filler_token = hello_ids[0] if hello_ids else 1

    if len(base_ids) >= target_length:
        ids = base_ids[:target_length]
    else:
        fill_count = target_length - len(base_ids)
        ids = base_ids + [filler_token] * fill_count

    return BenchmarkPrompt(
        name=f"synthetic_{target_length}",
        token_ids=ids,
        description=f"Synthetic prompt padded to {target_length} tokens",
    )


DEFAULT_CONTEXT_LENGTHS = [
    128, 256, 512, 1024, 2048, 4096,
    8192, 16384, 32768, 49152, 61440,
]


def synthetic_prompts(
    tokenizer, context_lengths: list[int] | None = None,
) -> list[BenchmarkPrompt]:
    """Return synthetic prompts at the requested context lengths."""
    if context_lengths is None:
        context_lengths = DEFAULT_CONTEXT_LENGTHS
    return [synthetic_prompt(tokenizer, n) for n in context_lengths]
