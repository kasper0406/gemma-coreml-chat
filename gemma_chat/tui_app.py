"""Textual chat UI: history, input, streaming tokens, context/generation stats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Input, RichLog, Static

from gemma_chat.chat import HELP_TEXT, format_chat_prompt
from gemma_chat.config import MAX_SEQ_LEN
from gemma_chat.generate import generate_kvcached, stop_token_ids, truncate_prompt_ids
from gemma_chat.jax_generate import generate_jax_stream


@dataclass
class ChatRuntimeConfig:
    backend: Literal["coreml", "jax"]
    tokenizer: Any
    max_new_tokens: int
    temperature: float
    top_p: float
    system_prompt: str | None
    model_path: str | None = None
    decode_only: bool = False
    decode_model: Any | None = None
    prefill_model: Any | None = None
    jax_params: dict[str, Any] | None = None
    jax_cfg: Any | None = None


class GemmaChatApp(App[None]):
    CSS = """
    Screen {
        background: $surface;
    }
    #top_row {
        layout: horizontal;
        height: 1;
        margin: 0 1;
    }
    #title {
        width: 1fr;
    }
    #stats {
        width: auto;
        min-width: 32;
        color: $text-muted;
        content-align: right middle;
    }
    #log {
        border: tall $primary;
        height: 1fr;
        margin: 0 1;
    }
    #stream_row {
        height: auto;
        min-height: 1;
        margin: 0 1;
        color: $accent;
    }
    #message_input {
        margin: 0 1 1 1;
    }
    """

    BINDINGS = [("ctrl+c", "quit", "Quit")]

    def __init__(self, cfg: ChatRuntimeConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self._history: list[dict[str, str]] = []
        self._generating = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="top_row"):
            yield Static("Gemma4-E2B Chat  [dim]/help[/]", id="title")
            yield Static("", id="stats")
        yield RichLog(id="log", wrap=True, highlight=False, markup=True)
        yield Static("", id="stream_row")
        yield Input(placeholder="Message…", id="message_input")

    def on_mount(self) -> None:
        self.query_one("#message_input", Input).focus()
        if self._cfg.system_prompt:
            self.query_one("#log", RichLog).write(
                f"[dim]System:[/] {self._cfg.system_prompt}"
            )
        self._set_stats(0, 0, False, 0, 0)

    def _set_stats(
        self,
        fed_len: int,
        n_gen: int,
        truncated: bool,
        raw_len: int,
        max_seq: int,
    ) -> None:
        c = self._cfg
        trunc = f" [yellow]trunc←{raw_len}[/]" if truncated else ""
        total = fed_len + n_gen
        line = (
            f"ctx [bold]{fed_len}[/]/{max_seq}{trunc}  "
            f"gen [bold]{n_gen}[/]/{c.max_new_tokens}  "
            f"seq [bold]{total}[/]/{max_seq}"
        )
        self.query_one("#stats", Static).update(line)

    def action_quit(self) -> None:
        self.exit()

    @on(Input.Submitted)
    def handle_submit(self, event: Input.Submitted) -> None:
        if event.input.id != "message_input":
            return
        if self._generating:
            return
        text = event.value.strip()
        event.input.clear()
        if not text:
            return

        low = text.lower()
        if text.startswith("/"):
            if low in ("/quit", "/exit"):
                self.exit()
                return
            if low == "/reset":
                self._history.clear()
                self.query_one("#log", RichLog).write("[dim][History cleared][/]")
                self.query_one("#stream_row", Static).update("")
                self._set_stats(0, 0, False, 0, MAX_SEQ_LEN)
                return
            if low == "/help":
                self.query_one("#log", RichLog).write(f"[dim]{HELP_TEXT}[/]")
                return
            self.query_one("#log", RichLog).write(
                f"[red]Unknown command[/] [bold]{text}[/]  (try /help)"
            )
            return

        self._history.append({"role": "user", "content": text})
        self.query_one("#log", RichLog).write(f"[bold green]You:[/] {text}")
        self.query_one("#stream_row", Static).update("")

        self._generating = True
        event.input.disabled = True
        self.run_worker(
            lambda: self._generate_turn(text),
            name="generate",
            thread=True,
            exclusive=True,
        )

    def _generate_turn(self, _user_text: str) -> None:
        cfg = self._cfg
        tok = cfg.tokenizer
        prompt = format_chat_prompt(
            self._history,
            tokenizer=tok,
            system_prompt=cfg.system_prompt,
        )
        prompt_ids = tok.encode(prompt, add_special_tokens=False)
        raw_len = len(prompt_ids)
        fed_ids = truncate_prompt_ids(
            prompt_ids,
            MAX_SEQ_LEN,
            reserve_for_generation=cfg.max_new_tokens,
        )
        fed_len = len(fed_ids)
        truncated = fed_len < raw_len
        stop_ids = stop_token_ids(tok)
        pad_id = tok.pad_token_id or 0

        def ui(callable_thunk) -> None:
            self.call_from_thread(callable_thunk)

        try:
            if cfg.backend == "coreml":
                assert cfg.model_path is not None
                it = generate_kvcached(
                    fed_ids,
                    model_path=cfg.model_path,
                    decode_model=cfg.decode_model,
                    prefill_model=cfg.prefill_model,
                    decode_only=cfg.decode_only,
                    pad_id=pad_id,
                    stop_ids=stop_ids,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_seq_len=MAX_SEQ_LEN,
                    verbose=False,
                )
            else:
                assert cfg.jax_params is not None and cfg.jax_cfg is not None
                it = generate_jax_stream(
                    fed_ids,
                    cfg.jax_params,
                    cfg.jax_cfg,
                    tok,
                    cfg.max_new_tokens,
                    cfg.temperature,
                    cfg.top_p,
                    MAX_SEQ_LEN,
                )

            gen_ids: list[int] = []
            n_gen = 0
            for tid in it:
                if tid in stop_ids:
                    break
                gen_ids.append(tid)
                n_gen = len(gen_ids)
                decoded = tok.decode(gen_ids, skip_special_tokens=True)

                def upd(
                    d: str = decoded,
                    fl: int = fed_len,
                    ng: int = n_gen,
                    tr: bool = truncated,
                    rl: int = raw_len,
                ) -> None:
                    self.query_one("#stream_row", Static).update(f"[bold magenta]Gemma:[/] {d}")
                    self._set_stats(fl, ng, tr, rl, MAX_SEQ_LEN)

                ui(upd)

            reply = tok.decode(gen_ids, skip_special_tokens=True)

            def finish(
                r: str = reply,
                fl: int = fed_len,
                ng: int = n_gen,
                tr: bool = truncated,
                rl: int = raw_len,
            ) -> None:
                self.query_one("#stream_row", Static).update("")
                self.query_one("#log", RichLog).write(f"[bold magenta]Gemma:[/] {r}")
                self._set_stats(fl, ng, tr, rl, MAX_SEQ_LEN)

            ui(finish)
            self._history.append({"role": "model", "content": reply})
        except Exception as exc:

            def err(e: BaseException = exc) -> None:
                self.query_one("#stream_row", Static).update("")
                self.query_one("#log", RichLog).write(f"[red]Error:[/] {e}")
                if self._history and self._history[-1].get("role") == "user":
                    self._history.pop()

            ui(err)
        finally:

            def reenable() -> None:
                self._generating = False
                inp = self.query_one("#message_input", Input)
                inp.disabled = False
                inp.focus()

            ui(reenable)


def run_chat_tui(cfg: ChatRuntimeConfig) -> None:
    GemmaChatApp(cfg).run()
