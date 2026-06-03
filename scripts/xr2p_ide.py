#!/usr/bin/env python3
"""Small Tkinter IDE for translating R scripts to Python with xr2p.py."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XR2P = ROOT / "xr2p.py"
R_FILETYPES = [("R files", "*.r *.R"), ("All files", "*.*")]
PY_FILETYPES = [("Python files", "*.py"), ("All files", "*.*")]
R_KEYWORDS = {
    "break", "else", "FALSE", "for", "function", "if", "in", "NA", "NaN",
    "next", "NULL", "repeat", "return", "TRUE", "while",
}
R_BUILTINS = {
    "c", "cat", "data.frame", "factor", "length", "list", "matrix", "mean",
    "print", "rnorm", "sd", "seq", "sum", "var",
}
PY_KEYWORDS = {
    "False", "None", "True", "and", "as", "break", "class", "continue",
    "def", "elif", "else", "except", "finally", "for", "from", "if",
    "import", "in", "is", "lambda", "not", "or", "pass", "raise", "return",
    "try", "while", "with",
}
STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')
NUMBER_RE = re.compile(r"\b\d+(?:\.\d*)?(?:[eE][+-]?\d+)?\b|\.\d+(?:[eE][+-]?\d+)?\b")


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    elapsed: float


@dataclass
class IdeResult:
    command_result: CommandResult
    python: str
    out_path: Path | None
    run_mode: str | None
    r_result: CommandResult | None = None
    py_result: CommandResult | None = None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


def run_command(command: list[str], *, cwd: Path, timeout: float | None) -> CommandResult:
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(command, completed.returncode, completed.stdout or "", completed.stderr or "", time.perf_counter() - start)
    except FileNotFoundError as exc:
        return CommandResult(command, 127, "", f"{exc.filename!r} was not found", time.perf_counter() - start)
    except subprocess.TimeoutExpired as exc:
        return CommandResult(command, 124, exc.stdout or "", f"timed out after {exc.timeout} seconds", time.perf_counter() - start)


def split_run_both_output(stdout: str) -> tuple[str, str]:
    lines = stdout.splitlines()
    r_lines: list[str] = []
    py_lines: list[str] = []
    section: str | None = None
    for line in lines:
        if line.startswith("Run (R):"):
            section = "r"
            continue
        if line.startswith("Run (Python):"):
            section = "python"
            continue
        if section == "r":
            r_lines.append(line)
        elif section == "python":
            py_lines.append(line)
    if not r_lines and not py_lines:
        return "", stdout
    return "\n".join(r_lines).strip(), "\n".join(py_lines).strip()


def format_process_output(result: CommandResult) -> str:
    parts: list[str] = []
    if result.stderr.strip():
        parts.append(result.stderr.rstrip())
    if result.stdout.strip():
        parts.append(result.stdout.rstrip())
    if result.returncode != 0:
        parts.insert(0, "$ " + " ".join(str(part) for part in result.command))
        parts.insert(1, f"exit={result.returncode}")
    return "\n".join(parts)


class Xr2pIde:
    def __init__(self, root: tk.Tk, *, xr2p: Path, rscript: str, source: Path | None = None) -> None:
        self.root = root
        self.xr2p = xr2p
        self.rscript = rscript
        self.source_path = source
        self.current_python = ""
        self.current_out_path: Path | None = None

        self.timeout_var = tk.StringVar(value="30")
        self.round_var = tk.StringVar(value="")
        self.font_size_var = tk.IntVar(value=12)
        self.pretty_var = tk.BooleanVar(value=True)
        self.no_compile_var = tk.BooleanVar(value=False)
        self.autocomplete_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")
        self.elapsed_r_var = tk.StringVar(value="R: ")
        self.elapsed_py_var = tk.StringVar(value="Python: ")
        self.output_mode = "single"
        self.text_widgets: list[tk.Text] = []

        self.build_ui()
        if source is not None:
            self.load_source(source)

    def build_ui(self) -> None:
        self.root.title("xr2p IDE")
        self.root.geometry("1180x760")

        toolbar = ttk.Frame(self.root, padding=(6, 4))
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Open R", command=self.open_source).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Save R", command=self.save_source).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(toolbar, text="Save Python", command=self.save_python).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Button(toolbar, text="Clear Code", command=self.clear_code).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(toolbar, text="Translate", command=self.translate_current).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Run R", command=self.run_r_current).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(toolbar, text="Translate + Run", command=lambda: self.translate_current(run_mode="run")).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(toolbar, text="Run Both", command=lambda: self.translate_current(run_mode="run-both")).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(toolbar, text="Diff", command=lambda: self.translate_current(run_mode="run-diff")).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(toolbar, text="Timeout").pack(side=tk.LEFT)
        ttk.Entry(toolbar, width=5, textvariable=self.timeout_var).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(toolbar, text="Round").pack(side=tk.LEFT)
        ttk.Entry(toolbar, width=5, textvariable=self.round_var).pack(side=tk.LEFT, padx=(2, 8))
        ttk.Label(toolbar, text="Font").pack(side=tk.LEFT)
        font_spin = ttk.Spinbox(toolbar, from_=8, to=24, width=4, textvariable=self.font_size_var, command=self.update_text_fonts)
        font_spin.pack(side=tk.LEFT, padx=(2, 8))
        font_spin.bind("<Return>", lambda _event: self.update_text_fonts())
        font_spin.bind("<FocusOut>", lambda _event: self.update_text_fonts())
        ttk.Checkbutton(toolbar, text="Pretty R", variable=self.pretty_var).pack(side=tk.LEFT)
        ttk.Checkbutton(toolbar, text="No compile", variable=self.no_compile_var).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Checkbutton(toolbar, text="Autocomplete", variable=self.autocomplete_var).pack(side=tk.LEFT, padx=(4, 0))

        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        left = ttk.Frame(pane)
        right = ttk.Frame(pane)
        pane.add(left, weight=1)
        pane.add(right, weight=1)

        ttk.Label(left, text="R source").pack(anchor="w")
        self.r_text = self.text_widget(left)

        ttk.Label(right, text="Generated Python").pack(anchor="w")
        self.py_text = self.text_widget(right)

        output_frame = ttk.Frame(self.root, padding=(6, 0, 6, 4))
        output_frame.pack(fill=tk.BOTH, expand=False)
        self.output_label = ttk.Label(output_frame, text="Output")
        self.output_label.pack(anchor="w")
        self.output_pane = ttk.PanedWindow(output_frame, orient=tk.HORIZONTAL)
        self.output_pane.pack(fill=tk.BOTH, expand=True)

        self.single_output_frame = ttk.Frame(self.output_pane)
        self.r_output_frame = ttk.Frame(self.output_pane)
        self.py_output_frame = ttk.Frame(self.output_pane)
        self.output_text = self.text_widget(self.single_output_frame, height=10)
        ttk.Label(self.r_output_frame, text="R output").pack(anchor="w")
        self.r_output_text = self.text_widget(self.r_output_frame, height=10)
        ttk.Label(self.py_output_frame, text="Python output").pack(anchor="w")
        self.py_output_text = self.text_widget(self.py_output_frame, height=10)
        self.show_single_output()

        status = ttk.Frame(self.root, padding=(6, 2))
        status.pack(fill=tk.X)
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self.elapsed_r_var).pack(side=tk.LEFT, padx=(16, 0))
        ttk.Label(status, textvariable=self.elapsed_py_var).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(status, text="Clear output", command=self.clear_output).pack(side=tk.RIGHT)

        self.configure_syntax_tags(self.r_text)
        self.configure_syntax_tags(self.py_text)
        self.r_text.bind("<<Modified>>", lambda _event: self.on_text_modified(self.r_text, "r"))
        self.py_text.bind("<<Modified>>", lambda _event: self.on_text_modified(self.py_text, "python"))
        self.r_text.bind("<KeyPress>", self.r_autocomplete)

    def text_widget(self, parent: tk.Widget, *, height: int | None = None) -> tk.Text:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True)
        yscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        xscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
        text = tk.Text(
            frame,
            wrap=tk.NONE,
            undo=True,
            height=height,
            yscrollcommand=yscroll.set,
            xscrollcommand=xscroll.set,
            font=("Consolas", self.font_size_var.get()),
        )
        self.text_widgets.append(text)
        yscroll.configure(command=text.yview)
        xscroll.configure(command=text.xview)
        text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.text = text  # type: ignore[attr-defined]
        return frame.text  # type: ignore[attr-defined]

    def update_text_fonts(self) -> None:
        try:
            size = int(self.font_size_var.get())
        except (TypeError, tk.TclError, ValueError):
            size = 12
        size = max(8, min(size, 24))
        self.font_size_var.set(size)
        for widget in self.text_widgets:
            widget.configure(font=("Consolas", size))

    def timeout(self) -> float | None:
        raw = self.timeout_var.get().strip()
        if not raw:
            return 30.0
        try:
            value = float(raw)
        except ValueError:
            self.status_var.set("Invalid timeout; using 30 seconds")
            return 30.0
        return None if value == 0 else max(value, 0.1)

    def source_text(self) -> str:
        return self.r_text.get("1.0", "end-1c")

    def set_text(self, widget: tk.Text, text: str) -> None:
        widget.edit_modified(False)
        widget.delete("1.0", tk.END)
        if text:
            widget.insert("1.0", text)
        widget.edit_modified(False)

    def append_output(self, text: str) -> None:
        if text:
            self.output_text.insert(tk.END, text if text.endswith("\n") else text + "\n")
            self.output_text.see(tk.END)

    def clear_output(self) -> None:
        self.set_text(self.output_text, "")
        self.set_text(self.r_output_text, "")
        self.set_text(self.py_output_text, "")
        self.elapsed_r_var.set("R: ")
        self.elapsed_py_var.set("Python: ")

    def clear_code(self) -> None:
        self.set_text(self.r_text, "")
        self.set_text(self.py_text, "")
        self.current_python = ""
        self.current_out_path = None
        self.status_var.set("Code cleared")

    def show_single_output(self) -> None:
        self.output_mode = "single"
        self.output_label.configure(text="Output")
        for pane in self.output_pane.panes():
            self.output_pane.forget(pane)
        self.output_pane.add(self.single_output_frame, weight=1)

    def show_split_output(self) -> None:
        self.output_mode = "split"
        self.output_label.configure(text="Run output")
        for pane in self.output_pane.panes():
            self.output_pane.forget(pane)
        self.output_pane.add(self.r_output_frame, weight=1)
        self.output_pane.add(self.py_output_frame, weight=1)

    def configure_syntax_tags(self, widget: tk.Text) -> None:
        widget.tag_configure("keyword", foreground="#003f8c")
        widget.tag_configure("builtin", foreground="#6b3a00")
        widget.tag_configure("string", foreground="#0b6b28")
        widget.tag_configure("number", foreground="#7a1f7a")
        widget.tag_configure("comment", foreground="#666666")

    def on_text_modified(self, widget: tk.Text, language: str) -> None:
        if not widget.edit_modified():
            return
        widget.edit_modified(False)
        self.highlight(widget, language)

    def r_autocomplete(self, event: tk.Event) -> str | None:
        if not self.autocomplete_var.get():
            return None
        char = event.char
        if not char:
            return None
        pairs = {"(": ")", "{": "}", "[": "]", '"': '"', "'": "'"}
        closing = {")", "}", "]"}
        text = self.r_text
        if char in pairs:
            if char in {'"', "'"} and self.next_char() == char:
                text.mark_set("insert", "insert+1c")
                return "break"
            text.insert("insert", char + pairs[char])
            text.mark_set("insert", "insert-1c")
            return "break"
        if char in closing and self.next_char() == char:
            text.mark_set("insert", "insert+1c")
            return "break"
        if char in {"\r", "\n"}:
            return self.r_auto_newline()
        return None

    def next_char(self) -> str:
        return self.r_text.get("insert", "insert+1c")

    def r_auto_newline(self) -> str | None:
        text = self.r_text
        line = text.get("insert linestart", "insert")
        indent = line[: len(line) - len(line.lstrip(" \t"))]
        before = text.get("insert-1c", "insert")
        after = text.get("insert", "insert+1c")
        if before == "{" and after == "}":
            inner = indent + "    "
            text.insert("insert", "\n" + inner + "\n" + indent)
            text.mark_set("insert", "insert-1l lineend")
            return "break"
        if line.rstrip().endswith("{"):
            text.insert("insert", "\n" + indent + "    ")
            return "break"
        text.insert("insert", "\n" + indent)
        return "break"

    def highlight(self, widget: tk.Text, language: str) -> None:
        for tag in ("keyword", "builtin", "string", "number", "comment"):
            widget.tag_remove(tag, "1.0", tk.END)
        text = widget.get("1.0", "end-1c")

        def add(tag: str, start: int, end: int) -> None:
            widget.tag_add(tag, f"1.0+{start}c", f"1.0+{end}c")

        for match in STRING_RE.finditer(text):
            add("string", match.start(), match.end())
        for match in NUMBER_RE.finditer(text):
            add("number", match.start(), match.end())
        for line_match in re.finditer(r"(?m)^.*$", text):
            line = line_match.group(0)
            if language == "r":
                pos = line.find("#")
            else:
                pos = line.find("#")
            if pos >= 0:
                add("comment", line_match.start() + pos, line_match.end())
        keywords = R_KEYWORDS if language == "r" else PY_KEYWORDS
        builtins = R_BUILTINS if language == "r" else {"np", "pd", "stats", "r_print", "r_c"}
        for word in keywords:
            for match in re.finditer(rf"\b{re.escape(word)}\b", text):
                add("keyword", match.start(), match.end())
        for word in builtins:
            for match in re.finditer(rf"\b{re.escape(word)}\b", text):
                add("builtin", match.start(), match.end())

    def open_source(self) -> None:
        selected = filedialog.askopenfilename(title="Open R source", filetypes=R_FILETYPES)
        if selected:
            self.load_source(Path(selected))

    def load_source(self, path: Path) -> None:
        self.source_path = path
        self.set_text(self.r_text, read_text(path))
        self.highlight(self.r_text, "r")
        self.root.title(f"xr2p IDE - {path}")
        self.status_var.set(f"Loaded {path}")

    def save_source(self) -> None:
        path = self.source_path
        if path is None:
            selected = filedialog.asksaveasfilename(title="Save R source", filetypes=R_FILETYPES, defaultextension=".r")
            if not selected:
                return
            path = Path(selected)
            self.source_path = path
        path.write_text(self.source_text(), encoding="utf-8")
        self.status_var.set(f"Saved {path}")

    def save_python(self) -> None:
        if not self.current_python.strip():
            self.translate_current()
        if not self.current_python.strip():
            return
        initial = self.source_path.with_suffix(".py").name if self.source_path is not None else "xr2p_output.py"
        selected = filedialog.asksaveasfilename(title="Save Python", filetypes=PY_FILETYPES, defaultextension=".py", initialfile=initial)
        if selected:
            Path(selected).write_text(self.current_python, encoding="utf-8")
            self.status_var.set(f"Saved {selected}")

    def translate_current(self, *, run_mode: str | None = None) -> None:
        source = self.source_text()
        if not source.strip():
            self.status_var.set("No R source to translate")
            return
        self.status_var.set("Running xr2p...")
        if run_mode == "run-both":
            self.show_split_output()
        else:
            self.show_single_output()
        self.clear_output()
        threading.Thread(target=self._translate_worker, args=(source, run_mode), daemon=True).start()

    def run_r_current(self) -> None:
        source = self.source_text()
        if not source.strip():
            self.status_var.set("No R source to run")
            return
        self.status_var.set("Running R...")
        self.show_single_output()
        self.clear_output()
        threading.Thread(target=self._run_r_worker, args=(source,), daemon=True).start()

    def _run_r_worker(self, source: str) -> None:
        with tempfile.TemporaryDirectory(prefix="xr2p_ide_r_") as tmp:
            tmpdir = Path(tmp)
            source_name = self.source_path.name if self.source_path is not None else "xr2p_session.r"
            r_path = tmpdir / source_name
            r_path.write_text(source, encoding="utf-8")
            cwd = self.source_path.parent if self.source_path is not None else tmpdir
            result = run_command([*shlex.split(self.rscript), str(r_path)], cwd=cwd, timeout=self.timeout())
            self.root.after(0, lambda: self.finish_run_r(result))

    def finish_run_r(self, result: CommandResult) -> None:
        self.elapsed_r_var.set(f"R: {result.elapsed:.3f}s")
        self.set_text(self.output_text, format_process_output(result))
        state = "OK" if result.returncode == 0 else f"exit={result.returncode}"
        self.status_var.set(f"R {state} in {result.elapsed:.3f}s")

    def _translate_worker(self, source: str, run_mode: str | None) -> None:
        with tempfile.TemporaryDirectory(prefix="xr2p_ide_") as tmp:
            tmpdir = Path(tmp)
            source_name = self.source_path.name if self.source_path is not None else "xr2p_session.r"
            r_path = tmpdir / source_name
            py_path = tmpdir / (Path(source_name).stem + ".py")
            r_path.write_text(source, encoding="utf-8")

            command = [sys.executable, str(self.xr2p), str(r_path), "-o", str(py_path)]
            if self.no_compile_var.get():
                command.append("--no-py-compile")
            cli_run_mode = run_mode
            if run_mode == "run-both":
                cli_run_mode = None

            if cli_run_mode == "run":
                command.append("--run")
            elif cli_run_mode == "run-diff":
                command.append("--run-diff")
            if cli_run_mode in {"run-diff"} and self.pretty_var.get():
                command.append("--pretty")
            round_value = self.round_var.get().strip()
            if round_value:
                command.extend(["--round-both" if cli_run_mode in {"run-diff"} else "--round", round_value])
            if self.rscript:
                command.extend(["--rscript", self.rscript])

            result = run_command(command, cwd=tmpdir, timeout=self.timeout())
            python = py_path.read_text(encoding="utf-8", errors="replace") if py_path.exists() else ""
            r_result = None
            py_result = None
            if run_mode == "run-both" and result.returncode == 0 and py_path.exists():
                run_cwd = self.source_path.parent if self.source_path is not None else tmpdir
                r_result = run_command([*shlex.split(self.rscript), str(r_path)], cwd=run_cwd, timeout=self.timeout())
                py_result = run_command([sys.executable, str(py_path)], cwd=run_cwd, timeout=self.timeout())
            ide_result = IdeResult(result, python, py_path if py_path.exists() else None, run_mode, r_result, py_result)
            self.root.after(0, lambda: self.finish_translate(ide_result))

    def finish_translate(self, ide_result: IdeResult) -> None:
        result = ide_result.command_result
        self.current_python = ide_result.python
        self.current_out_path = ide_result.out_path
        self.set_text(self.py_text, ide_result.python)
        self.highlight(self.py_text, "python")
        if ide_result.run_mode == "run-both":
            r_text = ""
            py_text = ""
            if ide_result.r_result is not None:
                self.elapsed_r_var.set(f"R: {ide_result.r_result.elapsed:.3f}s")
                r_text = format_process_output(ide_result.r_result)
            if ide_result.py_result is not None:
                self.elapsed_py_var.set(f"Python: {ide_result.py_result.elapsed:.3f}s")
                py_text = format_process_output(ide_result.py_result)
            self.set_text(self.r_output_text, r_text)
            self.set_text(self.py_output_text, py_text)
            if result.stderr:
                self.py_output_text.insert(tk.END, ("\n" if py_text else "") + result.stderr)
        else:
            if ide_result.run_mode is not None or result.returncode != 0:
                if result.returncode != 0:
                    self.append_output("$ " + " ".join(str(part) for part in result.command))
                if result.stdout and not (ide_result.run_mode is None and result.stdout.strip().startswith("wrote ")):
                    self.append_output(result.stdout)
                if result.stderr:
                    self.append_output(result.stderr)
            if ide_result.run_mode == "run":
                self.elapsed_py_var.set(f"Python: {result.elapsed:.3f}s")
        state = "OK" if result.returncode == 0 else f"exit={result.returncode}"
        self.status_var.set(f"xr2p {state} in {result.elapsed:.3f}s")

    def show_help(self) -> None:
        messagebox.showinfo(
            "xr2p IDE",
            "Open or type R code, then use Translate or one of the run buttons.\n\n"
            "The GUI calls xr2p.py through the command line. It does not implement "
            "a separate translator.",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open a small R-to-Python IDE using xr2p.py.")
    parser.add_argument("source", nargs="?", help="optional R source file to open")
    parser.add_argument("--xr2p", default=str(DEFAULT_XR2P), help="path to xr2p.py")
    parser.add_argument("--rscript", default="rscript", help="Rscript command for run-both/diff/stats")
    args = parser.parse_args(argv)

    xr2p = Path(args.xr2p)
    if not xr2p.exists():
        print(f"xr2p IDE: xr2p.py not found: {xr2p}", file=sys.stderr)
        return 2
    source = Path(args.source) if args.source else None
    if source is not None and not source.exists():
        print(f"xr2p IDE: source file not found: {source}", file=sys.stderr)
        return 2

    root = tk.Tk()
    Xr2pIde(root, xr2p=xr2p, rscript=args.rscript, source=source)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
