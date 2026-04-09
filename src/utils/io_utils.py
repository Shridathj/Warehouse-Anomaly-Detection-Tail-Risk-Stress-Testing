# src/utils/io_utils.py
#
# RunContext - context manager that:
#   1. Tees sys.stdout to both console and a results/*.txt file simultaneously.
#   2. Intercepts matplotlib plt.show() and auto-saves the current figure as
#      results/<section>_plot01.png, _plot02.png, ...
#   3. Exposes save_plotly(fig) for Plotly figures.
#      Replace every `fig.show()` call inside a function with `ctx.save_plotly(fig)`
#      (or define a local alias at the top of the function — see template pattern).

import io
import sys
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# Tee helper 
class _Tee:
    """Writes to multiple streams at once."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return False

    def fileno(self):
        raise io.UnsupportedOperation("fileno")

# Main context manager 
class RunContext:
    """
    Parameters
    results_dir  : directory to write output files into.
    section_name : prefix used for all output files of this run.
    """
    def __init__(self, results_dir: str, section_name: str):
        self.results_dir  = Path(results_dir)
        self.section_name = section_name
        self._plot_count  = 0
        self._orig_show   = None
        self._orig_stdout = None
        self._txt_file    = None

    # Plot helpers 
    def _matplotlib_show(self):
        """Replacement for plt.show() — saves figure and closes it."""
        self._plot_count += 1
        path = self.results_dir / f"{self.section_name}_plot{self._plot_count:02d}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"[matplotlib figure saved -> {path.name}]", flush=True)

    def save_plotly(self, fig):
        """
        Call this instead of fig.show() for every Plotly figure.

        Pattern inside a function:
            def run_something(df, scenario, cfg, ctx=None):
                def _show(fig):               # local alias
                    if ctx:
                        ctx.save_plotly(fig)
                    else:
                        fig.show()
                ...
                _show(fig)
        """
        self._plot_count += 1
        path = self.results_dir / f"{self.section_name}_plot{self._plot_count:02d}.html"
        fig.write_html(str(path))
        print(f"[plotly figure saved  -> {path.name}]", flush=True)

    # Context protocol 
    def __enter__(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        txt_path      = self.results_dir / f"{self.section_name}.txt"
        self._txt_file = open(txt_path, "w", encoding="utf-8")

        # Write to original stdout AND to file
        self._orig_stdout = sys.stdout
        sys.stdout = _Tee(self._orig_stdout, self._txt_file)

        # Intercept matplotlib show
        self._orig_show = plt.show
        plt.show = self._matplotlib_show

        print(f"{'='*60}", flush=True)
        print(f"SECTION : {self.section_name}", flush=True)
        print(f"Output  : {txt_path}", flush=True)
        print(f"{'='*60}", flush=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore matplotlib show
        plt.show = self._orig_show

        # Restore stdout
        sys.stdout = self._orig_stdout

        # Close log file
        if self._txt_file:
            self._txt_file.close()

        return False