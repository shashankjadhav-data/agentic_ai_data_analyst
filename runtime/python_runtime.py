# runtime/python_runtime.py
import contextlib
import io
import textwrap
import time
import traceback


def run_python(code: str, context: dict | None = None):
    """
    Execute Python code in a single shared namespace so functions can see injected vars.
    """
    start = time.time()
    ns = {} if context is None else dict(context)  # one namespace for globals+locals
    stdout = io.StringIO()
    error = None

    with contextlib.redirect_stdout(stdout):
        try:
            exec(textwrap.dedent(code), ns, ns)
        except Exception:
            error = traceback.format_exc()

    duration = time.time() - start
    return {
        "ok": error is None,
        "stdout": stdout.getvalue(),
        "error": error,
        "locals": ns,
        "duration": duration,
    }
