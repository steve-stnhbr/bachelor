import sys
from contextlib import contextmanager
from io import StringIO

class StdoutInterceptor:
    def __init__(self, callback):
        self.callback = callback
        self.stdout = sys.stdout
        self.stringio = StringIO()

    def write(self, data):
        self.stringio.write(data)
        self.stdout.write(data)
        self.callback(data)

    def flush(self):
        self.stdout.flush()

@contextmanager
def intercept_stdout(callback):
    interceptor = StdoutInterceptor(callback)
    sys.stdout = interceptor
    try:
        yield interceptor
    finally:
        sys.stdout = interceptor.stdout