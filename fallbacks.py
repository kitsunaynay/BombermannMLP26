import contextlib


class QuietFallback:
    def __getattr__(self, item):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __iter__(self):
        return iter([])


try:
    with contextlib.redirect_stdout(None):
        import pygame
        LOADED_PYGAME = True
    pygame.init()
except ModuleNotFoundError:
    pygame = QuietFallback()
    # Was `True`, which made main.py's "pygame could not loaded" guard
    # unreachable: running with a GUI without pygame installed silently drew
    # nothing instead of reporting the problem.
    LOADED_PYGAME = False

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda iterable, *args, **kwargs: iterable
