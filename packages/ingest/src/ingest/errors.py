"""User-facing errors. A tool catches these and prints one `error:` line."""


class UserError(Exception):
    """Base class: a problem the user can fix, shown as one readable line."""


class AudioError(UserError):
    """ffmpeg missing or input audio unreadable."""


class YouTubeError(UserError):
    """A YouTube URL that cannot be used (playlist, private video, network...)."""


class ModelError(UserError):
    """A model that cannot be downloaded, loaded or run."""
