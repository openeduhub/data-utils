from hypothesis import settings

# a special configuration that is more reproducible
settings.register_profile(
    "build",
    deadline=None,
    derandomize=True,
    max_examples=100,
)
