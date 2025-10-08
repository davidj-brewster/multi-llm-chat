SUPPORTED_MODELS = {
    "claude": [
        "claude", "sonnet", "haiku", "claude*", "claude-3-5-haiku",
        "claude-3-7", "claude-3-7-latest", "claude-3-7-sonnet-latest", "claude-3-7-sonnet",
        "claude-3-7-reasoning", "claude-3-7-reasoning-medium", "claude-3-7-reasoning-low",
        "claude-3-7-reasoning-none",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ],
    "gemini": [
        "gemini*", "gemini-2-flash-lite", "gemini-2.5-flash-exp", "gemini-2.5-pro",
        "gemini-2.5-pro-exp", "gemini-2.5-pro-preview-03-25",
        "gemini-2.5-flash-preview-04-17", "gemini-2-pro", "gemini-2-reasoning",
        "gemini-2.0-flash-exp", "gemini",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest",
    ],
    "openai": [
        "gpt-4-vision", "gpt-4o", "gpt-4.1", "gpt*", "gpt-4.1-mini",
        "gpt-4.1-nano", "chatgpt-latest", "o1", "o3",
        "o1-reasoning-high", "o1-reasoning-medium", "o1-reasoning-low",
        "o3-reasoning-high", "o3-reasoning-medium", "o3-reasoning-low",
        "o4-mini",
        "o4-mini-high",
    ],
    "gpt-4.1": ["gpt-4.1"],
    "gpt-4.1-mini": ["gpt-4.1-mini"],
    "gpt-4.1-nano": ["gpt-4.1-nano"],
    "o4-mini": ["o4-mini"],
    "o4-mini-high": ["o4-mini-high"],
    "o3": ["o3"],
    "ollama": ["*"],
    "mlx": ["*"],
}

SUPPORTED_FILE_TYPES = {
    "image": {
        "extensions": [".jpg", ".jpeg", ".png", ".gif", ".webp"],
        "max_size": 200 * 1024 * 1024,
        "max_resolution": (8192, 8192),
    },
    "video": {
        "extensions": [".mp4", ".mov", ".avi", ".webm"],
        "max_size": 3000 * 1024 * 1024,
        "max_resolution": (3840, 2160),
    },
    "text": {
        "extensions": sorted(list(set([
            ".txt", ".md", ".py", ".js", ".html", ".css", ".json", ".yaml", ".yml",
            ".java", ".cpp", ".c", ".h", ".cs", ".php", ".rb", ".go", ".rs", ".ts", ".swift"
        ]))),
        "max_size": 300 * 1024 * 1024,
    },
}