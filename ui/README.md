# PyWebView UI

HTML/CSS/JS UI using PyWebView (native WebView wrapper for Python).

## Dependencies

```bash
uv add pywebview pyzmq
```

## Run

```bash
uv run python ui-options/pywebview-ui/main.py
```

## Architecture

- `main.py`: Python backend that connects to ZMQ IPC sockets and exposes API to frontend
- `index.html`, `style.css`, `app.js`: Frontend that renders frames and vignette overlay

## Features

- Polls ZMQ sockets in background thread
- Exposes `get_all_state()` API to JavaScript
- CSS-based vignette using `box-shadow: inset`
- Press `F` or double-click to toggle fullscreen

## Platform Notes

- Works on macOS, Linux, Windows
- Uses native WebView (WebKit on macOS/Linux, Edge on Windows)
