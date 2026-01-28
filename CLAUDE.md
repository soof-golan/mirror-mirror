# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Mirror Mirror** - An art installation that transforms a live camera feed using ControlNet diffusion, responding to voice commands.

Python 3.12 project managed with `uv`.

## Commands

```bash
# Add dependencies
uv add <package>

# Sync virtual environment
uv sync
# NEVER use `uv pip` or `pip` directly

# Run any command in the venv
uv run <command>

# Never run python directly, always use:
uv run python <script.py>
```

## Workflow

**Commit often with atomic changes.** Each commit should be a single logical change that compiles and works.

**After a development round, always remove comments.** Clean up inline comments before the final commit.

## Design Documentation

**Read @README.md for architecture decisions.** This document contains:
- System architecture and process layout
- IPC patterns (ZeroMQ)
- Model choices and rationale
- WebRTC streaming design

Re-read the design doc after every context refresh to maintain continuity.

## Target Platforms

- **Deployment**: Linux with RTX 4090 (via SSH)

## Coding Standards

- NO DOCSTRINGS EVER, you may only remove them.
- NO COMMENTS EVER, you may only remove them.
- use `uvx ruff format` to re-format code.
- use `uvx ruff check` to lint code.
