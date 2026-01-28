"""ZeroMQ utilities for inter-process communication."""

from contextlib import ExitStack, closing
from typing import Literal

import zmq

from mirror.generated.messages import Audio, Prompt
from mirror.logger import logger


def create_socket(
    socket_type: int,
    address: str,
    mode: Literal["bind", "connect"],
    conflate: bool = True,
    hwm: int = 100,
    exit_stack: ExitStack = None,
) -> zmq.Socket:
    """Create a ZMQ socket with standard options.

    Args:
        socket_type: ZMQ socket type (zmq.PUB, zmq.SUB, zmq.PUSH, zmq.PULL, etc.)
        address: Socket address (tcp://, inproc://, etc.)
        mode: "bind" or "connect"
        conflate: If True, set CONFLATE=1 to keep only latest message (default True)

    Returns:
        Configured ZMQ socket
    """
    ctx = zmq.Context.instance()
    socket = ctx.socket(socket_type)

    if conflate:
        socket.setsockopt(zmq.CONFLATE, 1)
    elif hwm:
        socket.setsockopt(zmq.SNDHWM, hwm)
        socket.setsockopt(zmq.RCVHWM, hwm)

    if socket_type == zmq.SUB:
        socket.setsockopt(zmq.SUBSCRIBE, b"")

    if mode == "bind":
        socket.bind(address)
        logger.debug("Socket bound to %s", address)
    else:
        socket.connect(address)
        logger.debug("Socket connected to %s", address)

    if exit_stack is not None:
        exit_stack.enter_context(closing(socket))

    return socket


def recv_audio(socket: zmq.Socket) -> Audio:
    """Receive an audio message. Returns None if unavailable."""
    data = socket.recv()
    return Audio().parse(data)


def send_prompt(socket: zmq.Socket, prompt: Prompt) -> None:
    """Send a prompt message using protobuf serialization."""
    socket.send(bytes(prompt))


def recv_prompt(socket: zmq.Socket, flags: int = 0) -> Prompt | None:
    """Receive a prompt message. Returns None if unavailable."""
    try:
        data = socket.recv(flags=flags)
        return Prompt().parse(data)
    except zmq.Again:
        return None
