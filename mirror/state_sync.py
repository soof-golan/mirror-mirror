import logging
import time
from itertools import count
from typing import Annotated

import typer
import zmq
from tqdm import tqdm
from typer import Typer
from zmq import Socket

from mirror.config import config
from mirror.db import get_current_prompt, get_random_prompt, init_db, save_prompt
from mirror.logger import logger
from mirror.zmq_utils import create_socket, recv_prompt

app = Typer()


@app.command()
def main(
    shuffle_every: Annotated[
        int,
        typer.Option(
            help="Number of seconds without activity before shuffling from the prompt history"
        ),
    ] = 300,
    zmq_poller_timeout: Annotated[int, typer.Option(help="Polling timeout in milliseconds")] = 1000,
    log_level: str = "INFO",
) -> None:
    logging.basicConfig(level=log_level)
    logger.info("State sync started")
    init_db()
    prompt_sub = create_socket(zmq.SUB, config.zmq_prompt_bind_addr, "bind", conflate=False)
    republish = create_socket(zmq.PUB, config.zmq_prompt_pub_bind_addr, "bind", conflate=False)
    state_router = create_socket(zmq.DEALER, config.zmq_state_bind_addr, "bind", conflate=False)

    poller = zmq.Poller()
    poller.register(prompt_sub, zmq.POLLIN)
    poller.register(state_router, zmq.POLLIN)

    last_prompt_time = time.time()
    for _ in tqdm(count()):
        socks = dict(poller.poll(timeout=zmq_poller_timeout))
        if prompt_sub in socks:
            republish_and_save_prompt(prompt_sub, republish)
            last_prompt_time = time.time()
        if state_router in socks:
            handle_state_sync(state_router)

        if time.time() - last_prompt_time > shuffle_every:
            last_prompt_time = time.time()
            publish_random_prompt(republish)


def publish_random_prompt(republish: Socket):
    prompt = get_random_prompt()
    logger.info("Shuffling Prompt: %s", prompt)
    republish.send_string(prompt)


def handle_state_sync(state_router: Socket):
    # ROUTER receives: [identity, empty, request]
    frames = state_router.recv_multipart()
    if len(frames) != 3:
        logger.info("Invalid message received %s", frames)
        return

    identity = frames[0]
    request = frames[-1].decode() if frames[-1] else ""

    if request == "get_prompt":
        current_prompt = get_current_prompt()
        logger.debug("Sent prompt snapshot to worker: %s", current_prompt[:50])
        # ROUTER sends: [identity, empty, response]
        state_router.send_multipart([identity, b"", current_prompt.encode()])


def republish_and_save_prompt(prompt_sub: Socket, republish: Socket):
    prompt = recv_prompt(prompt_sub)
    if prompt:
        logger.info("Republish prompt %s", prompt.text)
        republish.send_string(prompt.text)
        save_prompt(prompt.text)


if __name__ == "__main__":
    app()
