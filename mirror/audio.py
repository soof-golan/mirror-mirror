"""Audio process - Gemma 3n for audio understanding."""

import logging
import random
import time
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from itertools import count
from typing import Any, Literal

import torch
import zmq
from transformers import (
    AutoProcessor,
    Gemma3nForConditionalGeneration,
    Gemma3nProcessor,
    GenerationConfig,
    StaticCache,
    TextIteratorStreamer,
)
from typer import Typer
from zmq import Socket

import mirror.native_mic
from mirror.audio_buffer import AudioBuffer
from mirror.config import config
from mirror.generated.messages import Prompt
from mirror.system_instructions import LEADING_PROMPT
from mirror.utils import BoundedThreadPoolExecutor, log_timing, quit_on_error
from mirror.zmq_utils import create_socket, recv_audio, send_prompt

logger = logging.getLogger(__name__)

app = Typer()


def de_lion(prompt):
    if "majestic lion" in prompt and "intense gaze" in prompt:
        logger.info("Skipped the lion prompt :(")
        return None
    return prompt


@torch.inference_mode()
@app.command()
def main(
    mic_mode: Literal["zmq", "zmq+native"] = "zmq+native",
    mic_lookup: str = "tascam",
    torch_mm_precision: str = "medium",
    process_interval: float = 1.5,
    min_buffer_duration: float = 5.0,
    partial_clear_keep_seconds: float = 20.0,
    max_buffer_seconds: int = 30,
    device="cuda",
) -> None:
    """Main audio process entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s\t%(levelname)s\t%(message)s",
    )
    logger.info("Audio process starting...")
    logger.info("Config: %s", config)
    torch.set_float32_matmul_precision(torch_mm_precision)
    buffer = AudioBuffer(max_seconds=max_buffer_seconds)

    with ExitStack() as stack:
        if "native" in mic_mode:
            launch_native_mic(stack, mic_lookup)

        generate_pool = BoundedThreadPoolExecutor(max_workers=1, max_queue_size=2)
        model, processor = load_audio_model(device=device)
        audio_sub, prompt_pub = setup_zmq_networking(stack)
        leading_prompt, prompt_cache = setup_prompt(model, processor)
        logger.info("Audio process ready, listening for audio chunks...")
        loop(
            buffer=buffer,
            audio_sub=audio_sub,
            prompt_pub=prompt_pub,
            min_buffer_duration=min_buffer_duration,
            process_interval=process_interval,
            partial_clear_keep_seconds=partial_clear_keep_seconds,
            leading_prompt=leading_prompt,
            prompt_cache=prompt_cache,
            processor=processor,
            model=model,
            generate_pool=generate_pool,
        )


def loop(
    *,
    buffer: AudioBuffer,
    audio_sub: zmq.Socket,
    prompt_pub: zmq.Socket,
    min_buffer_duration: float,
    process_interval: float,
    partial_clear_keep_seconds: float,
    leading_prompt: list[dict[str, Any]],
    prompt_cache: StaticCache,
    processor: Gemma3nProcessor,
    model: Gemma3nForConditionalGeneration,
    generate_pool: BoundedThreadPoolExecutor,
):
    last_process_time = time.time()
    for i in count():
        fill_buffer(audio_sub, buffer)

        if buffer.duration_seconds() <= min_buffer_duration:
            continue

        current_time = time.time() + random_jitter()
        if current_time - last_process_time < process_interval:
            continue
        last_process_time = current_time

        process_audio_chunk(
            buffer=buffer,
            generate_pool=generate_pool,
            model=model,
            leading_prompt=leading_prompt,
            partial_clear_keep_seconds=partial_clear_keep_seconds,
            processor=processor,
            prompt_cache=prompt_cache,
            prompt_pub=prompt_pub,
        )


def process_audio_chunk(
    buffer: AudioBuffer,
    generate_pool: BoundedThreadPoolExecutor,
    leading_prompt: list[dict[str, Any]],
    model: Gemma3nForConditionalGeneration,
    partial_clear_keep_seconds: float,
    processor: Gemma3nProcessor,
    prompt_cache: StaticCache,
    prompt_pub: Socket,
):
    audio_array = buffer.get_audio()
    messages = leading_prompt + [
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]}
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generate_pool.submit(
        model.generate,
        **inputs,
        past_key_values=deepcopy(prompt_cache),
        max_new_tokens=100,
        streamer=streamer,
        generation_config=GenerationConfig(),
    ).add_done_callback(quit_on_error)

    _ = next(streamer)
    _ = next(streamer)
    first_token = next(streamer)

    response = ""
    match first_token.strip().upper():
        case "WAIT":
            logger.info("Trigger detected but incomplete, waiting for more audio")
            return
        case "NONE":
            logger.debug("No trigger detected, continuing to buffer")
            return
        case "PROMPT":
            response = "".join(list(streamer)).strip()
        case _:
            response = list(streamer)
            logger.warning("Unexpected first token: %r FULL RESPONSE %r", first_token, response)
            return
    if not response:
        logger.debug("Empty response, continuing to buffer")
        return

    response = de_lion(response)
    if not response:
        return

    prompt = response  # + POSTFIX
    logger.info("Detected prompt: %s", prompt)

    prompt_msg = Prompt(text=prompt, timestamp=time.time())
    send_prompt(prompt_pub, prompt_msg)

    buffer.partial_clear(keep_seconds=partial_clear_keep_seconds)


def fill_buffer(audio_sub: Socket, buffer: AudioBuffer):
    while audio_sub.poll(timeout=0):
        audio_msg = recv_audio(audio_sub)
        buffer.add_chunk(audio_msg.data, audio_msg.sample_rate, audio_msg.channels)


@torch.inference_mode()
def setup_prompt(
    model: Gemma3nForConditionalGeneration, processor: Gemma3nProcessor
) -> tuple[list[dict[str, str | list[dict[str, str]]]], StaticCache]:
    logger.info("Initializing prompt cache")
    # system_message = {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCTION}]}
    leading_prompt = LEADING_PROMPT  # [system_message]  # , *load_fewshot_examples()]
    prompt_cache = initialize_prefix_cache(model, processor, leading_prompt)
    return leading_prompt, prompt_cache


def setup_zmq_networking(stack: ExitStack[bool | None]) -> tuple[Socket, Socket]:
    logger.info("Setting up ZMQ sockets, connecting to %s...", config.zmq_host)
    audio_sub = create_socket(
        zmq.SUB, config.zmq_audio_addr, "connect", conflate=False, exit_stack=stack
    )
    prompt_pub = create_socket(zmq.PUB, config.zmq_prompt_addr, mode="connect", exit_stack=stack)
    return audio_sub, prompt_pub


def random_jitter() -> float:
    return random.random() - 0.5


def initialize_prefix_cache(
    model: Gemma3nForConditionalGeneration,
    processor: Gemma3nProcessor,
    leading_prompt: dict[str, str | list[dict[str, str]]],
) -> StaticCache:
    system_inputs = processor.apply_chat_template(
        leading_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    prompt_cache = StaticCache(config=model.config, max_cache_len=system_inputs.input_ids.shape[1])
    prompt_cache = model(**system_inputs, past_key_values=prompt_cache).past_key_values
    return prompt_cache


def load_audio_model(device: str) -> tuple[Gemma3nForConditionalGeneration, Gemma3nProcessor]:
    logger.info("Using device: %s", device)
    logger.info("Loading model: %s", config.gemma_model)
    processor: Gemma3nProcessor = AutoProcessor.from_pretrained(config.gemma_model)
    model = Gemma3nForConditionalGeneration.from_pretrained(
        config.gemma_model,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device,
    )
    model.eval()
    model.__call__ = log_timing(model.__call__, "Gemma3n.forward")
    logger.info("Model %s loaded successfully", config.gemma_model)
    return model, processor


def launch_native_mic(stack: ExitStack, lookup: str):
    mic_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="NativeMicWorker")
    stack.enter_context(mic_pool)
    mic_pool.submit(
        mirror.native_mic.main,
        lookup=lookup,
        log_level="INFO",
        setup_logging=False,
    ).add_done_callback(quit_on_error)


if __name__ == "__main__":
    app()
