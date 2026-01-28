import base64
import os
import struct
import threading
from pathlib import Path

import webview
import zmq

from mirror.config import Config
from mirror.generated.messages import Frame


def unpack_vignette(data: bytes) -> tuple[float, float]:
    return struct.unpack("ff", data)


class MirrorAPI:
    def __init__(self, config: Config):
        self.config = config
        self._window = None
        self._running = False
        self._thread = None

        self._frame_b64 = ""
        self._vignette_intensity = 0.0
        self._vignette_hue = 200.0
        self._prompt = config.default_prompt

        self._frame_lock = threading.Lock()
        self._vignette_lock = threading.Lock()
        self._prompt_lock = threading.Lock()

    def set_window(self, window):
        self._window = window

    def start_polling(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop_polling(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _request_prompt_snapshot(self):
        ctx = zmq.Context.instance()
        state = ctx.socket(zmq.DEALER)
        state.connect(self.config.zmq_state_addr)
        state.send_multipart([b"ui", b"", b"get_prompt"])
        if state.poll(timeout=3000):
            reply = state.recv_multipart()
            if len(reply) >= 3:
                prompt = reply[2].decode()
                self._prompt = prompt
        state.close()

    def _poll_loop(self):
        ctx = zmq.Context.instance()

        self._request_prompt_snapshot()

        diffusion_sub = ctx.socket(zmq.SUB)
        diffusion_sub.setsockopt(zmq.CONFLATE, 1)
        diffusion_sub.setsockopt(zmq.SUBSCRIBE, b"")
        diffusion_sub.connect(self.config.ipc_ui_diffusion_addr)

        vignette_sub = ctx.socket(zmq.SUB)
        vignette_sub.setsockopt(zmq.CONFLATE, 1)
        vignette_sub.setsockopt(zmq.SUBSCRIBE, b"")
        vignette_sub.connect(self.config.ipc_ui_vignette_addr)

        prompt_sub = ctx.socket(zmq.SUB)
        prompt_sub.setsockopt(zmq.CONFLATE, 1)
        prompt_sub.setsockopt(zmq.SUBSCRIBE, b"")
        prompt_sub.connect(self.config.ipc_ui_prompt_addr)

        poller = zmq.Poller()
        poller.register(diffusion_sub, zmq.POLLIN)
        poller.register(vignette_sub, zmq.POLLIN)
        poller.register(prompt_sub, zmq.POLLIN)

        try:
            while self._running:
                socks = dict(poller.poll(timeout=16))

                if diffusion_sub in socks:
                    frame_msg = Frame.parse(diffusion_sub.recv())
                    b64_data = base64.b64encode(frame_msg.data).decode("ascii")
                    self._frame_b64 = b64_data

                if vignette_sub in socks:
                    intensity, hue = unpack_vignette(vignette_sub.recv())
                    self._vignette_intensity = intensity
                    self._vignette_hue = hue

                if prompt_sub in socks:
                    prompt_text = prompt_sub.recv_string()
                    self._prompt = prompt_text

        finally:
            diffusion_sub.close()
            vignette_sub.close()
            prompt_sub.close()

    def get_frame(self) -> str:
        return self._frame_b64

    def get_vignette(self) -> dict:
        return {
            "intensity": self._vignette_intensity,
            "hue": self._vignette_hue,
        }

    def get_prompt(self) -> str:
        return self._prompt

    def get_all_state(self) -> dict:
        frame = self._frame_b64
        intensity = self._vignette_intensity
        hue = self._vignette_hue
        prompt = self._prompt
        return {
            "frame": frame,
            "vignette": {"intensity": intensity, "hue": hue},
            "prompt": prompt,
        }

    def toggle_fullscreen(self):
        if self._window:
            self._window.toggle_fullscreen()


def main():
    import logging

    logging.basicConfig(level="INFO", format="%(levelname)s\t%(name)s\t%(message)s")

    config = Config()
    os.makedirs(config.ipc_dir, exist_ok=True)

    logging.info("PyWebView UI starting...")
    logging.info("Subscribing to diffusion at %s", config.ipc_ui_diffusion_addr)
    logging.info("Subscribing to vignette at %s", config.ipc_ui_vignette_addr)
    logging.info("Subscribing to prompt at %s", config.ipc_ui_prompt_addr)

    api = MirrorAPI(config)

    html_path = Path(__file__).parent / "index.html"

    window = webview.create_window(
        "Mirror Mirror",
        url=str(html_path),
        js_api=api,
        width=1024,
        height=1024,
        resizable=True,
        frameless=False,
    )

    api.set_window(window)

    def on_loaded():
        logging.info("PyWebView window loaded, starting ZMQ polling...")
        api.start_polling()

    window.events.loaded += on_loaded

    def on_closed():
        api.stop_polling()

    window.events.closed += on_closed

    webview.start(debug=False)


if __name__ == "__main__":
    main()
