import cv2
from faststream import FastStream
from faststream.redis import RedisBroker, PubSub
from faststream.redis.message import RedisMessage
from pydantic_settings import BaseSettings

from mirror_mirror.decode import decode_frame
from mirror_mirror.models import CarrierMessage, FrameMessage


class Config(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    channel: str = "frames:camera:*"


config = Config()

broker = RedisBroker(url=config.redis_url)
app = FastStream(broker=broker)


@broker.subscriber(channel=PubSub(config.channel, pattern=True))
def to_diffuse(
    carrier: CarrierMessage,
    _message: RedisMessage,
):
    match carrier.content:
        case FrameMessage(frame=frame):
            frame = decode_frame(frame)
            cv2.imshow(config.channel, frame)
    return None
