import io
from typing import Literal

import faststream
from faststream import FastStream
from faststream.redis import RedisBroker, PubSub
from faststream.redis.message import RedisMessage
from pydantic import BaseModel, Base64Bytes
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    redis_url: str = "redis://localhost:6379"
    mode: Literal["fake"] = "fake"

config = Config()

broker = RedisBroker(url=config.redis_url)
app = FastStream(broker=broker)

class FrameMessage(BaseModel):
    img: Base64Bytes

@broker.subscriber(channel=PubSub("to-diffuse:{feed}", pattern=True))
@broker.subscriber(channel=PubSub("prompts:*", pattern=True))
@broker.publisher(channel="diffused_images")
def to_diffuse(msg: FrameMessage, _message: RedisMessage, logger: faststream.Logger, feed: str = faststream.Path()):
    logger.info("feed %s. frame nbytes %s", feed, len(msg.img))

    io.BytesIO(msg.img)



