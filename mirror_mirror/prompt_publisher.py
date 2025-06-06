import asyncio
import time
from faststream import FastStream
from faststream.redis import RedisBroker
from pydantic_settings import BaseSettings

from mirror_mirror.models import PromptMessage, CarrierMessage


class Config(BaseSettings):
    redis_url: str = "redis://localhost:6379"


config = Config()


async def publish_prompt(prompt: str):
    """Publish a prompt message to update the diffuser"""
    broker = RedisBroker(url=config.redis_url)
    
    try:
        # Connect to broker
        await broker.connect()
        
        # Create message
        prompt_msg = PromptMessage(prompt=prompt, timestamp=time.time())
        carrier = CarrierMessage(content=prompt_msg)
        
        # Publish message
        await broker.publish(
            carrier,
            channel="prompts:global",
        )
        print(f"Published prompt: {prompt}")
    
    finally:
        # Release resources
        await broker.close()


async def main():
    """CLI prompt publisher - pass prompt as argument"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m mirror_mirror.prompt_publisher 'your prompt here'")
        sys.exit(1)
    
    prompt = " ".join(sys.argv[1:])
    await publish_prompt(prompt)


if __name__ == "__main__":
    asyncio.run(main()) 