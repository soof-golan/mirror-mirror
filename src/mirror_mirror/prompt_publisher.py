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
    """Interactive prompt publisher"""
    print("Prompt Publisher - Enter prompts to send to the diffuser")
    print("Type 'exit' to quit")
    
    while True:
        user_input = input("Enter a prompt: ")
        if user_input.lower() == "exit":
            break
        
        await publish_prompt(user_input)


if __name__ == "__main__":
    asyncio.run(main()) 