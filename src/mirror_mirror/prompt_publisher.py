import asyncio
from faststream import FastStream
from faststream.redis import RedisBroker
from pydantic import BaseModel
from typing import Literal
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    redis_url: str = "redis://localhost:6379"


config = Config()


class PromptMessage(BaseModel):
    msg_type: Literal["prompt"] = "prompt"
    prompt: str


async def publish_prompt(prompt: str):
    """Publish a prompt message to update the diffuser"""
    broker = RedisBroker(url=config.redis_url)
    
    try:
        # Create message
        message = PromptMessage(prompt=prompt)
        
        # Publish message
        await broker.publish(
            message,
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