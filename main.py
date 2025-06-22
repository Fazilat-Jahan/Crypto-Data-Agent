from dotenv import load_dotenv
import os
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, function_tool
import requests
import asyncio
import chainlit as cl
load_dotenv()

@function_tool
def get_crypto_price(symbol: str) -> str:
    """
    Fetch the current market price of a crypto symbol from Binance.

    Args:
        symbol: e.g., BTCUSDT, ETHUSDT, etc.
    """

    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}"
    res = requests.get(url)
    if res.status_code == 200:
        price = res.json()['price']
        return f"The current price of {symbol.upper()} is ${price}"
    return f"Could not fetch price for {symbol.upper()}"


@cl.on_chat_start
async def start():
    MODEL_NAME = "gemini-2.0-flash"
    API_KEY= os.getenv("GEMINI_API_KEY")

    if not API_KEY:
        raise ValueError("GEMINI_API_KEY is not set yet.")

    external_client = AsyncOpenAI(
            api_key = API_KEY,
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    
    model = OpenAIChatCompletionsModel(
            model= MODEL_NAME,
            openai_client = external_client
        )


    cl.user_session.set("chat_history", [])
    
    CryptoDataAgent = Agent(
        name="Crypto Data Agent",
        instructions="""You are a helpful crypto data agent. 
        Only call the get_crypto_price tool when a user explicitly asks for a specific coin (e.g., BTCUSDT, ETHUSDT). 
        Do not provide prices for multiple coins unless asked.""",
        model=model,
        tools=[get_crypto_price]
    )

    cl.user_session.set("agent", CryptoDataAgent)
    await cl.Message(content= """Hi! I'm your Crypto Rates Agent. Ask me the current price of any coin — BTCUSDT, ETHUSDT, and more!""").send()

@cl.on_message
async def main(message: cl.Message):
    msg = await cl.Message(content = "Thinking your Query...").send()

    CryptoDataAgent = cl.user_session.get("agent")
    history = cl.user_session.get("chat_history")
    history.append({"role":"user", "content": message.content})

    result = await Runner.run(starting_agent=CryptoDataAgent, input = history)
    msg.content = result.final_output
    await msg.update()
    cl.user_session.set("chat_history", history)
    
    print(result.final_output)
