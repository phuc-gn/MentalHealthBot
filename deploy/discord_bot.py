import os
import discord
import requests
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)

server = 'http://localhost:5000/generate'

async def get_vllm_response(prompt):
    try:
        response = requests.post(server, json={"prompt": prompt})
        return response.json().get('text', 'No response from vLLM.')
    except Exception as e:
        return f"Error communicating with LLM server: {str(e)}"

@bot.command(name="ask", help="Ask the AI a question.")
async def ask(ctx, *, question: str):
    response = await get_vllm_response(question)
    await ctx.send(response)

@bot.event
async def on_ready():
    print(f'Bot is ready and logged in as {bot.user}')

bot.run(os.getenv('DISCORD_TOKEN'))