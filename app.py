from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, load_tool, tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI

@tool
def pokemon_info(name: str) -> str:
    """Get Pokémon stats such as height, weight, type."""
    name = name.lower().replace(" ", "-")  # safer
    url = f"https://pokeapi.co/api/v2/pokemon/{name}"
    res = requests.get(url)
    if res.status_code != 200:
        return "Pokémon not found."
    data = res.json()
    types = ", ".join([t["type"]["name"] for t in data["types"]])
    return (
        f"{data['name'].capitalize()}: "
        f"Height {data['height']}, Weight {data['weight']}, Type(s): {types}"
    )

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """Fetch current time in a specified timezone."""
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

final_answer = FinalAnswerTool()

model = InferenceClientModel(
    max_tokens=2096,
    temperature=0.5,
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
)

image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", "r") as stream:
    prompt_templates = yaml.safe_load(stream)

agent = CodeAgent(
    model=model,
    tools=[
        final_answer,
        pokemon_info,
        get_current_time_in_timezone,
        image_generation_tool,   # ← REQUIRED
    ],
    max_steps=6,
    verbosity_level=1,
    prompt_templates=prompt_templates
)

GradioUI(agent).launch()
