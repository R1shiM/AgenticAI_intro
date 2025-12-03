from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, load_tool, tool
import datetime
import requests
import pytz
import yaml
from tools.final_answer import FinalAnswerTool

from Gradio_UI import GradioUI

@tool
def pokemon_info(name: str) -> str:
    """Get Pokémon stats such as height, weight, type.
    Args:
        name: The name of the pokemon.
    """
    import requests
    name_of_pokemon = name
    url = f"https://pokeapi.co/api/v2/pokemon/{name_of_pokemon.lower()}"
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
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: the timezone the user wants to fetch the time in."""
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"

final_answer = FinalAnswerTool()

model = HfApiModel(
    max_tokens=2096,
    temperature=0.5,
    model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
    custom_role_conversions=None,
)

image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)
    
agent = CodeAgent(
    model=model,
    tools=[final_answer, pokemon_info, get_current_time_in_timezone],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

GradioUI(agent).launch()
