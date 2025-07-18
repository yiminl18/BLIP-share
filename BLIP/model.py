import os, sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from BLIP.models.gpt_4o import gpt_4o 
from BLIP.models.gpt_4o_mini import gpt_4o_mini
from BLIP.models.gpt_4o_mini_azure import gpt_4o_mini_azure
from BLIP.models.gpt_4o_azure import gpt_4o_azure
from BLIP.models.gemini_2 import gemini_2_flash

#this is the models API. You pass the model (name of the model) and prompt, the API will return the response out 
def model(model_name, prompt, image_path = ''):
    if(model_name == 'gpt4o'):
        return gpt_4o(prompt)
    if(model_name == 'gpt4omini'):
        return gpt_4o_mini(prompt)
    if(model_name == 'gpt_4o_mini_azure'):
        return gpt_4o_mini_azure(prompt)
    if(model_name == 'gpt_4o_azure'):
        return gpt_4o_azure(prompt)
    if(model_name == 'gemini2flash'):
        return gemini_2_flash(prompt)
    return 'input model does not exist'


