import os
from openai import AzureOpenAI

def gpt_4o_azure(input, 
                            key_path='/Users/yiminglin/Documents/Codebase/api_keys/azure_openai/azuregpt4o.txt',
                            max_tokens=800,
                            temperature=0):
    """
    Get response from Azure OpenAI API.
    
    Args:
        prompt (str): The text prompt to send to the model
        key_path (str): Path to the API key file
        max_tokens (int): Maximum tokens for response
        temperature (float): Response randomness (0-1)
        
    Returns:
        str: The response content from the model
    """
    prompt = input[0] + input[1] 
    # Read API key
    with open(key_path, 'r') as f:
        api_key = f.read().strip()
    
    # Initialize client
    client = AzureOpenAI(
        azure_endpoint=os.getenv("ENDPOINT_URL", "https://text-db.openai.azure.com/"),
        api_key=api_key,
        api_version="2025-01-01-preview",
    )
    
    # Generate response
    completion = client.chat.completions.create(
        model=os.getenv("DEPLOYMENT_NAME", "gpt-4o"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stream=False
    )
    
    return completion.choices[0].message.content

