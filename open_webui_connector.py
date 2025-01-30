from enum import Enum 
import requests as r 

URL = 'http://orion.ece.seas.gwu.edu:3000/api/'

class Models(Enum): 
    SEVEN_B = 'deepseek-r1:latest'
    FOURTEEN_B = 'deepseek-r1:14b-qwen-distill-fp16'

class LLMConnector: 
    def __init__(self, url, api, model, history_len=10, system_prompt=''): 
        self.url = url 
        self.api = api 
        self.model = model 
        self.system_prompt = system_prompt

        self.history_len = history_len
        self.history = []

    def chat(self, message, use_history=True, use_system_prompt=True): 
        endpoint = self.url + 'chat/completions'
        headers = {
            'Authorization': f'Bearer {self.api}',
            'Content-Type': 'application/json'
        }

        if use_history: 
            messages = self.history + [{'role': 'user', 'content': message}]
        else: 
            messages = [{'role': 'user', 'content': message}]

        # Careful w this. I think some models use a different keyword for system prompt now
        if self.system_prompt and use_system_prompt: 
            messages = [{'role': 'system', 'content': self.system_prompt}] + messages

        payload = {
            'model': self.model, 
            'messages': messages 
        }

        response = r.post(endpoint, headers=headers, json=payload)
        return response.json() 