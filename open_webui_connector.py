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
        
        if isinstance(model, Models): 
            self.model = model.value 
        else: 
            self.model = model 

        self.system_prompt = system_prompt

        self.history_len = history_len
        self.history = []

    def chat(self, message, use_history=True, use_system_prompt=True, remember_chat=True): 
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
        ret = response.json() 

        if remember_chat: 
            self._remember_chat(messages[-1], ret['choices'][0]['message'])

        return ret 

    def clear_history(self): 
        self.history = []

    def _remember_chat(self, q,a):
        '''
        Logs user's messages and llm's responses 
        '''
        self.history += [q,a]
        self.history = self.history[-self.history_len:]
