import enum
import json
from json_repair import repair_json
import re
import requests
import yaml

class PromptType(str, enum.Enum):
    HEADER = 'header'
    REFERENCES = 'references'

class LLMExtractor:
    def __init__(self, url, model_name, path_to_prompts = 'data/prompts/prompts.yaml'):
        self.url = url
        self.model_name = model_name
        with open(path_to_prompts, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)

    def extract(self, type, text):
        prompt = self.prompts[type]['text'].replace('{{input_text}}', text)
        output = self._send_request(prompt)
        return output

    def _send_request(self, prompt, think = False, stream = False):
        payload = {
            'model': self.model_name,
            'prompt': prompt,
            'think': think,
            'stream': stream,
            'options': {
                'temperature': 0.0
            }
        }

        try:
            response = requests.post(self.url, json=payload, timeout=120)
            response.raise_for_status()
            raw_result = response.json()['response']
            result = re.sub(r'\n?```\w{0,4}\n?', '', raw_result)
            fixed_result = repair_json(result)
            output = json.loads(fixed_result)
            return output
        except requests.exceptions.HTTPError as e:
            print('Ошибка HTTP: ', e)
            print('Ответ сервера: ', response.text)
        except requests.exceptions.RequestException as e:
            print('Ошибка запроса: ', e)
        except json.decoder.JSONDecodeError as e:
            print('Ошибка JSON: ', e)
