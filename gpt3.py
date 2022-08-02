import urllib.request
from urllib.error import HTTPError
import json

def predict(api_key, model_name, prompt):
	header = { 'Content-Type' : 'application/json', 'Authorization' : 'Bearer ' + api_key }
	data = { 'model' : model_name, 'prompt' : prompt, 'temperature' : 0, 'max_tokens' : 256 }
	request = urllib.request.Request('https://api.openai.com/v1/completions', headers=header, data=json.dumps(data).encode())
	try:
		with urllib.request.urlopen(request) as response:
			outputs = json.loads(response.read())['choices']
			if len(outputs) != 1:
				print('gpt3.predict ERROR: The length of the `choices` field is not 1.')
				return None
			if outputs[0]['finish_reason'] != 'stop':
				print('gpt3.predict WARNING: GPT3 stopped predicting tokens for a reason other than predicting a STOP token.')
			return outputs[0]['text']
	except HTTPError as e:
		if e.code != 400:
			raise e
	return None
