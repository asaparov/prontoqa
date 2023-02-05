import urllib.request
from urllib.error import HTTPError
import json

def predict(api_key, model_name, prompt, temperature=0, logprobs=5, n=1):
	header = { 'Content-Type' : 'application/json', 'Authorization' : 'Bearer ' + api_key }
	data = { 'model' : model_name, 'prompt' : prompt, 'temperature' : temperature, 'max_tokens' : 1024 }
	if logprobs != None:
		data['logprobs'] = logprobs
	if n != 1:
		data['n'] = n
	request = urllib.request.Request('https://api.openai.com/v1/completions', headers=header, data=json.dumps(data).encode())
	try:
		with urllib.request.urlopen(request) as response:
			outputs = json.loads(response.read())['choices']
			if len(outputs) != n:
				print('gpt3.predict ERROR: The length of the `choices` field is not `n`.')
				return None, None
			if outputs[0]['finish_reason'] != 'stop':
				print('gpt3.predict WARNING: GPT3 stopped predicting tokens for a reason other than predicting a STOP token.')
			if logprobs == None:
				if n != 1:
					return [output['text'] for output in outputs], None
				return outputs[0]['text'], None
			else:
				results = []
				for i in range(n):
					results.append((outputs[i]['text'], outputs[i]['logprobs']))
				if n == 1:
					return results[0]
				return results
	except HTTPError as e:
		if e.code != 400:
			raise e
	return None, None
