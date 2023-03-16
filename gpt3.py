import urllib.request
from urllib.error import HTTPError
import json
import time

last_request_time = float('-inf')

def predict(api_key, model_name, prompt, temperature=0, logprobs=5, n=1, stop=None, max_tokens=1024, echo=False, min_query_interval=None):
	chat = model_name.startswith('gpt-3.5-turbo')
	if chat:
		# TODO: for testing, perhaps its better to split this into a separate `predict_chat` function
		examples = prompt.split('Q:')
		qa_pairs = [example.split('A:') for example in examples[1:]]
		messages = [{'role':'system', 'content':'You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. Knowledge cutoff: 2021-09-01 Current date: 2023-03-14'}]
		for qa in qa_pairs[:-1]:
			messages.append({'role':'user', 'content':'Q: '+qa[0].strip()})
			messages.append({'role':'assistant', 'content':'A: '+qa[1].strip()})
		messages.append({'role':'user', 'content':'Q: '+qa_pairs[-1][0].strip()})
		logprobs = None

	def filter_answer(ans):
		if ans.startswith('A:'):
			ans = ans[len('A:'):]
		return ans

	# limit the rate of the queries
	if min_query_interval == None:
		if model_name in ['code-davinci-001', 'code-cushman-001', 'code-davinci-002']:
			min_query_interval = 10.0
		else:
			min_query_interval = 0.0
	global last_request_time
	current_request_time = time.monotonic()
	if current_request_time - last_request_time < min_query_interval:
		time.sleep(min_query_interval - (current_request_time - last_request_time))
	last_request_time = time.monotonic()

	header = { 'Content-Type' : 'application/json', 'Authorization' : 'Bearer ' + api_key }
	if chat:
		data = { 'model' : model_name, 'messages' : messages, 'temperature' : temperature }
	else:
		data = { 'model' : model_name, 'prompt' : prompt, 'temperature' : temperature, 'max_tokens' : max_tokens }
	if logprobs != None:
		data['logprobs'] = logprobs
	if stop != None:
		data['stop'] = stop
	if n != 1:
		data['n'] = n
	if echo:
		data['echo'] = True
	url = ('https://api.openai.com/v1/chat/completions' if chat else 'https://api.openai.com/v1/completions')
	request = urllib.request.Request(url, headers=header, data=json.dumps(data).encode())
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
					if chat:
						return [filter_answer(output['message']['content']) for output in outputs], None
					return [output['text'] for output in outputs], None
				if chat:
					return filter_answer(outputs[0]['message']['content']), None
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
