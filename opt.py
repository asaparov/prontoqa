models = {}
tokenizers = {}

def predict(model_size, prompt, query_server=None):
	if query_server != None:
		import urllib.request
		import json

		header = { 'Content-Type' : 'application/json' }
		data = { 'prompt' : prompt, 'temperature' : 0, 'max_tokens' : 256 }
		request = urllib.request.Request(query_server + '/completions', headers=header, data=json.dumps(data).encode())
		with urllib.request.urlopen(request) as response:
			outputs = json.loads(response.read())['choices']
			if len(outputs) != 1:
				print('opt.predict ERROR: The length of the `choices` field is not 1.')
				return None
			if outputs[0]['finish_reason'] != 'stop':
				print('opt.predict WARNING: GPT3 stopped predicting tokens for a reason other than predicting a STOP token.')
			return outputs[0]['text']
		return None

	else:
		global models, tokenizers
		model_size = model_size.lower()

		from transformers import AutoTokenizer, AutoModelForCausalLM
		import torch

		# Load the model
		if model_size not in models:
			tokenizers[model_size] = AutoTokenizer.from_pretrained("facebook/opt-" + model_size, use_fast=False)
			models[model_size] = AutoModelForCausalLM.from_pretrained("facebook/opt-" + model_size, torch_dtype=torch.float16).cuda()
		tokenizer = tokenizers[model_size]
		model = models[model_size]

		# Generate
		input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
		output = model.generate(input_ids=input_ids, max_new_tokens=256, do_sample=False)
		generated_string = tokenizer.batch_decode(output)

		if len(generated_string) != 1:
			print("WARNING: len(generated_string) is not 1.")
		return generated_string[0][len(prompt):]
