models = {}
tokenizers = {}

use_alpa = False

def predict(model_size, prompt, query_server=None):
	if query_server != None:
		import urllib.request
		import json

		header = { 'Content-Type' : 'application/json' }
		data = { 'prompt' : prompt, 'temperature' : 0, 'max_tokens' : 256, 'top_p' : 0 }
		request = urllib.request.Request(query_server + '/completions', headers=header, data=json.dumps(data).encode())
		with urllib.request.urlopen(request) as response:
			outputs = json.loads(response.read())['choices']
			if len(outputs) != 1:
				print('opt.predict ERROR: The length of the `choices` field is not 1.')
				return None
			return outputs[0]['text']
		return None

	else:
		global models, tokenizers
		model_size = model_size.lower()

		if use_alpa:
			from transformers import AutoTokenizer
			from opt_serving.model.wrapper import get_model

			global tokenizer
			try:
				tokenizer
			except NameError:
				import ray
				ray.init()
				tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
				tokenizer.add_bos_token = False

			if model_size not in models:
				models[model_size] = get_model(model_name="alpa/opt-" + model_size,
											   device="cuda",
											   path="/scratch/as17582/opt_weights/")
			model = models[model_size]

			input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
			import torch
			torch.set_printoptions(profile="full")
			print('INFO: input_ids: {}, min_length: {}, max_new_tokens: {}, temperature: {}, do_sample: {}, top_p: {}'.format(input_ids, None, 256, None, False, None))
			torch.set_printoptions(profile="default")
			output = model.generate(input_ids=input_ids, max_new_tokens=256, do_sample=False)
			generated_string = tokenizer.batch_decode(output, skip_special_tokens=True)

		else:
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
