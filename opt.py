from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

models = {}
tokenizers = {}

def predict(model_size, prompt):
	global models, tokenizers
	model_size = model_size.lower()

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
