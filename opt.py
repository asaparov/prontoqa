import ray
ray.init()

import torch

from transformers import AutoTokenizer
from examples.opt_serving.model.wrapper import get_model

# Load the tokenizer. We have to use the 30B version because
# other versions have some issues. The 30B version works for all OPT models.
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
tokenizer.add_bos_token = False

models = {}

def predict(model_size, prompt, path_to_weights):
	global models
	model_size = model_size.lower()

	# Load the model
	if model_size not in models:
		models[model_size] =  get_model(model_name="alpa/opt-" + model_size,
										device="cuda",
										path=path_to_weights)
	model = models[model_size]

	# Generate
	input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
	output = model.generate(input_ids=input_ids, max_new_tokens=256, do_sample=False)
	generated_string = tokenizer.batch_decode(output)

	if len(generated_string) != 1:
		print("WARNING: len(generated_string) is not 1.")
	return generated_string[0][len(prompt):]
