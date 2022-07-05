from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizers = {}
models = {}

def predict(model_size, prompt):
	global tokenizers, models
	model_size = model_size.lower()
	if model_size[0:2] == 'v2':
		model_size = model_size + '-1251000'

	if model_size not in models:
		tokenizers[model_size] = T5Tokenizer.from_pretrained("allenai/unifiedqa-" + model_size)
		models[model_size] = T5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-" + model_size).cuda()
	tokenizer = tokenizers[model_size]
	model = models[model_size]

	input_ids = tokenizer.encode(prompt.replace('\n', ' \\n '), return_tensors="pt").cuda()
	res = model.generate(input_ids, max_new_tokens=256, do_sample=False)
	generated_string = tokenizer.batch_decode(res, skip_special_tokens=True)

	if len(generated_string) != 1:
		print("WARNING: len(generated_string) is not 1.")
	return generated_string[0]
