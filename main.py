from theory import *
from syntax import *
from proof import *
from random import choice, randrange
import numpy as np
from scipy.special import betaincinv
import argparse
import getpass

class Morphology(object):
	def __init__(self):
		self.plural_nouns = {}

	def add_noun(self, noun, plural):
		self.plural_nouns[noun] = plural

	def is_noun(self, word):
		return word in self.plural_nouns

	def to_plural(self, noun):
		return self.plural_nouns[noun]

morphology = Morphology()
morphology.add_noun("wumpus", "wumpuses")
morphology.add_noun("yumpus", "yumpuses")
morphology.add_noun("zumpus", "zumpuses")
morphology.add_noun("dumpus", "dumpuses")
morphology.add_noun("rompus", "rompuses")
morphology.add_noun("numpus", "numpuses")
morphology.add_noun("tumpus", "tumpuses")
morphology.add_noun("vumpus", "vumpuses")
morphology.add_noun("impus", "impuses")
morphology.add_noun("jompus", "jompuses")

config = OntologyConfig(max_child_count=1, generate_negation=True, generate_properties=True, stop_probability=0.3)

def generate_question(num_deduction_steps):
	if num_deduction_steps < 2:
		# `num_deduction_steps` includes the axiom step
		raise ValueError("num_deduction_steps must be at least 2.")
	available_concept_names = ["wumpus", "yumpus", "zumpus", "dumpus", "rompus", "numpus", "tumpus", "vumpus", "impus", "jompus"]
	available_entity_names = ["Fae", "Rex", "Sally", "Max", "Alex", "Sam", "Polly", "Stella", "Wren"]
	index = randrange(len(available_concept_names))
	distractor_concept = available_concept_names[index]
	del available_concept_names[index]

	config.stop_probability = 1 / (num_deduction_steps + 1)
	theory = generate_theory(
					available_concept_names,
					[["blue", "red", "brown", "orange"],
					 ["small", "large"],
					 ["metallic", "wooden", "luminous", "liquid"],
					 ["transparent", "opaque"],
					 ["nervous", "happy", "feisty", "shy"],
					 ["bright", "dull"],
					 ["sweet", "sour", "spicy", "bitter"],
					 ["floral", "fruity", "earthy"],
					 ["hot", "cold", "temperate"],
					 ["kind", "mean", "angry", "amenable", "aggressive"]],
					config)
	formulas = get_formulas(theory)
	sentences = []
	for formula in formulas:
		sentences.append(inflect(yield_tokens(formula_to_clause(formula, morphology)), end_punctuation='.'))

	(premise, conclusion, proof, num_steps) = generate_membership_question(theory, choice(available_entity_names), num_deduction_steps, False, True)
	if proof == None or num_steps != num_deduction_steps:
		return (None, None, None)
	proof_formulas = get_proof_intermediate_formulas(proof)

	distractor_lf = None
	if type(conclusion) == fol.FOLFuncApplication:
		# if the question is the form `x is A`, and there is an existing sentence of the form `every B is A`, then create a distractor sentence of the form `every D is not A`
		distractor_lf = fol.FOLForAll(1, fol.FOLIfThen(
				fol.FOLFuncApplication(distractor_concept, [fol.FOLVariable(1)]),
				fol.FOLNot(fol.FOLFuncApplication(conclusion.function, [fol.FOLVariable(1)]))
			))
	elif type(conclusion) == fol.FOLNot and type(conclusion.operand) == fol.FOLFuncApplication:
		# if the question is the form `x is not A`, and there is an existing sentence of the form `every B is not A`, then create a distractor sentence of the form `every D is A`
		distractor_lf = fol.FOLForAll(1, fol.FOLIfThen(
				fol.FOLFuncApplication(distractor_concept, [fol.FOLVariable(1)]),
				fol.FOLFuncApplication(conclusion.operand.function, [fol.FOLVariable(1)])
			))
	if distractor_lf != None:
		distractor_sentence = inflect(yield_tokens(formula_to_clause(distractor_lf, morphology)), end_punctuation='.')
		index = randrange(len(formulas) + 1)
		formulas.insert(index, distractor_lf)
		sentences.insert(index, distractor_sentence)

	expected_answer = True
	question = conclusion
	if choice([True, False]):
		# with some probability, negate the conclusion so the answer is false
		expected_answer = False
		if type(question) == fol.FOLNot:
			question = question.operand
		else:
			question = fol.FOLNot(question)

	question_text =  ' '.join(sentences)
	question_text += ' ' + inflect(yield_tokens(formula_to_clause(premise, morphology)), end_punctuation='.')
	question_text += ' True or false: ' + inflect(yield_tokens(formula_to_clause(question, morphology)), end_punctuation='.')

	# print the chain-of-thought and answer
	chain_of_thought = ''
	for proof_formula in proof_formulas:
		# find the sentence corresponding to this formula
		found_formula = False
		for i in range(len(formulas)):
			if formulas[i] == proof_formula:
				if len(chain_of_thought) != 0:
					chain_of_thought += ' '
				chain_of_thought += sentences[i]
				found_formula = True
				break
		if not found_formula:
			if len(chain_of_thought) != 0:
				chain_of_thought += ' '
			chain_of_thought += inflect(yield_tokens(formula_to_clause(proof_formula, morphology)), end_punctuation='.')
	return (question_text, chain_of_thought, str(expected_answer))

def print_output(str, log):
	log.write(str + '\n')
	print(str)

gpt_api_key = None

def evaluate_response(response, expected_answer):
	answer = expected_answer[(expected_answer.rfind(' ') + 1):]

	if answer == 'True':
		acceptable_answers = {'true', 't', 'yes', 'y', 'correct', 'right'}
	else:
		acceptable_answers = {'false', 'f', 'no', 'n', 'incorrect', 'wrong'}

	index = response.find('Q:')
	if index != -1:
		response = response[:index]
	while len(response) != 0 and response[-1].isspace():
		response = response[:-1]
	if response[(response.rfind(' ') + 1):].lower() in acceptable_answers:
		return 1.0
	else:
		return 0.0

def parse_log(log):
	trial = 0
	results = []
	resume_position = 0
	line_number = 0
	while True:
		# look for the next line beginning with 'Predicted answer:'
		line = log.readline()
		line_number += 1
		if not line:
			break # found the end of the file
		elif not line.startswith('Predicted answer:'):
			continue

		# read the predicted answer
		expected_answer = None
		predicted_answer = line[len('Predicted answer:'):]
		while True:
			line = log.readline()
			line_number += 1
			if not line:
				break # found the end of the file
			elif line.startswith('Expected answer: '):
				expected_answer = line[len('Expected answer: '):]
				break
			predicted_answer += line

		# read the expected answer
		mean = None
		while expected_answer is not None:
			line = log.readline()
			line_number += 1
			if not line:
				break # found the end of the file
			elif line.startswith('n: '):
				# read the summary statistics
				current_trial = int(line[len('n: '):line.index(',')])
				if current_trial != trial + 1:
					raise ValueError('Trial number is inconsistent on line ' + str(line_number))
				trial = current_trial
				normal_statistics = log.readline()
				if normal_statistics is not None:
					index = normal_statistics.find('mean: ')
					mean = float(normal_statistics[(index + len('mean: ')):normal_statistics.index(',')])
				log.readline() # consume the empty line separating each example
				line_number += 2
				resume_position = log.tell()
				break
			expected_answer += line

		# evaluate the correctness of this example
		if predicted_answer[-1] == '\n':
			predicted_answer = predicted_answer[:-1]
		if expected_answer[-1] == '\n':
			expected_answer = expected_answer[:-1]
		results.append(evaluate_response(predicted_answer, expected_answer))
		expected_mean = np.sum(results) / trial
		if mean == None or np.abs(mean - expected_mean) > 1.0e-9:
			raise ValueError('parse_log ERROR: The reported mean ({}) differs from the calculated mean ({}).'.format(mean, expected_mean))
	print('Resuming previous experiment at trial ' + str(trial + 1))
	return (trial, results, resume_position)

def run_experiment(model_name, model_size, num_proof_steps, num_fewshot_examples, num_trials, log_file, resume=False):
	global gpt_api_key
	if model_name == 'gpt3':
		if model_size.lower() != '175b':
			raise ValueError('model_size must be "175B" when model_name is "gpt3"')
		import gpt3
		if gpt_api_key == None:
			gpt_api_key = getpass.getpass(prompt='Enter OpenAI API Key:')
	elif model_name == 'opt':
		import opt
	elif model_name == 'unifiedqa':
		import unifiedqa
	elif model_name != 'dummy':
		raise ValueError('Unrecognized model_name "' + model_name + '"')

	if resume:
		log = open(log_file, "a+")
		log.seek(0)
		(trial, results, truncate_pos) = parse_log(log)
		log.truncate(truncate_pos)
	else:
		log = open(log_file, "w")
		trial = 0
		results = []

	while trial < num_trials:
		prompt = ''
		for i in range(num_fewshot_examples):
			while True:
				(question, chain_of_thought, answer) = generate_question(num_proof_steps)
				if question != None:
					break
			prompt += 'Q: ' + question + '\nA: ' + chain_of_thought + ' ' + answer + '\n\n'

		while True:
			(question, chain_of_thought, answer) = generate_question(num_proof_steps)
			if question != None:
				break
		prompt += 'Q: ' + question + '\nA:'
		print_output(prompt, log)
		if model_name == 'gpt3':
			response = gpt3.predict(gpt_api_key, prompt)
		elif model_name == 'opt':
			response = opt.predict(model_size, prompt)
		elif model_name == 'unifiedqa':
			response = unifiedqa.predict(model_size, prompt)
		elif model_name == 'dummy':
			response = ''
		print_output('\nPredicted answer:' + response, log)
		print_output('\nExpected answer: ' + chain_of_thought + ' ' + answer, log)

		results.append(evaluate_response(response, chain_of_thought + ' ' + answer))

		# compute the posterior beta parameters
		trial += 1
		alpha = np.sum(results) + 1
		beta = trial - np.sum(results) + 1
		print_output('n: ' + str(trial) + ', (beta prior) mean: ' + str(alpha/(alpha+beta)) + ', 95% lower bound: ' + str(betaincinv(alpha, beta, 0.025)) + ', 95% upper bound: ' + str(betaincinv(alpha, beta, 0.975)), log)
		mu = np.sum(results) / trial
		stddev = np.sqrt(mu*(1 - mu)/trial)
		print_output('  (normal approximation) mean: ' + str(mu) + ', 95% lower bound: ' + str(mu - 1.96*stddev) + ', 95% upper bound: ' + str(mu + 1.96*stddev) + '\n', log)
		log.flush()
	log.close()
	return results

parser = argparse.ArgumentParser()
parser.add_argument("--resume", action='store_true')
parser.add_argument("--model-name", type=str, required=True)
parser.add_argument("--model-size", type=str, required=True)
parser.add_argument("--num-trials", type=int, default=500)
parser.add_argument("--few-shot-examples", type=int, default=8)
args = parser.parse_args()

for hops in range(1,8+1):
	if args.model_name == 'gpt3':
		run_experiment("gpt3", args.model_size, 1 + hops, args.few_shot_examples, args.num_trials, "gpt_" + str(hops) + "hop.log", args.resume)
	elif args.model_name == 'opt':
		run_experiment("opt", args.model_size, 1 + hops, args.few_shot_examples, args.num_trials, "opt" + args.model_size.lower() + "_" + str(hops) + "hop.log", args.resume)
	elif args.model_name == 'unifiedqa':
		run_experiment("unifiedqa", args.model_size.lower(), 1 + hops, args.few_shot_examples, args.num_trials, "unifiedqa_" + args.model_size.lower() + "_" + str(hops) + "hop.log", args.resume)
	else:
		print('ERROR: --model-name must be either ' + str({'gpt3', 'opt', 'unifiedqa'}))
		break
