from theory import *
from syntax import *
from proof import *
from random import choice, randrange
import numpy as np
from scipy.special import betaincinv
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

def run_experiment(model_name, model_size, num_proof_steps, num_fewshot_examples, num_trials, log_file):
	global gpt_api_key
	if model_name == 'gpt3':
		if model_size.lower() != '175b':
			raise ValueError('model_size must be "175B" when model_name is "gpt3"')
		import gpt3
		if gpt_api_key == None:
			gpt_api_key = getpass.getpass(prompt='Enter OpenAI API Key:')
	elif model_name == 'opt':
		import opt
	elif model_name != 'dummy':
		raise ValueError('Unrecognized model_name "' + model_name + '"')

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
			response = opt.predict(model_size, prompt, "/scratch/as17582/opt_weights/")
		elif model_name == 'dummy':
			response = ''
		print_output('\nPredicted answer:' + response, log)
		print_output('\nExpected answer: ' + chain_of_thought + ' ' + answer, log)

		if answer == 'True':
			acceptable_answers = {'true', 't', 'yes', 'y', 'correct', 'right'}
		else:
			acceptable_answers = {'false', 'f', 'no', 'n', 'incorrect', 'wrong'}

		trial += 1
		index = response.find('Q:')
		if index != -1:
			response = response[:index]
		while len(response) != 0 and response[-1].isspace():
			response = response[:-1]
		if response[(response.rfind(' ') + 1):].lower() in acceptable_answers:
			results.append(1.0)
		else:
			results.append(0.0)

		# compute the posterior beta parameters
		alpha = np.sum(results) + 1
		beta = trial - np.sum(results) + 1
		print_output('n: ' + str(trial) + ', (beta prior) mean: ' + str(alpha/(alpha+beta)) + ', 95% lower bound: ' + str(betaincinv(alpha, beta, 0.025)) + ', 95% upper bound: ' + str(betaincinv(alpha, beta, 0.975)), log)
		mu = np.sum(results) / trial
		stddev = np.sqrt(mu*(1 - mu)/trial)
		print_output('  (normal approximation) mean: ' + str(mu) + ', 95% lower bound: ' + str(mu - 1.96*stddev) + ', 95% upper bound: ' + str(mu + 1.96*stddev) + '\n', log)
	log.close()
	return results

#run_experiment("gpt3", "175B", 2, 8, 10, "gpt_1hop.log")
#run_experiment("opt", "30B", 2, 8, 500, "opt30b_1hop.log")
run_experiment("dummy", '', 2, 8, 1, "dummy_1hop.log")
