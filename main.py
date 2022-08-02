from theory import *
from syntax import *
from proof import *
from random import choice, randrange, shuffle, seed
import numpy as np
from scipy.special import betaincinv
import argparse
import getpass
import re

class Morphology(object):
	def __init__(self):
		self.plural_nouns = {}
		self.reverse_plural_nouns = {}
		self.proper_nouns = []

	def add_noun(self, noun, plural):
		self.plural_nouns[noun] = plural
		self.reverse_plural_nouns[plural] = noun

	def add_proper_noun(self, noun):
		self.proper_nouns.append(noun)

	def is_noun(self, word):
		return word in self.plural_nouns

	def is_plural_noun(self, word):
		return word in self.reverse_plural_nouns

	def is_proper_noun(self, word):
		return word in self.proper_nouns

	def to_plural(self, noun):
		return self.plural_nouns[noun]

	def to_root(self, plural):
		return self.reverse_plural_nouns[plural]

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

morphology.add_noun("animal", "animals")
morphology.add_noun("bilaterian", "bilaterians")
morphology.add_noun("chordate", "chordates")
morphology.add_noun("vertebrate", "vertebrates")
morphology.add_noun("mammal", "mammals")
morphology.add_noun("carnivore", "carnivores")
morphology.add_noun("feline", "felines")
morphology.add_noun("cat", "cats")
morphology.add_noun("tabby", "tabbies")
morphology.add_noun("bacteria", "bacteria")
morphology.add_noun("insect", "insects")
morphology.add_noun("reptile", "reptiles")
morphology.add_noun("snake", "snakes")
morphology.add_noun("cow", "cows")
morphology.add_noun("sheep", "sheep")
morphology.add_noun("dog", "dogs")

morphology.add_noun("invertebrate", "invertebrates")
morphology.add_noun("protostome", "protostomes")
morphology.add_noun("arthropod", "arthropods")
morphology.add_noun("lepidopteran", "lepidopterans")
morphology.add_noun("butterfly", "butterflies")
morphology.add_noun("painted lady", "painted ladies")
morphology.add_noun("mullosc", "mulloscs")
morphology.add_noun("nematode", "nematodes")
morphology.add_noun("whale", "whales")
morphology.add_noun("crustacean", "crustaceans")
morphology.add_noun("spider", "spiders")
morphology.add_noun("ant", "ants")
morphology.add_noun("moth", "moths")

morphology.add_noun("plant", "plants")
morphology.add_noun("vascular plant", "vascular plants")
morphology.add_noun("angiosperm", "angiosperms")
morphology.add_noun("eudicot", "eudicots")
morphology.add_noun("rosid", "rosids")
morphology.add_noun("rose", "roses")
morphology.add_noun("fish", "fishes")
morphology.add_noun("whale", "whales")
morphology.add_noun("moss", "mosses")
morphology.add_noun("conifer", "conifers")
morphology.add_noun("cabbage", "cabbages")
morphology.add_noun("asterid", "asterids")
morphology.add_noun("carrot", "carrots")

morphology.add_noun("number", "numbers")
morphology.add_noun("real number", "real numbers")
morphology.add_noun("integer", "integers")
morphology.add_noun("natural number", "natural numbers")
morphology.add_noun("prime number", "prime numbers")
morphology.add_noun("Mersenne prime", "Mersenne primes")
morphology.add_noun("function", "functions")
morphology.add_noun("real number", "real numbers")
morphology.add_noun("imaginary number", "imaginary numbers")
morphology.add_noun("complex number", "complex numbers")
morphology.add_noun("fraction", "fractions")
morphology.add_noun("negative number", "negative numbers")
morphology.add_noun("composite number", "composite numbers")
morphology.add_noun("even number", "even numbers")

# for being able to parse overgenerated sentences
morphology.add_noun("person", "people")
morphology.add_noun("object", "objects")
morphology.add_noun("food", "food")

available_entity_names = ["Fae", "Rex", "Sally", "Max", "Alex", "Sam", "Polly", "Stella", "Wren"]
for name in available_entity_names:
	morphology.add_proper_noun(name)

config = OntologyConfig(max_child_count=1, generate_negation=True, generate_properties=True, stop_probability=0.3)

def generate_question(num_deduction_steps, formula_ordering="postorder", ontology="fictional", add_distractor=True):
	if num_deduction_steps < 2:
		# `num_deduction_steps` includes the axiom step
		raise ValueError("num_deduction_steps must be at least 2.")
	if ontology == "true":
		(theory, selected_entity, distractor_map) = sample_real_ontology(available_entity_names, num_deduction_steps)
	else:
		if ontology == "false":
			r = randrange(3)
			if r == 0:
				available_concept_names = ["animal", "vertebrate", "mammal", "carnivore", "feline", "cat", "dog", "sheep", "cow", "snake"]
			elif r == 1:
				available_concept_names = ["animal", "invertebrate", "arthropod", "insect", "lepidopteran", "butterfly", "moth", "ant", "spider", "crustacean"]
			else:
				available_concept_names = ["number", "real number", "integer", "natural number", "prime number", "Mersenne prime", "even number", "composite number", "negative number", "fraction"]
		else:
			available_concept_names = ["wumpus", "yumpus", "zumpus", "dumpus", "rompus", "numpus", "tumpus", "vumpus", "impus", "jompus"]
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
		selected_entity = choice(available_entity_names)

	if formula_ordering == "random":
		formulas = get_formulas(theory, ordering="preorder")
		shuffle(formulas)
	else:
		formulas = get_formulas(theory, ordering=formula_ordering)
	sentences = []
	for formula in formulas:
		sentences.append(inflect(yield_tokens(formula_to_clause(formula, morphology)), end_punctuation='.'))
		parsed_lf = parse_sentence(sentences[-1][:-1], morphology, False)
		if parsed_lf != formula:
			raise Exception("Parsed sentence does not match original logical form:\n  Sentence: \"{}\"\n  Original: {}\n  Parsed:   {}".format(sentences[-1], fol.fol_to_tptp(formula), fol.fol_to_tptp(parsed_lf)))

	(premise, conclusion, proof, num_steps) = generate_membership_question(theory, selected_entity, num_deduction_steps, False, True)
	if proof == None or num_steps != num_deduction_steps:
		return (None, None, None, None)
	proof_formulas = get_proof_intermediate_formulas(proof)

	distractor_lf = None
	if type(conclusion) == fol.FOLFuncApplication:
		# if the question is the form `x is A`, and there is an existing sentence of the form `every B is A`, then create a distractor sentence of the form `every D is not A`
		if ontology == "true":
			distractor_concept = distractor_map[conclusion.function]
		distractor_lf = fol.FOLForAll(1, fol.FOLIfThen(
				fol.FOLFuncApplication(distractor_concept, [fol.FOLVariable(1)]),
				fol.FOLNot(fol.FOLFuncApplication(conclusion.function, [fol.FOLVariable(1)]))
			))
	elif type(conclusion) == fol.FOLNot and type(conclusion.operand) == fol.FOLFuncApplication:
		# if the question is the form `x is not A`, and there is an existing sentence of the form `every B is not A`, then create a distractor sentence of the form `every D is A`
		if ontology == "true":
			distractor_concept = distractor_map[conclusion.operand.function]
		distractor_lf = fol.FOLForAll(1, fol.FOLIfThen(
				fol.FOLFuncApplication(distractor_concept, [fol.FOLVariable(1)]),
				fol.FOLFuncApplication(conclusion.operand.function, [fol.FOLVariable(1)])
			))
	if add_distractor and distractor_lf != None:
		distractor_sentence = inflect(yield_tokens(formula_to_clause(distractor_lf, morphology)), end_punctuation='.')
		index = randrange(len(formulas) + 1)
		formulas.insert(index, distractor_lf)
		sentences.insert(index, distractor_sentence)
		parsed_lf = parse_sentence(distractor_sentence[:-1], morphology, False)
		if parsed_lf != distractor_lf:
			raise Exception("Parsed sentence does not match original logical form:\n  Sentence: \"{}\"\n  Original: {}\n  Parsed:   {}".format(distractor_sentence, fol.fol_to_tptp(distractor_lf), fol.fol_to_tptp(parsed_lf)))

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
	return (question_text, formulas + [premise], chain_of_thought, str(expected_answer))

def print_output(str, log):
	log.write(str + '\n')
	print(str)

gpt_api_key = None
opt_server = None
bad_patterns = []

with open('bad_patterns.txt', 'r') as reader:
	for line in reader:
		bad_patterns.append(re.compile(line.strip()))

def parse_reasoning(response):
	# first tokenize the response
	tokens = re.findall(r"[\w\-]+|[^\w\s]", response)

	start = 0
	lfs = []
	for end in range(len(tokens)):
		if tokens[end] == '.':
			# parse this sentence
			lf = parse_sentence(tokens[start:end], morphology, False)
			if lf == None:
				# check that this sentence is in the list of expected bad patterns
				has_bad_pattern = False
				for bad_pattern in bad_patterns:
					if bad_pattern.match(' '.join(tokens[start:end])) != None:
						has_bad_pattern = True
						break
				if not has_bad_pattern:
					raise Exception("Unable to parse sentence \"{}.\"".format(' '.join(tokens[start:end])))
			lfs.append(lf)
			start = end + 1
	return lfs

def evaluate_response(response, expected_answer, axioms):
	acceptable_answers = {
		'True':{'true', 't', 'yes', 'y', 'correct', 'right'},
		'False':{'false', 'f', 'no', 'n', 'incorrect', 'wrong'}
	}

	index = response.find('Q:')
	if index != -1:
		response = response[:index]
	while len(response) != 0 and response[-1].isspace():
		response = response[:-1]
	label_index = response.rfind(' ')
	last_period_index = response.rfind('.')
	if last_period_index > label_index:
		# check if the period comes after a label (e.g. "True." or "no.")
		possible_label = response[label_index:last_period_index].strip().lower()
		if possible_label not in acceptable_answers['True'] and possible_label not in acceptable_answers['False']:
			label_index = last_period_index + 1

	expected_label_index = expected_answer.rfind(' ')
	expected_proof = parse_reasoning(expected_answer[:expected_label_index])

	# evaluate the proof
	proof = parse_reasoning(response[:label_index])
	correct_steps = []
	correct_and_useful_steps = []
	redundant_steps = []
	unparseable_steps = []
	incorrect_steps = []
	wrong_branch_steps = []
	found_conclusion = False
	for i in range(len(proof)):
		proof_step = proof[i]
		if proof_step == None:
			unparseable_steps.append(i)
			incorrect_steps.append(i)
			continue

		is_useful = (proof_step in expected_proof)
		is_conclusion = (proof_step == expected_proof[-1])

		# check if we've gone down the wrong branch
		if not is_useful and proof_step in axioms and type(proof_step) == fol.FOLForAll:
			wrong_branch_steps.append(i)

		if proof_step in proof[:i]:
			redundant_steps.append(i)
			continue

		# check if this is an axiom
		if proof_step in axioms:
			correct_steps.append(i)
			if is_useful:
				correct_and_useful_steps.append(i)
			if is_conclusion:
				found_conclusion = True
			continue

		# check if this is an instance of UNIVERSAL_INSTANTIATION
		is_valid = False
		for prev_step in proof[:i]:
			first_var_map = {}
			second_var_map = {}
			if type(prev_step) == fol.FOLForAll and type(prev_step.operand) == fol.FOLIfThen and fol.unify(proof_step, prev_step.operand.consequent, first_var_map, second_var_map):
				other_premise = fol.substitute(prev_step.operand.antecedent, fol.FOLVariable(prev_step.variable), second_var_map[prev_step.variable])
				if other_premise in proof[:i]:
					is_valid = True
					break
		if is_valid:
			correct_steps.append(i)
			if is_useful:
				correct_and_useful_steps.append(i)
			if is_conclusion:
				found_conclusion = True
			continue

		# we can't find a matching deduction rule, so label this step as incorrect
		incorrect_steps.append(i)

	# TODO: for debugging; delete this
	if len(wrong_branch_steps) != 0 and found_conclusion:
		print('DEBUG: delete this')

	# evaluate the label
	answer = expected_answer[(expected_label_index + 1):]

	if (response[(label_index + 1):].lower() in acceptable_answers[answer]):
		label_correctness = 1.0
	else:
		label_correctness = 0.0
	return (label_correctness, correct_steps, correct_and_useful_steps, redundant_steps, unparseable_steps, wrong_branch_steps, incorrect_steps, found_conclusion)

def parse_log(log):
	trial = 0
	too_long_responses = 0
	results = []
	label_results = []
	resume_position = 0
	line_number = 0
	last_question = None
	while True:
		# look for the next line beginning with 'Predicted answer:'
		line = log.readline()
		line_number += 1
		if not line:
			break # found the end of the file
		elif line.startswith('Q: '):
			last_question = line[len('Q: '):]
			continue
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
		found_summary = False
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
				found_summary = True
				break
			expected_answer += line

		if not found_summary:
			break

		# evaluate the correctness of this example
		if predicted_answer[-1] == '\n':
			predicted_answer = predicted_answer[:-1]
		if expected_answer[-1] == '\n':
			expected_answer = expected_answer[:-1]
		if 'CONTEXT_TOO_LONG_ERROR' in predicted_answer:
			too_long_responses += 1
		else:
			last_question = last_question[:last_question.index('True or false:')]
			result = evaluate_response(predicted_answer, expected_answer, parse_reasoning(last_question))
			results.append(result)
			(label, correct_steps, correct_and_useful_steps, redundant_steps, unparseable_steps, wrong_branch_steps, incorrect_steps, found_conclusion) = result
			label_results.append(label)
		expected_mean = np.sum(label_results) / (trial - too_long_responses)
		if mean == None or np.abs(mean - expected_mean) > 1.0e-9:
			raise ValueError('parse_log ERROR: The reported mean ({}) differs from the calculated mean ({}).'.format(mean, expected_mean))
	return (trial, too_long_responses, results, label_results, resume_position)

def run_experiment(model_name, model_size, num_proof_steps, test_num_proof_steps, num_fewshot_examples, num_trials, repetitions_per_test, log_file, formula_ordering="postorder", ontology="fictional", add_distractor=True, resume=False, random_seed=62471893):
	global gpt_api_key
	if model_name == 'gpt3':
		import gpt3
		if gpt_api_key == None:
			gpt_api_key = getpass.getpass(prompt='Enter OpenAI API Key:')
	elif model_name == 'opt':
		import opt
	elif model_name == 'unifiedqa':
		import unifiedqa
	elif model_name != 'dummy':
		raise ValueError('Unrecognized model_name "' + model_name + '"')

	# set the random seed for reproducibility
	seed(random_seed)
	np.random.seed(random_seed)

	trial = 0
	too_long_responses = 0
	if resume:
		log = open(log_file, "a+")
		log.seek(0)
		(resume_trial, too_long_responses, results, label_results, truncate_pos) = parse_log(log)
		print('Resuming previous experiment at trial ' + str(resume_trial + 1))
		log.truncate(truncate_pos)
	else:
		log = open(log_file, "w")
		resume_trial = 0
		label_results = []

	while trial < num_trials * repetitions_per_test:
		for t in range(repetitions_per_test):
			prompt = ''
			for i in range(num_fewshot_examples):
				while True:
					(question, _, chain_of_thought, answer) = generate_question(num_proof_steps, formula_ordering, ontology, add_distractor)
					if question != None:
						break
				prompt += 'Q: ' + question + '\nA: ' + chain_of_thought + ' ' + answer + '\n\n'

			if t == 0:
				while True:
					test_question = generate_question(test_num_proof_steps, formula_ordering, ontology, add_distractor)
					(question, question_lfs, chain_of_thought, answer) = test_question
					if question != None:
						break
			else:
				# re-use the same test question from the first sub-iteration
				(question, question_lfs, chain_of_thought, answer) = test_question

			trial += 1
			if trial <= resume_trial:
				continue

			prompt += 'Q: ' + question + '\nA:'
			print_output(prompt, log)
			try_num = 0
			while True:
				try:
					if model_name == 'gpt3':
						response = gpt3.predict(gpt_api_key, model_size, prompt)
					elif model_name == 'opt':
						response = opt.predict(model_size, prompt, opt_server)
					elif model_name == 'unifiedqa':
						response = unifiedqa.predict(model_size, prompt)
					elif model_name == 'dummy':
						response = ''
					break
				except RuntimeError:
					try_num += 1
					if try_num == 5:
						raise
					print("Encountered runtime error. This may be due to CUDA instability. Trying again (try \#{})...".format(try_num + 1))
					continue
			if response == None:
				print('WARNING: Context has too many tokens for this model. Skipping question...')
				print_output('\nPredicted answer: CONTEXT_TOO_LONG_ERROR', log)
			else:
				print_output('\nPredicted answer:' + response, log)
			print_output('\nExpected answer: ' + chain_of_thought + ' ' + answer, log)

			if response == None:
				too_long_responses += 1
			else:
				result = evaluate_response(response, chain_of_thought + ' ' + answer, question_lfs)
				(label, correct_steps, correct_and_useful_steps, redundant_steps, unparseable_steps, wrong_branch_steps, incorrect_steps, found_conclusion) = result
				label_results.append(label)

			# compute the posterior beta parameters
			alpha = np.sum(label_results) + 1
			beta = (trial - too_long_responses) - np.sum(label_results) + 1
			print_output('n: ' + str(trial) + ', (beta prior) mean: ' + str(alpha/(alpha+beta)) + ', 95% lower bound: ' + str(betaincinv(alpha, beta, 0.025)) + ', 95% upper bound: ' + str(betaincinv(alpha, beta, 0.975)), log)
			mu = np.sum(label_results) / (trial - too_long_responses)
			stddev = np.sqrt(mu*(1 - mu)/(trial - too_long_responses))
			print_output('  (normal approximation) mean: ' + str(mu) + ', 95% lower bound: ' + str(mu - 1.96*stddev) + ', 95% upper bound: ' + str(mu + 1.96*stddev) + '\n', log)
			log.flush()
	log.close()
	return label_results

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--resume", action='store_true')
	parser.add_argument("--model-name", type=str, required=True)
	parser.add_argument("--model-size", type=str, required=True)
	parser.add_argument("--ordering", type=str, default="postorder", choices=["postorder", "preorder", "random"])
	parser.add_argument("--num-trials", type=int, default=500)
	parser.add_argument("--few-shot-examples", type=int, default=8)
	parser.add_argument("--ontology", type=str, default="fictional", choices=["fictional", "true", "false"])
	parser.add_argument("--opt-server", type=str, default=None)
	parser.add_argument("--no-distractor", action='store_true')
	parser.add_argument("--api-key", type=str, default=None)
	parser.add_argument("--min-hops", type=int, default=1)
	parser.add_argument("--max-hops", type=int, default=8)
	parser.add_argument("--hops-skip", type=int, default=1)
	parser.add_argument("--test-hops-diff", type=int, default=0)
	parser.add_argument("--repetitions-per-test", type=int, default=1)
	parser.add_argument("--seed", type=int, default=62471893)
	args = parser.parse_args()

	opt_server = args.opt_server
	gpt_api_key = args.api_key

	for hops in range(args.min_hops, args.max_hops+1, args.hops_skip):
		log_suffix = '_' + str(hops) + 'hop'
		if args.test_hops_diff != 0:
			log_suffix += '_' + str(hops + args.test_hops_diff) + 'testhops'
		if args.ordering != 'postorder':
			log_suffix += '_' + args.ordering
		if args.ontology == "true":
			log_suffix += '_trueontology'
		elif args.ontology == "false":
			log_suffix += '_falseontology'
		if args.no_distractor:
			log_suffix += '_nodistractor'
		if args.repetitions_per_test != 1:
			log_suffix += '_' + str(args.repetitions_per_test) + 'repetitions'
		if args.seed != 62471893:
			log_suffix += '_seed' + str(args.seed)
		if args.model_name == 'gpt3':
			run_experiment("gpt3", args.model_size, 1 + hops, 1 + hops + args.test_hops_diff, args.few_shot_examples, args.num_trials, args.repetitions_per_test, "gpt_" + args.model_size.lower().replace('-', '') + log_suffix + ".log", args.ordering, args.ontology, not args.no_distractor, args.resume, args.seed)
		elif args.model_name == 'opt':
			run_experiment("opt", args.model_size, 1 + hops, 1 + hops + args.test_hops_diff, args.few_shot_examples, args.num_trials, args.repetitions_per_test, "opt" + args.model_size.lower() + log_suffix + ".log", args.ordering, args.ontology, not args.no_distractor, args.resume, args.seed)
		elif args.model_name == 'unifiedqa':
			run_experiment("unifiedqa", args.model_size.lower(), 1 + hops, 1 + hops + args.test_hops_diff, args.few_shot_examples, args.num_trials, args.repetitions_per_test, "unifiedqa_" + args.model_size.lower() + log_suffix + ".log", args.ordering, args.ontology, not args.no_distractor, args.resume, args.seed)
		elif args.model_name == 'dummy':
			run_experiment("dummy", args.model_size, 1 + hops, 1 + hops + args.test_hops_diff, args.few_shot_examples, args.num_trials, args.repetitions_per_test, "dummy" + log_suffix + ".log", args.ordering, args.ontology, not args.no_distractor, args.resume, args.seed)
		else:
			print('ERROR: --model-name must be either ' + str({'gpt3', 'opt', 'unifiedqa', 'dummy'}))
			break
