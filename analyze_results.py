import numpy as np
from run_experiment import parse_log, parse_response, evaluate_response, parse_reasoning
from syntax import UnableToParseError, AmbiguousParseError
from sys import argv, exit

# see https://www.mikulskibartosz.name/wilson-score-in-python-example/
def wilson_conf_interval(p, n, z=1.96):
	denominator = 1 + z*z/n
	center_adjusted_probability = p + z*z / (2*n)
	adjusted_standard_deviation = np.sqrt((p*(1 - p) + z*z / (4*n)) / n)

	lower_bound = (center_adjusted_probability - z*adjusted_standard_deviation) / denominator
	upper_bound = (center_adjusted_probability + z*adjusted_standard_deviation) / denominator
	return (lower_bound, upper_bound)

def lighten_color(color, amount=0.5):
	import matplotlib.colors as mc
	import colorsys
	try:
		c = mc.cnames[color]
	except:
		c = color
	c = colorsys.rgb_to_hls(*mc.to_rgb(c))
	return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def get_count(result_array, index):
	if index < 2:
		return result_array[index]
	else:
		if index % 2 == 0:
			return result_array[index] - result_array[1]
		else:
			return result_array[1] + result_array[index]

def analyze_log(logfile):
	with open(logfile, "r") as log:
		parse_errors = []
		if logfile.endswith(".json"):
			import json
			results = []
			proofs_only = False
			first_example = True
			try:
				json_data = json.load(log)
			except json.JSONDecodeError:
				# try reading as JSONL
				log.seek(0)
				json_data = {}
				index = 1
				for line in log:
					json_data['example' + str(index)] = json.loads(line)
					index += 1
			log.close()
			for key, value in json_data.items():
				example_id = int(key[len("example"):])
				last_question = value["test_example"]["question"] + " " + value["test_example"]["query"]
				expected_answer = " ".join(value["test_example"]["chain_of_thought"])
				if "model_output" in value["test_example"]:
					predicted_answer = value["test_example"]["model_output"]
				elif "model_output" in value:
					predicted_answer = value["model_output"]
				elif "output" in value:
					predicted_answer = value["output"]
				else:
					print("ERROR: Example {} is missing model output.".format(example_id))
					continue

				if "logprobs" in value["test_example"]:
					logprobs = value["test_example"]["logprobs"]
				elif "logprobs" in value:
					logprobs = value["logprobs"]
				else:
					logprobs = None

				if type(predicted_answer) == list and len(predicted_answer) == 1:
					predicted_answer = predicted_answer[0]
				if predicted_answer.startswith("Output: "):
					predicted_answer = predicted_answer[len("Output: "):]

				try:
					if not first_example and proofs_only:
						raise ValueError("Log contains examples generated with and without 'proofs-only'.")
					last_question = last_question[:last_question.index('True or false:')]
				except ValueError:
					if not first_example and not proofs_only:
						raise ValueError("Log contains examples generated with and without 'proofs-only'.")
					last_question = last_question[:last_question.index('Prove:')]
					proofs_only = True

				(predicted_proof, predicted_label, errors) = parse_response(predicted_answer)
				parse_errors.extend(errors)
				result = evaluate_response(predicted_proof, predicted_label, expected_answer, parse_reasoning(last_question, parse_errors), proofs_only, parse_errors)
				result = result + (logprobs,)

				while example_id > len(results):
					results.append(None)
				results[example_id - 1] = result
				first_example = False
		else:
			parse_errors = []
			(_, _, results, _, _, parse_errors) = parse_log(log)
		if len(parse_errors) != 0:
			print('There were errors during semantic parsing for file ' + logfile + ':')
			for sentence, e in parse_errors:
				if type(e) in [UnableToParseError, AmbiguousParseError]:
					print('  Error parsing {}: {}'.format(sentence, e))
				else:
					import traceback
					print('  Error parsing {}: '.format(sentence))
					traceback.print_exception(e)
			exit(1)

	# collect incorrect steps
	all_correct_steps = []
	all_correct_and_useful_steps = []
	all_redundant_steps = []
	all_unparseable_steps = []
	all_incorrect_steps = []
	proof_lengths = []
	contains_correct_proof = [0, 0]
	does_not_contain_correct_proof = [0, 0]
	contains_wrong_branch = [0] * 8
	contains_useful_skip_step = [0] * 8
	contains_wrong_skip_step = [0] * 8
	contains_useful_non_atomic_step = [0] * 8
	contains_wrong_non_atomic_step = [0] * 8
	contains_invalid_step = [0] * 8
	contains_any_wrong_branch = [0] * 8
	contains_any_useful_skip_step = [0] * 8
	contains_any_wrong_skip_step = [0] * 8
	contains_any_useful_non_atomic_step = [0] * 8
	contains_any_wrong_non_atomic_step = [0] * 8
	contains_any_invalid_step = [0] * 8
	contains_wrong_branch_or_useful_non_atomic_step = [0] * 8
	contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step = [0] * 8
	contains_correct_proof_with_skip_step = 0
	contains_correct_proof_with_non_atomic_step = 0
	contains_correct_proof_with_skip_step_or_non_atomic_step = 0
	correct_labels = 0

	contains_wrong_branch_or_useful_skip_step = [0] * 8
	contains_wrong_branch_or_wrong_skip_step = [0] * 8
	contains_wrong_branch_or_wrong_non_atomic_step = [0] * 8
	contains_wrong_branch_or_invalid_step = [0] * 8
	contains_useful_skip_or_wrong_skip_step = [0] * 8
	contains_useful_skip_or_useful_non_atomic_step = [0] * 8
	contains_useful_skip_or_wrong_non_atomic_step = [0] * 8
	contains_useful_skip_or_invalid_step = [0] * 8
	contains_wrong_skip_or_useful_non_atomic_step = [0] * 8
	contains_wrong_skip_or_wrong_non_atomic_step = [0] * 8
	contains_wrong_skip_or_invalid_step = [0] * 8
	contains_useful_non_atomic_or_wrong_non_atomic_step = [0] * 8
	contains_useful_non_atomic_or_invalid_step = [0] * 8
	contains_wrong_non_atomic_or_invalid_step = [0] * 8

	contains_wrong_branch_or_non_atomic_step = [0] * 8
	contains_wrong_branch_or_useful_non_atomic_or_invalid_step = [0] * 8
	contains_wrong_branch_or_wrong_non_atomic_or_invalid_step = [0] * 8

	contains_wrong_branch_or_non_atomic_or_invalid_step = [0] * 8

	wrong_branch_first = [0] * 8
	useful_skip_step_first = [0] * 8
	wrong_skip_step_first = [0] * 8
	useful_non_atomic_step_first = [0] * 8
	wrong_non_atomic_step_first = [0] * 8
	invalid_step_first = [0] * 8
	expected_label_true = [0] * 8

	wrong_branch_lengths = []

	labels = []
	correct_proofs = []
	correct_proofs_with_skip_steps = []
	correct_proofs_with_non_atomic_steps = []
	correct_proofs_with_skip_or_non_atomic_steps = []

	question_id = 0
	correct_step_count = 0
	non_atomic_step_count = 0
	skip_step_count = 0
	invalid_step_count = 0
	perplexities = []
	fully_correct_proofs = 0
	for result in results:
		question_id += 1
		(label, expected_label, correct_steps, correct_and_useful_steps, redundant_steps, unparseable_steps, wrong_branch_steps, useful_skip_steps, wrong_skip_steps, useful_non_atomic_steps, wrong_non_atomic_steps, invalid_steps, incorrect_steps, found_conclusion, found_conclusion_with_skip_steps, found_conclusion_with_non_atomic_steps, logprobs) = result
		all_correct_steps.extend(correct_steps)
		correct_step_count += len(correct_steps)
		non_atomic_step_count += len(useful_non_atomic_steps) + len(wrong_non_atomic_steps)
		skip_step_count += len(useful_skip_steps) + len(wrong_skip_steps)
		invalid_step_count += len(incorrect_steps)
		all_correct_and_useful_steps.extend(correct_and_useful_steps)
		all_redundant_steps.extend(redundant_steps)
		all_unparseable_steps.extend(unparseable_steps)
		all_incorrect_steps.extend(incorrect_steps)
		safe_label = int(label) if label != None else 0
		if found_conclusion:
			contains_correct_proof[safe_label] += 1
		else:
			does_not_contain_correct_proof[safe_label] += 1

		found_conclusion_with_skip_or_non_atomic_steps = (found_conclusion_with_skip_steps or found_conclusion_with_non_atomic_steps)
		if found_conclusion_with_skip_steps and not found_conclusion:
			contains_correct_proof_with_skip_step += 1
		if found_conclusion_with_non_atomic_steps and not found_conclusion:
			contains_correct_proof_with_non_atomic_step += 1
		if found_conclusion_with_skip_or_non_atomic_steps and not found_conclusion:
			contains_correct_proof_with_skip_step_or_non_atomic_step += 1

		labels.append(label)
		correct_proofs.append(int(found_conclusion))
		correct_proofs_with_skip_steps.append(int(found_conclusion_with_skip_steps and not found_conclusion))
		correct_proofs_with_non_atomic_steps.append(int(found_conclusion_with_non_atomic_steps and not found_conclusion))
		correct_proofs_with_skip_or_non_atomic_steps.append(int(found_conclusion_with_skip_or_non_atomic_steps and not found_conclusion))

		def increment_count(result_array):
			result_array[int(found_conclusion)] += 1
			result_array[2 + int(found_conclusion_with_skip_steps and not found_conclusion)] += 1
			result_array[4 + int(found_conclusion_with_non_atomic_steps and not found_conclusion)] += 1
			result_array[6 + int(found_conclusion_with_skip_or_non_atomic_steps and not found_conclusion)] += 1

		if expected_label:
			increment_count(expected_label_true)
		if len(wrong_branch_steps) != 0 and (len(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + incorrect_steps) == 0 or min(wrong_branch_steps) <= min(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + incorrect_steps)):
			increment_count(wrong_branch_first)
		if len(useful_skip_steps) != 0 and (len(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + incorrect_steps) == 0 or min(useful_skip_steps) <= min(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + incorrect_steps)):
			increment_count(useful_skip_step_first)
		if len(wrong_skip_steps) != 0 and (len(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + incorrect_steps) == 0 or min(wrong_skip_steps) <= min(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + incorrect_steps)):
			increment_count(wrong_skip_step_first)
		if len(useful_non_atomic_steps) != 0 and (len(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + incorrect_steps) == 0 or min(useful_non_atomic_steps) <= min(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + incorrect_steps)):
			increment_count(useful_non_atomic_step_first)
		if len(wrong_non_atomic_steps) != 0 and (len(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + incorrect_steps) == 0 or min(wrong_non_atomic_steps) <= min(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + incorrect_steps)):
			increment_count(wrong_non_atomic_step_first)
		if len(incorrect_steps) != 0 and (len(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + incorrect_steps) == 0 or min(incorrect_steps) <= min(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + incorrect_steps)):
			increment_count(invalid_step_first)

		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(incorrect_steps) == 0:
			increment_count(contains_wrong_branch)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) != 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(incorrect_steps) == 0:
			increment_count(contains_useful_skip_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) != 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(incorrect_steps) == 0:
			increment_count(contains_wrong_skip_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) == 0 and len(incorrect_steps) == 0:
			increment_count(contains_useful_non_atomic_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) != 0 and len(incorrect_steps) == 0:
			increment_count(contains_wrong_non_atomic_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(incorrect_steps) != 0:
			increment_count(contains_invalid_step)

		if len(wrong_branch_steps) != 0:
			increment_count(contains_any_wrong_branch)
		if len(useful_skip_steps) != 0:
			increment_count(contains_any_useful_skip_step)
		if len(wrong_skip_steps) != 0:
			increment_count(contains_any_wrong_skip_step)
		if len(useful_non_atomic_steps) != 0:
			increment_count(contains_any_useful_non_atomic_step)
		if len(wrong_non_atomic_steps) != 0:
			increment_count(contains_any_wrong_non_atomic_step)
		if len(incorrect_steps) != 0:
			increment_count(contains_any_invalid_step)

		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) != 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(incorrect_steps) == 0:
			increment_count(contains_wrong_branch_or_useful_skip_step)
		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) != 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(incorrect_steps) == 0:
			increment_count(contains_wrong_branch_or_wrong_skip_step)
		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) == 0 and len(incorrect_steps) == 0:
			increment_count(contains_wrong_branch_or_useful_non_atomic_step)
		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) != 0 and len(incorrect_steps) == 0:
			increment_count(contains_wrong_branch_or_wrong_non_atomic_step)
		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(incorrect_steps) != 0:
			increment_count(contains_wrong_branch_or_invalid_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) != 0 and len(wrong_skip_steps) != 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(incorrect_steps) == 0:
			increment_count(contains_useful_skip_or_wrong_skip_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) != 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) == 0 and len(incorrect_steps) == 0:
			increment_count(contains_useful_skip_or_useful_non_atomic_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) != 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) != 0 and len(incorrect_steps) == 0:
			increment_count(contains_useful_skip_or_wrong_non_atomic_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) != 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(incorrect_steps) != 0:
			increment_count(contains_useful_skip_or_invalid_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) != 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) == 0 and len(incorrect_steps) == 0:
			increment_count(contains_wrong_skip_or_useful_non_atomic_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) != 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) != 0 and len(incorrect_steps) == 0:
			increment_count(contains_wrong_skip_or_wrong_non_atomic_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) != 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(incorrect_steps) != 0:
			increment_count(contains_wrong_skip_or_invalid_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) != 0 and len(incorrect_steps) == 0:
			increment_count(contains_useful_non_atomic_or_wrong_non_atomic_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) == 0 and len(incorrect_steps) != 0:
			increment_count(contains_useful_non_atomic_or_invalid_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) != 0 and len(incorrect_steps) != 0:
			increment_count(contains_wrong_non_atomic_or_invalid_step)

		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) != 0 and len(incorrect_steps) == 0:
			increment_count(contains_wrong_branch_or_non_atomic_step)
		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) == 0 and len(incorrect_steps) != 0:
			increment_count(contains_wrong_branch_or_useful_non_atomic_or_invalid_step)
		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) != 0 and len(incorrect_steps) != 0:
			increment_count(contains_wrong_branch_or_wrong_non_atomic_or_invalid_step)

		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) != 0 and len(incorrect_steps) != 0:
			increment_count(contains_wrong_branch_or_non_atomic_or_invalid_step)

		if len(wrong_branch_steps) != 0 or len(useful_skip_steps) != 0 or len(wrong_skip_steps) != 0 or len(useful_non_atomic_steps) != 0 or len(wrong_non_atomic_steps) != 0 or len(incorrect_steps) != 0:
			increment_count(contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step)
		if len(correct_steps + redundant_steps + incorrect_steps) == 0:
			proof_lengths.append(0)
		else:
			proof_lengths.append(max(correct_steps + redundant_steps + incorrect_steps) + 1)
		if len(correct_steps) + len(useful_non_atomic_steps) + len(wrong_non_atomic_steps) + len(useful_skip_steps) + len(wrong_skip_steps) == proof_lengths[-1]:
			fully_correct_proofs += 1

		# count the number of steps after a wrong branch step before a correct step
		if found_conclusion_with_skip_or_non_atomic_steps and len(wrong_branch_steps) > 0:
			# find the first useful step after the wrong branch
			index = wrong_branch_steps[0]
			corrected_indices = [step - index for step in (useful_skip_steps + useful_non_atomic_steps + correct_and_useful_steps) if step > index]
			if len(corrected_indices) != 0:
				wrong_branch_lengths.append(min([step - index for step in (useful_skip_steps + useful_non_atomic_steps + correct_and_useful_steps) if step > index]))
		if label == None:
			correct_labels = None
		else:
			correct_labels += label

		if logprobs != None:
			answer_index = 0
			for i in range(len(logprobs['tokens'])-1, 2, -1):
				if logprobs['tokens'][(i-4):(i-1)] == ['\n', 'A', ':']:
					answer_index = i
					break
			answer_logprobs = logprobs['token_logprobs'][answer_index:]
			perplexities.append(np.exp(-np.sum(answer_logprobs) / len(answer_logprobs)))
		else:
			perplexities.append(float('NaN'))
	return (len(results), proof_lengths, correct_labels, expected_label_true, correct_step_count, non_atomic_step_count, skip_step_count, invalid_step_count, all_correct_and_useful_steps, all_redundant_steps, all_unparseable_steps, all_incorrect_steps, contains_correct_proof, does_not_contain_correct_proof, contains_wrong_branch, contains_useful_skip_step, contains_wrong_skip_step, contains_useful_non_atomic_step, contains_wrong_non_atomic_step, contains_invalid_step, contains_wrong_branch_or_useful_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_step, contains_wrong_branch_or_invalid_step, contains_wrong_branch_or_useful_non_atomic_or_invalid_step, contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step, contains_wrong_branch_or_useful_skip_step, contains_wrong_branch_or_wrong_skip_step, contains_useful_skip_or_wrong_skip_step, contains_useful_skip_or_useful_non_atomic_step, contains_useful_skip_or_wrong_non_atomic_step, contains_useful_skip_or_invalid_step, contains_wrong_skip_or_useful_non_atomic_step, contains_wrong_skip_or_wrong_non_atomic_step, contains_wrong_skip_or_invalid_step, contains_useful_non_atomic_or_wrong_non_atomic_step, contains_useful_non_atomic_or_invalid_step, contains_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_or_invalid_step, contains_correct_proof_with_skip_step, contains_correct_proof_with_non_atomic_step, contains_correct_proof_with_skip_step_or_non_atomic_step, wrong_branch_first, useful_skip_step_first, wrong_skip_step_first, useful_non_atomic_step_first, wrong_non_atomic_step_first, invalid_step_first, contains_any_wrong_branch, contains_any_useful_skip_step, contains_any_wrong_skip_step, contains_any_useful_non_atomic_step, contains_any_wrong_non_atomic_step, contains_any_invalid_step, wrong_branch_lengths, labels, correct_proofs, correct_proofs_with_skip_steps, correct_proofs_with_non_atomic_steps, correct_proofs_with_skip_or_non_atomic_steps, fully_correct_proofs, perplexities)

if __name__ != "__main__":
	pass
elif len(argv) > 1:
	(num_examples, proof_lengths, correct_labels, expected_label_true, correct_step_count, non_atomic_step_count, skip_step_count, invalid_step_count, all_correct_and_useful_steps, all_redundant_steps, all_unparseable_steps, all_incorrect_steps, contains_correct_proof, does_not_contain_correct_proof, contains_wrong_branch, contains_useful_skip_step, contains_wrong_skip_step, contains_useful_non_atomic_step, contains_wrong_non_atomic_step, contains_invalid_step, contains_wrong_branch_or_useful_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_step, contains_wrong_branch_or_invalid_step, contains_wrong_branch_or_useful_non_atomic_or_invalid_step, contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step, contains_wrong_branch_or_useful_skip_step, contains_wrong_branch_or_wrong_skip_step, contains_useful_skip_or_wrong_skip_step, contains_useful_skip_or_useful_non_atomic_step, contains_useful_skip_or_wrong_non_atomic_step, contains_useful_skip_or_invalid_step, contains_wrong_skip_or_useful_non_atomic_step, contains_wrong_skip_or_wrong_non_atomic_step, contains_wrong_skip_or_invalid_step, contains_useful_non_atomic_or_wrong_non_atomic_step, contains_useful_non_atomic_or_invalid_step, contains_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_or_invalid_step, contains_correct_proof_with_skip_step, contains_correct_proof_with_non_atomic_step, contains_correct_proof_with_skip_step_or_non_atomic_step, wrong_branch_first, useful_skip_step_first, wrong_skip_step_first, useful_non_atomic_step_first, wrong_non_atomic_step_first, invalid_step_first, contains_any_wrong_branch, contains_any_useful_skip_step, contains_any_wrong_skip_step, contains_any_useful_non_atomic_step, contains_any_wrong_non_atomic_step, contains_any_invalid_step, wrong_branch_lengths, labels, correct_proofs, correct_proofs_with_skip_steps, correct_proofs_with_non_atomic_steps, correct_proofs_with_skip_or_non_atomic_steps, fully_correct_proofs, perplexities) = analyze_log(argv[1])

	max_proof_length = max(proof_lengths)
	total_steps = np.sum(proof_lengths)
	print("Correct steps: {}".format(correct_step_count / total_steps))
	print("Non-atomic steps: {}".format(non_atomic_step_count / total_steps))
	print("Skip steps: {}".format(skip_step_count / total_steps))
	print("Invalid steps: {}".format(invalid_step_count / total_steps))
	print("Correct and useful steps: {}".format(len(all_correct_and_useful_steps) / total_steps))
	#print("Redundant steps: {}".format(len(all_redundant_steps) / total_steps))
	print("Unparseable steps: {}".format(len(all_unparseable_steps) / total_steps))
	print("Incorrect steps: {}".format(len(all_incorrect_steps) / total_steps))
	print("Perplexity sample mean: {}".format(np.mean(perplexities)))
	print("Perplexity sample std dev: {}".format(np.std(perplexities)))

	offset = 6
	strictly_correct_proofs = np.sum(contains_correct_proof)
	if offset == 0:
		num_correct_proofs = strictly_correct_proofs
	elif offset == 2:
		num_correct_proofs = strictly_correct_proofs + contains_correct_proof_with_skip_step
	elif offset == 4:
		num_correct_proofs = strictly_correct_proofs + contains_correct_proof_with_non_atomic_step
	elif offset == 6:
		num_correct_proofs = strictly_correct_proofs + contains_correct_proof_with_skip_step_or_non_atomic_step
	print("Proportion of proofs that contain the correct proof: {}".format(num_correct_proofs / num_examples))
	print("Proportion of proofs that only contain correct, non-atomic, or skip steps: {}".format(fully_correct_proofs / num_examples))

	if correct_labels != None:
		print("Proof accuracy of examples with label `True`: {}".format(get_count(expected_label_true, offset + 1) / (expected_label_true[0] + expected_label_true[1])))
		print("Proof accuracy of examples with label `False`: {}".format((num_correct_proofs - get_count(expected_label_true, offset + 1)) / (num_examples - expected_label_true[0] - expected_label_true[1])))

	if num_correct_proofs == 0:
		print("(there are no correct proofs)")
	else:
		print("Proportion of correct proofs that contain a \"useless branch\":          {}".format(get_count(contains_wrong_branch, offset + 1) / num_correct_proofs))
		print("Proportion of correct proofs that contain a \"useful skip step\":        {}".format(get_count(contains_useful_skip_step, offset + 1) / num_correct_proofs))
		print("Proportion of correct proofs that contain a \"useless skip step\":       {}".format(get_count(contains_wrong_skip_step, offset + 1) / num_correct_proofs))
		print("Proportion of correct proofs that contain a \"useful non-atomic step\":  {}".format(get_count(contains_useful_non_atomic_step, offset + 1) / num_correct_proofs))
		print("Proportion of correct proofs that contain a \"useless non-atomic step\": {}".format(get_count(contains_wrong_non_atomic_step, offset + 1) / num_correct_proofs))
		print("Proportion of correct proofs that contain an \"invalid step\":           {}".format(get_count(contains_invalid_step, offset + 1) / num_correct_proofs))
		print("Proportion of correct proofs that contain a \"useless branch\" AND \"useful non-atomic step\": {}".format(get_count(contains_wrong_branch_or_useful_non_atomic_step, offset + 1) / num_correct_proofs))
		print("Proportion of correct proofs that contain a \"useless branch\" AND \"useless non-atomic step\": {}".format(get_count(contains_wrong_branch_or_wrong_non_atomic_step, offset + 1) / num_correct_proofs))
		print("Proportion of correct proofs that contain a \"useless branch\" AND \"invalid step\": {}".format(get_count(contains_wrong_branch_or_invalid_step, offset + 1) / num_correct_proofs))
		print("Proportion of correct proofs that contain a \"useless branch\" AND \"useful non-atomic step\" AND \"invalid step\": {}".format(get_count(contains_wrong_branch_or_useful_non_atomic_or_invalid_step, offset + 1) / num_correct_proofs))
		print("Proportion of correct proofs with ANY OF THE ABOVE:                    {}".format(get_count(contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step, offset + 1) / num_correct_proofs))
	if num_examples - num_correct_proofs == 0:
		print("(there are no incorrect proofs)")
	else:
		print("Proportion of incorrect proofs that contain a \"useless branch\":          {}".format(get_count(contains_wrong_branch, offset + 0) / (num_examples - num_correct_proofs)))
		print("Proportion of incorrect proofs that contain a \"useful skip step\":        {}".format(get_count(contains_useful_skip_step, offset + 0) / (num_examples - num_correct_proofs)))
		print("Proportion of incorrect proofs that contain a \"useless skip step\":       {}".format(get_count(contains_wrong_skip_step, offset + 0) / (num_examples - num_correct_proofs)))
		print("Proportion of incorrect proofs that contain a \"useful non-atomic step\":  {}".format(get_count(contains_useful_non_atomic_step, offset + 0) / (num_examples - num_correct_proofs)))
		print("Proportion of incorrect proofs that contain a \"useless non-atomic step\": {}".format(get_count(contains_wrong_non_atomic_step, offset + 0) / (num_examples - num_correct_proofs)))
		print("Proportion of incorrect proofs that contain an \"invalid step\":           {}".format(get_count(contains_invalid_step, offset + 0) / (num_examples - num_correct_proofs)))
		print("Proportion of incorrect proofs that contain a \"useless branch\" AND \"useful non-atomic step\": {}".format(get_count(contains_wrong_branch_or_useful_non_atomic_step, offset + 0) / (num_examples - num_correct_proofs)))
		print("Proportion of incorrect proofs that contain a \"useless branch\" AND \"useless non-atomic step\": {}".format(get_count(contains_wrong_branch_or_wrong_non_atomic_step, offset + 0) / (num_examples - num_correct_proofs)))
		print("Proportion of incorrect proofs that contain a \"useless branch\" AND \"invalid step\": {}".format(get_count(contains_wrong_branch_or_invalid_step, offset + 0) / (num_examples - num_correct_proofs)))
		print("Proportion of incorrect proofs that contain a \"useless branch\" AND \"useful non-atomic step\" AND \"invalid step\": {}".format(get_count(contains_wrong_branch_or_useful_non_atomic_or_invalid_step, offset + 0) / (num_examples - num_correct_proofs)))
		print("Proportion of incorrect proofs with ANY OF THE ABOVE:                    {}".format(get_count(contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step, offset + 0) / (num_examples - num_correct_proofs)))

	print("\nProportion of ALL proofs that are strictly correct: {}".format(strictly_correct_proofs / num_examples))
	print("Proportion of ALL proofs that would be correct if \"skip steps\" are considered correct: {}".format((strictly_correct_proofs + contains_correct_proof_with_skip_step) / num_examples))
	print("Proportion of ALL proofs that would be correct if \"non-atomic steps\" are considered correct: {}".format((strictly_correct_proofs + contains_correct_proof_with_non_atomic_step) / num_examples))
	print("Proportion of ALL proofs that would be correct if both \"skip steps\" and \"non-atomic steps\" are considered correct: {}".format((strictly_correct_proofs + contains_correct_proof_with_skip_step_or_non_atomic_step) / num_examples))

	if num_correct_proofs == 0:
		print("(there are no correct proofs)")
	else:
		print("\nProportion of correct proofs where the \"useless branch\" is the first mistake: {}".format(get_count(wrong_branch_first, offset + 1) / num_correct_proofs))
		print("Proportion of correct proofs where the \"useful skip step\" is the first mistake: {}".format(get_count(useful_skip_step_first, offset + 1) / num_correct_proofs))
		print("Proportion of correct proofs where the \"useless skip step\" is the first mistake: {}".format(get_count(wrong_skip_step_first, offset + 1) / num_correct_proofs))
		print("Proportion of correct proofs where the \"useful non-atomic step\" is the first mistake: {}".format(get_count(useful_non_atomic_step_first, offset + 1) / num_correct_proofs))
		print("Proportion of correct proofs where the \"useless non-atomic step\" is the first mistake: {}".format(get_count(wrong_non_atomic_step_first, offset + 1) / num_correct_proofs))
		print("Proportion of correct proofs where the \"invalid step\" is the first mistake: {}".format(get_count(invalid_step_first, offset + 1) / num_correct_proofs))
	if num_examples - num_correct_proofs == 0:
		print("(there are no incorrect proofs)")
	else:
		print("Proportion of incorrect proofs where the \"useless branch\" is the first mistake: {}".format(get_count(wrong_branch_first, offset + 0) / (num_examples - num_correct_proofs)))
		print("Proportion of incorrect proofs where the \"useful skip step\" is the first mistake: {}".format(get_count(useful_skip_step_first, offset + 0) / (num_examples - num_correct_proofs)))
		print("Proportion of incorrect proofs where the \"useless skip step\" is the first mistake: {}".format(get_count(wrong_skip_step_first, offset + 0) / (num_examples - num_correct_proofs)))
		print("Proportion of incorrect proofs where the \"useful non-atomic step\" is the first mistake: {}".format(get_count(useful_non_atomic_step_first, offset + 0) / (num_examples - num_correct_proofs)))
		print("Proportion of incorrect proofs where the \"useless non-atomic step\" is the first mistake: {}".format(get_count(wrong_non_atomic_step_first, offset + 0) / (num_examples - num_correct_proofs)))
		print("Proportion of incorrect proofs where the \"invalid step\" is the first mistake: {}".format(get_count(invalid_step_first, offset + 0) / (num_examples - num_correct_proofs)))

	incorrect_proof_ids_with_skip_steps = []
	incorrect_proof_ids_with_skip_or_non_atomic_steps = []
	diff = []
	for i in range(len(proof_lengths)):
		if correct_proofs[i] == 0 and correct_proofs_with_skip_steps[i] == 0:
			incorrect_proof_ids_with_skip_steps.append(i + 1)
			if correct_proofs_with_skip_or_non_atomic_steps[i] != 0:
				diff.append(i + 1)
		if correct_proofs[i] == 0 and correct_proofs_with_skip_or_non_atomic_steps[i] == 0:
			incorrect_proof_ids_with_skip_or_non_atomic_steps.append(i + 1)
	print("incorrect_proof_ids_with_skip_steps: {}".format(incorrect_proof_ids_with_skip_steps))
	print("incorrect_proof_ids_with_skip_or_non_atomic_steps: {}".format(incorrect_proof_ids_with_skip_or_non_atomic_steps))
	print("incorrect_proof_ids_with_skip_or_non_atomic_steps \ incorrect_proof_ids_with_skip_steps: {}".format(diff))
	incorrect_proof_with_only_invalid_steps = []

	if correct_labels != None:
		print("Proportion of correct labels: {}".format(correct_labels / num_examples))

else:
	import glob
	import matplotlib.pyplot as plt
	from matplotlib import rcParams
	from matplotlib.transforms import Bbox

	plt.rcParams.update({
		"text.usetex": True,
		"font.family": "serif",
		"font.serif": ["Palatino"],
	})

	def make_step_type_plot(chart_title, filename_glob, group_labels, chart_filename, first_error_chart_filename, wrong_branch_lengths_filename, add_bar_legend=False, figure_height=2.4, first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.8, show_ylabel=True, show_first_error_ylabel=True, first_error_title=None):
		if type(filename_glob) == str:
			logfiles = glob.glob(filename_glob)
		elif type(filename_glob) == list:
			logfiles = filename_glob
		else:
			raise TypeError("'filename_glob' must be either a string or list of strings (i.e. filenames).")

		correct_proof_fraction = []
		correct_proofs_wrong_branch = []
		correct_proofs_useful_non_atomic_step = []
		correct_proofs_wrong_non_atomic_step = []
		correct_proofs_useful_skip_step = []
		correct_proofs_wrong_skip_step = []
		correct_proofs_invalid_step = []
		incorrect_proofs_wrong_branch = []
		incorrect_proofs_useful_non_atomic_step = []
		incorrect_proofs_wrong_non_atomic_step = []
		incorrect_proofs_useful_skip_step = []
		incorrect_proofs_wrong_skip_step = []
		incorrect_proofs_invalid_step = []
		correct_proofs_wrong_branch_first = []
		correct_proofs_useful_skip_step_first = []
		correct_proofs_wrong_skip_step_first = []
		correct_proofs_useful_non_atomic_step_first = []
		correct_proofs_wrong_non_atomic_step_first = []
		correct_proofs_invalid_step_first = []
		incorrect_proofs_wrong_branch_first = []
		incorrect_proofs_useful_skip_step_first = []
		incorrect_proofs_wrong_skip_step_first = []
		incorrect_proofs_useful_non_atomic_step_first = []
		incorrect_proofs_wrong_non_atomic_step_first = []
		incorrect_proofs_invalid_step_first = []
		wrong_branch_lengths_array = []

		example_count = []
		offset = 6
		for logfile in logfiles:
			print('parsing "{}"'.format(logfile))
			(num_examples, proof_lengths, correct_labels, expected_label_true, correct_step_count, non_atomic_step_count, skip_step_count, invalid_step_count, all_correct_and_useful_steps, all_redundant_steps, all_unparseable_steps, all_incorrect_steps, contains_correct_proof, does_not_contain_correct_proof, contains_wrong_branch, contains_useful_skip_step, contains_wrong_skip_step, contains_useful_non_atomic_step, contains_wrong_non_atomic_step, contains_invalid_step, contains_wrong_branch_or_useful_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_step, contains_wrong_branch_or_invalid_step, contains_wrong_branch_or_useful_non_atomic_or_invalid_step, contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step, contains_wrong_branch_or_useful_skip_step, contains_wrong_branch_or_wrong_skip_step, contains_useful_skip_or_wrong_skip_step, contains_useful_skip_or_useful_non_atomic_step, contains_useful_skip_or_wrong_non_atomic_step, contains_useful_skip_or_invalid_step, contains_wrong_skip_or_useful_non_atomic_step, contains_wrong_skip_or_wrong_non_atomic_step, contains_wrong_skip_or_invalid_step, contains_useful_non_atomic_or_wrong_non_atomic_step, contains_useful_non_atomic_or_invalid_step, contains_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_or_invalid_step, contains_correct_proof_with_skip_step, contains_correct_proof_with_non_atomic_step, contains_correct_proof_with_skip_step_or_non_atomic_step, wrong_branch_first, useful_skip_step_first, wrong_skip_step_first, useful_non_atomic_step_first, wrong_non_atomic_step_first, invalid_step_first, contains_any_wrong_branch, contains_any_useful_skip_step, contains_any_wrong_skip_step, contains_any_useful_non_atomic_step, contains_any_wrong_non_atomic_step, contains_any_invalid_step, wrong_branch_lengths, labels, correct_proofs, correct_proofs_with_skip_steps, correct_proofs_with_non_atomic_steps, correct_proofs_with_skip_or_non_atomic_steps, fully_correct_proofs, total_perplexity) = analyze_log(logfile)

			correct_proof_count = np.sum(contains_correct_proof) + contains_correct_proof_with_skip_step_or_non_atomic_step
			example_count.append(num_examples)
			correct_proof_fraction.append(correct_proof_count / num_examples)
			correct_proofs_wrong_branch.append(get_count(contains_any_wrong_branch, offset + 1) / num_examples)
			correct_proofs_useful_non_atomic_step.append(get_count(contains_any_useful_non_atomic_step, offset + 1) / num_examples)
			correct_proofs_wrong_non_atomic_step.append(get_count(contains_any_wrong_non_atomic_step, offset + 1) / num_examples)
			correct_proofs_useful_skip_step.append(get_count(contains_any_useful_skip_step, offset + 1) / num_examples)
			correct_proofs_wrong_skip_step.append(get_count(contains_any_wrong_skip_step, offset + 1) / num_examples)
			correct_proofs_invalid_step.append(get_count(contains_any_invalid_step, offset + 1) / num_examples)
			incorrect_proofs_wrong_branch.append(get_count(contains_any_wrong_branch, offset + 0) / num_examples)
			incorrect_proofs_useful_non_atomic_step.append(get_count(contains_any_useful_non_atomic_step, offset + 0) / num_examples)
			incorrect_proofs_wrong_non_atomic_step.append(get_count(contains_any_wrong_non_atomic_step, offset + 0) / num_examples)
			incorrect_proofs_useful_skip_step.append(get_count(contains_any_useful_skip_step, offset + 0) / num_examples)
			incorrect_proofs_wrong_skip_step.append(get_count(contains_any_wrong_skip_step, offset + 0) / num_examples)
			incorrect_proofs_invalid_step.append(get_count(contains_any_invalid_step, offset + 0) / num_examples)

			correct_proofs_wrong_branch_first.append(get_count(wrong_branch_first, offset + 1) / num_examples)
			correct_proofs_useful_skip_step_first.append(get_count(useful_skip_step_first, offset + 1) / num_examples)
			correct_proofs_wrong_skip_step_first.append(get_count(wrong_skip_step_first, offset + 1) / num_examples)
			correct_proofs_useful_non_atomic_step_first.append(get_count(useful_non_atomic_step_first, offset + 1) / num_examples)
			correct_proofs_wrong_non_atomic_step_first.append(get_count(wrong_non_atomic_step_first, offset + 1) / num_examples)
			correct_proofs_invalid_step_first.append(get_count(invalid_step_first, offset + 1) / num_examples)
			incorrect_proofs_wrong_branch_first.append(get_count(wrong_branch_first, offset + 0) / num_examples)
			incorrect_proofs_useful_skip_step_first.append(get_count(useful_skip_step_first, offset + 0) / num_examples)
			incorrect_proofs_wrong_skip_step_first.append(get_count(wrong_skip_step_first, offset + 0) / num_examples)
			incorrect_proofs_useful_non_atomic_step_first.append(get_count(useful_non_atomic_step_first, offset + 0) / num_examples)
			incorrect_proofs_wrong_non_atomic_step_first.append(get_count(wrong_non_atomic_step_first, offset + 0) / num_examples)
			incorrect_proofs_invalid_step_first.append(get_count(invalid_step_first, offset + 0) / num_examples)

			counter = {}
			for length in wrong_branch_lengths:
				if length not in counter:
					counter[length] = 1
				else:
					counter[length] += 1
			wrong_branch_lengths_array.append((np.array(list(counter.keys())), np.array(list(counter.values()))))

		example_count = np.array(example_count)
		correct_proof_fraction = np.array(correct_proof_fraction)
		correct_proofs_wrong_branch = np.array(correct_proofs_wrong_branch)
		correct_proofs_useful_non_atomic_step = np.array(correct_proofs_useful_non_atomic_step)
		correct_proofs_wrong_non_atomic_step = np.array(correct_proofs_wrong_non_atomic_step)
		correct_proofs_useful_skip_step = np.array(correct_proofs_useful_skip_step)
		correct_proofs_wrong_skip_step = np.array(correct_proofs_wrong_skip_step)
		correct_proofs_invalid_step = np.array(correct_proofs_invalid_step)
		incorrect_proofs_wrong_branch = np.array(incorrect_proofs_wrong_branch)
		incorrect_proofs_useful_non_atomic_step = np.array(incorrect_proofs_useful_non_atomic_step)
		incorrect_proofs_wrong_non_atomic_step = np.array(incorrect_proofs_wrong_non_atomic_step)
		incorrect_proofs_useful_skip_step = np.array(incorrect_proofs_useful_skip_step)
		incorrect_proofs_wrong_skip_step = np.array(incorrect_proofs_wrong_skip_step)
		incorrect_proofs_invalid_step = np.array(incorrect_proofs_invalid_step)
		correct_proofs_wrong_branch_first = np.array(correct_proofs_wrong_branch_first)
		correct_proofs_useful_skip_step_first = np.array(correct_proofs_useful_skip_step_first)
		correct_proofs_wrong_skip_step_first = np.array(correct_proofs_wrong_skip_step_first)
		correct_proofs_useful_non_atomic_step_first = np.array(correct_proofs_useful_non_atomic_step_first)
		correct_proofs_wrong_non_atomic_step_first = np.array(correct_proofs_wrong_non_atomic_step_first)
		correct_proofs_invalid_step_first = np.array(correct_proofs_invalid_step_first)
		incorrect_proofs_wrong_branch_first = np.array(incorrect_proofs_wrong_branch_first)
		incorrect_proofs_useful_skip_step_first = np.array(incorrect_proofs_useful_skip_step_first)
		incorrect_proofs_wrong_skip_step_first = np.array(incorrect_proofs_wrong_skip_step_first)
		incorrect_proofs_useful_non_atomic_step_first = np.array(incorrect_proofs_useful_non_atomic_step_first)
		incorrect_proofs_wrong_non_atomic_step_first = np.array(incorrect_proofs_wrong_non_atomic_step_first)
		incorrect_proofs_invalid_step_first = np.array(incorrect_proofs_invalid_step_first)

		def sanity_check(array):
			if np.min(array) < 0.0 or np.max(array) > 1.0:
				raise ValueError("Given array's values are not between 0 and 1.")

		sanity_check(correct_proofs_wrong_branch)
		sanity_check(correct_proofs_useful_non_atomic_step)
		sanity_check(correct_proofs_wrong_non_atomic_step)
		sanity_check(correct_proofs_useful_skip_step)
		sanity_check(correct_proofs_wrong_skip_step)
		sanity_check(correct_proofs_invalid_step)
		sanity_check(incorrect_proofs_wrong_branch)
		sanity_check(incorrect_proofs_useful_non_atomic_step)
		sanity_check(incorrect_proofs_wrong_non_atomic_step)
		sanity_check(incorrect_proofs_useful_skip_step)
		sanity_check(incorrect_proofs_wrong_skip_step)
		sanity_check(incorrect_proofs_invalid_step)
		sanity_check(correct_proofs_wrong_branch_first)
		sanity_check(correct_proofs_useful_skip_step_first)
		sanity_check(correct_proofs_wrong_skip_step_first)
		sanity_check(correct_proofs_useful_non_atomic_step_first)
		sanity_check(correct_proofs_wrong_non_atomic_step_first)
		sanity_check(correct_proofs_invalid_step_first)
		sanity_check(incorrect_proofs_wrong_branch_first)
		sanity_check(incorrect_proofs_useful_skip_step_first)
		sanity_check(incorrect_proofs_wrong_skip_step_first)
		sanity_check(incorrect_proofs_useful_non_atomic_step_first)
		sanity_check(incorrect_proofs_wrong_non_atomic_step_first)
		sanity_check(incorrect_proofs_invalid_step_first)

		bar_group_size = 6
		bar_spacing = 0.2
		bar_width = (1.0 - bar_spacing) / bar_group_size

		x1 = np.arange(len(correct_proof_fraction))
		x2 = [x + bar_width for x in x1]
		x3 = [x + bar_width for x in x2]
		x4 = [x + bar_width for x in x3]
		x5 = [x + bar_width for x in x4]
		x6 = [x + bar_width for x in x5]

		fig = plt.gcf()
		fig.set_size_inches(10.0, figure_height, forward=True)
		plt.bar(x1, correct_proofs_wrong_branch, width=bar_width, color=lighten_color(colors[1], 1.3))
		plt.bar(x1, correct_proof_fraction - correct_proofs_wrong_branch, width=bar_width, bottom=correct_proofs_wrong_branch, color=lighten_color(colors[1], 0.8))
		plt.bar(x1, incorrect_proofs_wrong_branch, width=bar_width, bottom=correct_proof_fraction, color=lighten_color(colors[0], 1.3))
		plt.bar(x1, 1.0 - correct_proof_fraction - incorrect_proofs_wrong_branch, width=bar_width, bottom=correct_proof_fraction+incorrect_proofs_wrong_branch, color=lighten_color(colors[0], 0.8))

		plt.bar(x2, correct_proofs_useful_non_atomic_step, width=bar_width, color=lighten_color(colors[1], 1.3))
		plt.bar(x2, correct_proof_fraction - correct_proofs_useful_non_atomic_step, width=bar_width, bottom=correct_proofs_useful_non_atomic_step, color=lighten_color(colors[1], 0.8))
		plt.bar(x2, incorrect_proofs_useful_non_atomic_step, width=bar_width, bottom=correct_proof_fraction, color=lighten_color(colors[0], 1.3))
		plt.bar(x2, 1.0 - correct_proof_fraction - incorrect_proofs_useful_non_atomic_step, width=bar_width, bottom=correct_proof_fraction+incorrect_proofs_useful_non_atomic_step, color=lighten_color(colors[0], 0.8))

		plt.bar(x3, correct_proofs_wrong_non_atomic_step, width=bar_width, color=lighten_color(colors[1], 1.3))
		plt.bar(x3, correct_proof_fraction - correct_proofs_wrong_non_atomic_step, width=bar_width, bottom=correct_proofs_wrong_non_atomic_step, color=lighten_color(colors[1], 0.8))
		plt.bar(x3, incorrect_proofs_wrong_non_atomic_step, width=bar_width, bottom=correct_proof_fraction, color=lighten_color(colors[0], 1.3))
		plt.bar(x3, 1.0 - correct_proof_fraction - incorrect_proofs_wrong_non_atomic_step, width=bar_width, bottom=correct_proof_fraction+incorrect_proofs_wrong_non_atomic_step, color=lighten_color(colors[0], 0.8))

		plt.bar(x4, correct_proofs_useful_skip_step, width=bar_width, color=lighten_color(colors[1], 1.3))
		plt.bar(x4, correct_proof_fraction - correct_proofs_useful_skip_step, width=bar_width, bottom=correct_proofs_useful_skip_step, color=lighten_color(colors[1], 0.8))
		plt.bar(x4, incorrect_proofs_useful_skip_step, width=bar_width, bottom=correct_proof_fraction, color=lighten_color(colors[0], 1.3))
		plt.bar(x4, 1.0 - correct_proof_fraction - incorrect_proofs_useful_skip_step, width=bar_width, bottom=correct_proof_fraction+incorrect_proofs_useful_skip_step, color=lighten_color(colors[0], 0.8))

		plt.bar(x5, correct_proofs_wrong_skip_step, width=bar_width, color=lighten_color(colors[1], 1.3))
		plt.bar(x5, correct_proof_fraction - correct_proofs_wrong_skip_step, width=bar_width, bottom=correct_proofs_wrong_skip_step, color=lighten_color(colors[1], 0.8))
		plt.bar(x5, incorrect_proofs_wrong_skip_step, width=bar_width, bottom=correct_proof_fraction, color=lighten_color(colors[0], 1.3))
		plt.bar(x5, 1.0 - correct_proof_fraction - incorrect_proofs_wrong_skip_step, width=bar_width, bottom=correct_proof_fraction+incorrect_proofs_wrong_skip_step, color=lighten_color(colors[0], 0.8))

		plt.bar(x6, correct_proofs_invalid_step, width=bar_width, color=lighten_color(colors[1], 1.3))
		plt.bar(x6, correct_proof_fraction - correct_proofs_invalid_step, width=bar_width, bottom=correct_proofs_invalid_step, color=lighten_color(colors[1], 0.8))
		plt.bar(x6, incorrect_proofs_invalid_step, width=bar_width, bottom=correct_proof_fraction, color=lighten_color(colors[0], 1.3))
		plt.bar(x6, 1.0 - correct_proof_fraction - incorrect_proofs_invalid_step, width=bar_width, bottom=correct_proof_fraction+incorrect_proofs_invalid_step, color=lighten_color(colors[0], 0.8))

		# draw the error bars
		(lower_bound, upper_bound) = wilson_conf_interval(correct_proof_fraction, example_count)
		plt.errorbar(x1 + (1.0 - bar_spacing - bar_width) / 2, correct_proof_fraction, yerr=np.array((correct_proof_fraction - lower_bound, upper_bound - correct_proof_fraction)), fmt='none', ecolor=(0.0,0.0,0.0), capsize=3.0)

		ax = plt.gca()
		if show_ylabel:
			if offset == 0:
				plt.ylabel('strict proof accuracy')
			elif offset == 2:
				plt.ylabel('broad proof accuracy')
			elif offset == 4:
				plt.ylabel('``skip\'\' proof accuracy')
			elif offset == 6:
				plt.ylabel('valid proof accuracy')
		plt.xlim([-bar_spacing, len(correct_proof_fraction) - 0.12])
		plt.ylim([0.0, 1.0])
		plt.title(chart_title, fontsize=13)
		delta = (1.0 - bar_spacing) / (2 * bar_group_size)
		xticks_fontsize = 9
		if 'text-ada-001' in group_labels[0]:
			xticks_fontsize = 10
		plt.xticks([x + ((1.0 - bar_spacing) / 2) - delta for x in x1], group_labels, fontsize=xticks_fontsize)
		plt.tick_params(axis='x', which='both', length=0)
		if add_bar_legend:
			plt.text(0.0 + 0*bar_width - delta, 1.0, "misleading branches", rotation=70, color=(0.3,0.3,0.3), fontsize=8.0, horizontalalignment='left', verticalalignment='bottom')
			plt.text(0.0 + 1*bar_width - delta, 1.0, "correct non-atomic steps", rotation=70, color=(0.3,0.3,0.3), fontsize=8.0, horizontalalignment='left', verticalalignment='bottom')
			plt.text(0.0 + 2*bar_width - delta, 1.0, "misleading non-atomic steps", rotation=70, color=(0.3,0.3,0.3), fontsize=8.0, horizontalalignment='left', verticalalignment='bottom')
			plt.text(0.0 + 3*bar_width - delta, 1.0, "correct ``skip steps''", rotation=70, color=(0.3,0.3,0.3), fontsize=8.0, horizontalalignment='left', verticalalignment='bottom')
			plt.text(0.0 + 4*bar_width - delta, 1.0, "misleading ``skip steps''", rotation=70, color=(0.3,0.3,0.3), fontsize=8.0, horizontalalignment='left', verticalalignment='bottom')
			plt.text(0.0 + 5*bar_width - delta, 1.0, "invalid steps", rotation=70, color=(0.3,0.3,0.3), fontsize=8.0, horizontalalignment='left', verticalalignment='bottom')
		fig.savefig(chart_filename, dpi=128, bbox_inches=Bbox([[0.3, -0.1], [10.0 - 0.3, figure_height]]))
		plt.clf()

		if wrong_branch_lengths_filename != None:
			fig = plt.gcf()
			if "1hop" in logfiles[0]:
				fig.set_size_inches(7.0, wrong_branch_lengths_figure_height, forward=True)
				proof_range = range(2, len(correct_proof_fraction))
			else:
				fig.set_size_inches(10.0, wrong_branch_lengths_figure_height, forward=True)
				proof_range = range(len(correct_proof_fraction))
			for i in proof_range:
				max_length = 15
				ax = plt.subplot(1, len(proof_range), i - proof_range.start + 1)
				(wrong_branch_lengths, wrong_branch_lengths_counts) = wrong_branch_lengths_array[i]
				if len(wrong_branch_lengths) != 0:
					total_count = np.sum(wrong_branch_lengths_counts)
					(lower_bound, upper_bound) = wilson_conf_interval(wrong_branch_lengths_counts/total_count, total_count)
					err = np.array((wrong_branch_lengths_counts - lower_bound*total_count, upper_bound*total_count - wrong_branch_lengths_counts))
					ax.bar(wrong_branch_lengths, wrong_branch_lengths_counts, width=1.0, yerr=err, error_kw=dict(elinewidth=0.7, capthick=0.7, capsize=1.5), color=colors[1])
					max_length = max(max_length, np.max(wrong_branch_lengths_array[i][0]) + 2)
				ax.set_xlabel(group_labels[i], fontsize=9)
				if show_first_error_ylabel and i == proof_range.start:
					ax.set_ylabel('number of \n correct proofs')
				ax.xaxis.set_ticks(np.arange(0, 15 + 1, 5))
				ax.set_xlim(0, max_length)
				ax.set_ylim(0.0, 40)
				ax.tick_params(axis='x', which='both', length=0)
				if i != proof_range.start:
					ax.axes.yaxis.set_ticklabels([])
					plt.tick_params(axis='y', which='both', length=0)
			if first_error_title != None:
				plt.suptitle(first_error_title, fontsize=13)
			else:
				plt.suptitle(chart_title, fontsize=13)
			if "1hop" in logfiles[0]:
				fig.savefig(wrong_branch_lengths_filename, dpi=128, bbox_inches=Bbox([[0.0, -0.4], [7.0 - 0.3, wrong_branch_lengths_figure_height]]))
			else:
				fig.savefig(wrong_branch_lengths_filename, dpi=128, bbox_inches=Bbox([[0.0, -0.4], [10.0 - 0.3, wrong_branch_lengths_figure_height]]))
			plt.clf()

		if first_error_chart_filename != None:
			fig = plt.gcf()
			if "1hop" in logfiles[0]:
				fig.set_size_inches(7.0, first_error_figure_height, forward=True)
				x1 = [x - 2 for x in x1[2:]]
				x2 = [x - 2 for x in x2[2:]]
				x3 = [x - 2 for x in x3[2:]]
				x4 = [x - 2 for x in x4[2:]]
				x5 = [x - 2 for x in x5[2:]]
				x6 = [x - 2 for x in x6[2:]]
				group_labels = group_labels[2:]
				example_count = example_count[2:]
				correct_proof_fraction = correct_proof_fraction[2:]
				incorrect_proofs_wrong_branch_first = incorrect_proofs_wrong_branch_first[2:]
				incorrect_proofs_useful_non_atomic_step_first = incorrect_proofs_useful_non_atomic_step_first[2:]
				incorrect_proofs_wrong_non_atomic_step_first = incorrect_proofs_wrong_non_atomic_step_first[2:]
				incorrect_proofs_useful_skip_step_first = incorrect_proofs_useful_skip_step_first[2:]
				incorrect_proofs_wrong_skip_step_first = incorrect_proofs_wrong_skip_step_first[2:]
				incorrect_proofs_invalid_step_first = incorrect_proofs_invalid_step_first[2:]
			else:
				fig.set_size_inches(10.0, first_error_figure_height, forward=True)
			incorrect_proofs = 1.0 - correct_proof_fraction
			(lower_bound, upper_bound) = wilson_conf_interval(incorrect_proofs_wrong_branch_first/incorrect_proofs, example_count * incorrect_proofs)
			lower_bound[np.isnan(lower_bound)] = 0.0; upper_bound[np.isnan(upper_bound)] = 1.0
			err = np.maximum(0, np.array((incorrect_proofs_wrong_branch_first/incorrect_proofs - lower_bound, upper_bound - incorrect_proofs_wrong_branch_first/incorrect_proofs)))
			plt.bar(x1, incorrect_proofs_wrong_branch_first/incorrect_proofs, width=bar_width, yerr=err, capsize=3.0, color=colors[5])
			(lower_bound, upper_bound) = wilson_conf_interval(incorrect_proofs_useful_non_atomic_step_first/incorrect_proofs, example_count * incorrect_proofs)
			lower_bound[np.isnan(lower_bound)] = 0.0; upper_bound[np.isnan(upper_bound)] = 1.0
			err = np.maximum(0, np.array((incorrect_proofs_useful_non_atomic_step_first/incorrect_proofs - lower_bound, upper_bound - incorrect_proofs_useful_non_atomic_step_first/incorrect_proofs)))
			plt.bar(x2, incorrect_proofs_useful_non_atomic_step_first/incorrect_proofs, width=bar_width, yerr=err, capsize=3.0, color=lighten_color(colors[1], 0.8))
			(lower_bound, upper_bound) = wilson_conf_interval(incorrect_proofs_wrong_non_atomic_step_first/incorrect_proofs, example_count * incorrect_proofs)
			lower_bound[np.isnan(lower_bound)] = 0.0; upper_bound[np.isnan(upper_bound)] = 1.0
			err = np.maximum(0, np.array((incorrect_proofs_wrong_non_atomic_step_first/incorrect_proofs - lower_bound, upper_bound - incorrect_proofs_wrong_non_atomic_step_first/incorrect_proofs)))
			plt.bar(x3, incorrect_proofs_wrong_non_atomic_step_first/incorrect_proofs, width=bar_width, yerr=err, capsize=3.0, color=lighten_color(colors[1], 1.2))
			(lower_bound, upper_bound) = wilson_conf_interval(incorrect_proofs_useful_skip_step_first/incorrect_proofs, example_count * incorrect_proofs)
			lower_bound[np.isnan(lower_bound)] = 0.0; upper_bound[np.isnan(upper_bound)] = 1.0
			err = np.maximum(0, np.array((incorrect_proofs_useful_skip_step_first/incorrect_proofs - lower_bound, upper_bound - incorrect_proofs_useful_skip_step_first/incorrect_proofs)))
			plt.bar(x4, incorrect_proofs_useful_skip_step_first/incorrect_proofs, width=bar_width, yerr=err, capsize=3.0, color=lighten_color(colors[2], 0.8))
			(lower_bound, upper_bound) = wilson_conf_interval(incorrect_proofs_wrong_skip_step_first/incorrect_proofs, example_count * incorrect_proofs)
			lower_bound[np.isnan(lower_bound)] = 0.0; upper_bound[np.isnan(upper_bound)] = 1.0
			err = np.maximum(0, np.array((incorrect_proofs_wrong_skip_step_first/incorrect_proofs - lower_bound, upper_bound - incorrect_proofs_wrong_skip_step_first/incorrect_proofs)))
			plt.bar(x5, incorrect_proofs_wrong_skip_step_first/incorrect_proofs, width=bar_width, yerr=err, capsize=3.0, color=lighten_color(colors[2], 1.2))
			(lower_bound, upper_bound) = wilson_conf_interval(incorrect_proofs_invalid_step_first/incorrect_proofs, example_count * incorrect_proofs)
			lower_bound[np.isnan(lower_bound)] = 0.0; upper_bound[np.isnan(upper_bound)] = 1.0
			err = np.maximum(0, np.array((incorrect_proofs_invalid_step_first/incorrect_proofs - lower_bound, upper_bound - incorrect_proofs_invalid_step_first/incorrect_proofs)))
			plt.bar(x6, incorrect_proofs_invalid_step_first/incorrect_proofs, width=bar_width, yerr=err, capsize=3.0, color=colors[0])

			show_legend = False
			if show_legend:
				labels = ['strictly-valid atomic misleading steps', 'strictly-valid non-atomic correct steps', 'strictly-valid non-atomic misleading steps', 'broadly-valid correct steps', 'broadly-valid misleading steps', 'invalid steps']
				plt.legend(labels, loc='center left', bbox_to_anchor=(1.04, 0.5))

			ax = plt.gca()
			if show_first_error_ylabel:
				plt.ylabel('proportion of \n incorrect proofs')
			plt.xlim([-bar_spacing, len(correct_proof_fraction) - 0.12])
			plt.ylim([0.0, 1.0])
			if first_error_title == None:
				plt.title(chart_title, fontsize=13)
			else:
				plt.title(first_error_title, fontsize=13)
			delta = (1.0 - bar_spacing) / (3 * bar_group_size)
			plt.xticks([x + ((1.0 - bar_spacing) / 2) - delta for x in x1], group_labels, fontsize=xticks_fontsize)
			plt.tick_params(axis='x', which='both', length=0)
			xlabel_line_count = np.max([label.count('\n') for label in group_labels])
			if "1hop" in logfiles[0]:
				width = 7.0
				if show_legend:
					width += 1.5
				fig.savefig(first_error_chart_filename, dpi=128, bbox_inches=Bbox([[0.0, (xlabel_line_count + 1) * -0.1], [width - 0.3, first_error_figure_height]]))
			else:
				fig.savefig(first_error_chart_filename, dpi=128, bbox_inches=Bbox([[0.0, (xlabel_line_count + 1) * -0.1], [10.0 - 0.3, first_error_figure_height]]))
			plt.clf()


	logfiles = ['gpt_textdavinci002_1hop.log', 'gpt_textdavinci002_1hop_preorder.log', 'gpt_textdavinci002_3hop.log', 'gpt_textdavinci002_3hop_preorder.log', 'gpt_textdavinci002_5hop.log', 'gpt_textdavinci002_5hop_preorder.log'] \
			 + ['gpt_textdavinci002_1hop_falseontology.log', 'gpt_textdavinci002_1hop_preorder_falseontology.log', 'gpt_textdavinci002_3hop_falseontology.log', 'gpt_textdavinci002_3hop_preorder_falseontology.log', 'gpt_textdavinci002_5hop_falseontology.log', 'gpt_textdavinci002_5hop_preorder_falseontology.log'] \
			 + ['gpt_textdavinci002_1hop_trueontology.log', 'gpt_textdavinci002_1hop_preorder_trueontology.log', 'gpt_textdavinci002_3hop_trueontology.log', 'gpt_textdavinci002_3hop_preorder_trueontology.log', 'gpt_textdavinci002_5hop_trueontology.log', 'gpt_textdavinci002_5hop_preorder_trueontology.log'] \
			 + ['gpt_textada001_3hop_preorder.log', 'gpt_textbabbage001_3hop_preorder.log', 'gpt_textcurie001_3hop_preorder.log', 'gpt_davinci_3hop_preorder.log', 'gpt_textdavinci001_3hop_preorder.log', 'gpt_textdavinci002_3hop_preorder.log'] \
			 + ['gpt_textada001_3hop_preorder_falseontology.log', 'gpt_textbabbage001_3hop_preorder_falseontology.log', 'gpt_textcurie001_3hop_preorder_falseontology.log', 'gpt_davinci_3hop_preorder_falseontology.log', 'gpt_textdavinci001_3hop_preorder_falseontology.log', 'gpt_textdavinci002_3hop_preorder_falseontology.log'] \
			 + ['gpt_textada001_3hop_preorder_trueontology.log', 'gpt_textbabbage001_3hop_preorder_trueontology.log', 'gpt_textcurie001_3hop_preorder_trueontology.log', 'gpt_davinci_3hop_preorder_trueontology.log', 'gpt_textdavinci001_3hop_preorder_trueontology.log', 'gpt_textdavinci002_3hop_preorder_trueontology.log'] \
			 + ['gpt_textada001_1hop_preorder.log', 'gpt_textbabbage001_1hop_preorder.log', 'gpt_textcurie001_1hop_preorder.log', 'gpt_davinci_1hop_preorder.log', 'gpt_textdavinci001_1hop_preorder.log', 'gpt_textdavinci002_1hop_preorder.log'] \
			 + ['gpt_textada001_1hop_preorder_falseontology.log', 'gpt_textbabbage001_1hop_preorder_falseontology.log', 'gpt_textcurie001_1hop_preorder_falseontology.log', 'gpt_davinci_1hop_preorder_falseontology.log', 'gpt_textdavinci001_1hop_preorder_falseontology.log', 'gpt_textdavinci002_1hop_preorder_falseontology.log'] \
			 + ['gpt_textada001_1hop_preorder_trueontology.log', 'gpt_textbabbage001_1hop_preorder_trueontology.log', 'gpt_textcurie001_1hop_preorder_trueontology.log', 'gpt_davinci_1hop_preorder_trueontology.log', 'gpt_textdavinci001_1hop_preorder_trueontology.log', 'gpt_textdavinci002_1hop_trueontology.log']
	logfiles = set(logfiles)
	example_count = []
	label_accuracy = []
	proof_accuracy = []
	proof_accuracy_with_skip_steps = []
	proof_accuracy_with_non_atomic_steps = []
	proof_accuracy_with_skip_steps_and_non_atomic_steps = []

	correct_proofs_wrong_branch = []
	incorrect_proofs_wrong_branch = []
	incorrect_proofs_useful_non_atomic_step = []
	incorrect_proofs_wrong_non_atomic_step = []
	incorrect_proofs_useful_skip_step = []
	incorrect_proofs_wrong_skip_step = []
	incorrect_proofs_invalid_step = []
	incorrect_proofs_wrong_branch_and_useful_non_atomic_step = []
	incorrect_proofs_wrong_branch_and_wrong_non_atomic_step = []
	incorrect_proofs_wrong_branch_and_invalid_step = []
	incorrect_proofs_wrong_branch_and_useful_non_atomic_step_and_invalid_step = []
	incorrect_proofs_other = []

	confusion_matrix = np.zeros((2, 2))
	confusion_matrix_with_skip_steps = np.zeros((2, 2))
	confusion_matrix_with_non_atomic_steps = np.zeros((2, 2))
	confusion_matrix_with_skip_and_non_atomic_steps = np.zeros((2, 2))

	plt.style.use('ggplot')
	colors = []
	for c in rcParams["axes.prop_cycle"]:
		colors.append(c['color'])
	# colors are red, blue, purple, grey, yellow, green, pink

	point_colors = []
	color_code = "model_size"
	for logfile in logfiles:
		if "seed" in logfile:
			continue
		print('parsing "{}"'.format(logfile))
		(num_examples, proof_lengths, correct_labels, expected_label_true, correct_step_count, non_atomic_step_count, skip_step_count, invalid_step_count, all_correct_and_useful_steps, all_redundant_steps, all_unparseable_steps, all_incorrect_steps, contains_correct_proof, does_not_contain_correct_proof, contains_wrong_branch, contains_useful_skip_step, contains_wrong_skip_step, contains_useful_non_atomic_step, contains_wrong_non_atomic_step, contains_invalid_step, contains_wrong_branch_or_useful_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_step, contains_wrong_branch_or_invalid_step, contains_wrong_branch_or_useful_non_atomic_or_invalid_step, contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step, contains_wrong_branch_or_useful_skip_step, contains_wrong_branch_or_wrong_skip_step, contains_useful_skip_or_wrong_skip_step, contains_useful_skip_or_useful_non_atomic_step, contains_useful_skip_or_wrong_non_atomic_step, contains_useful_skip_or_invalid_step, contains_wrong_skip_or_useful_non_atomic_step, contains_wrong_skip_or_wrong_non_atomic_step, contains_wrong_skip_or_invalid_step, contains_useful_non_atomic_or_wrong_non_atomic_step, contains_useful_non_atomic_or_invalid_step, contains_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_or_invalid_step, contains_correct_proof_with_skip_step, contains_correct_proof_with_non_atomic_step, contains_correct_proof_with_skip_step_or_non_atomic_step, wrong_branch_first, useful_skip_step_first, wrong_skip_step_first, useful_non_atomic_step_first, wrong_non_atomic_step_first, invalid_step_first, contains_any_wrong_branch, contains_any_useful_skip_step, contains_any_wrong_skip_step, contains_any_useful_non_atomic_step, contains_any_wrong_non_atomic_step, contains_any_invalid_step, wrong_branch_lengths, labels, correct_proofs, correct_proofs_with_skip_steps, correct_proofs_with_non_atomic_steps, correct_proofs_with_skip_or_non_atomic_steps, fully_correct_proofs, total_perplexity) = analyze_log(logfile)
		print(np.sum(contains_wrong_branch_or_useful_skip_step + contains_wrong_branch_or_wrong_skip_step + contains_useful_skip_or_wrong_skip_step + contains_useful_skip_or_useful_non_atomic_step + contains_useful_skip_or_wrong_non_atomic_step + contains_useful_skip_or_invalid_step + contains_wrong_skip_or_useful_non_atomic_step + contains_wrong_skip_or_wrong_non_atomic_step + contains_wrong_skip_or_invalid_step + contains_useful_non_atomic_or_wrong_non_atomic_step + contains_useful_non_atomic_or_invalid_step + contains_wrong_non_atomic_or_invalid_step + contains_wrong_branch_or_non_atomic_step + contains_wrong_branch_or_wrong_non_atomic_or_invalid_step + contains_wrong_branch_or_non_atomic_or_invalid_step))

		correct_proof_count = np.sum(contains_correct_proof)
		example_count.append(num_examples)
		label_accuracy.append(correct_labels / num_examples)
		proof_accuracy.append(correct_proof_count / num_examples)
		proof_accuracy_with_skip_steps.append((correct_proof_count + contains_correct_proof_with_skip_step) / num_examples)
		proof_accuracy_with_non_atomic_steps.append((correct_proof_count + contains_correct_proof_with_non_atomic_step) / num_examples)
		proof_accuracy_with_skip_steps_and_non_atomic_steps.append((correct_proof_count + contains_correct_proof_with_skip_step_or_non_atomic_step) / num_examples)

		labels = np.array(labels)
		correct_proofs = np.array(correct_proofs)
		correct_proofs_with_skip_steps = np.logical_or(np.array(correct_proofs_with_skip_steps), correct_proofs)
		correct_proofs_with_non_atomic_steps = np.logical_or(np.array(correct_proofs_with_non_atomic_steps), correct_proofs)
		correct_proofs_with_skip_or_non_atomic_steps = np.logical_or(np.array(correct_proofs_with_skip_or_non_atomic_steps), correct_proofs)
		confusion_matrix[0, 0] += np.sum(np.logical_and(labels == 0, correct_proofs == 0))
		confusion_matrix[0, 1] += np.sum(np.logical_and(labels == 0, correct_proofs == 1))
		confusion_matrix[1, 0] += np.sum(np.logical_and(labels == 1, correct_proofs == 0))
		confusion_matrix[1, 1] += np.sum(np.logical_and(labels == 1, correct_proofs == 1))
		confusion_matrix_with_skip_steps[0, 0] += np.sum(np.logical_and(labels == 0, correct_proofs_with_skip_steps == 0))
		confusion_matrix_with_skip_steps[0, 1] += np.sum(np.logical_and(labels == 0, correct_proofs_with_skip_steps == 1))
		confusion_matrix_with_skip_steps[1, 0] += np.sum(np.logical_and(labels == 1, correct_proofs_with_skip_steps == 0))
		confusion_matrix_with_skip_steps[1, 1] += np.sum(np.logical_and(labels == 1, correct_proofs_with_skip_steps == 1))
		confusion_matrix_with_non_atomic_steps[0, 0] += np.sum(np.logical_and(labels == 0, correct_proofs_with_non_atomic_steps == 0))
		confusion_matrix_with_non_atomic_steps[0, 1] += np.sum(np.logical_and(labels == 0, correct_proofs_with_non_atomic_steps == 1))
		confusion_matrix_with_non_atomic_steps[1, 0] += np.sum(np.logical_and(labels == 1, correct_proofs_with_non_atomic_steps == 0))
		confusion_matrix_with_non_atomic_steps[1, 1] += np.sum(np.logical_and(labels == 1, correct_proofs_with_non_atomic_steps == 1))
		confusion_matrix_with_skip_and_non_atomic_steps[0, 0] += np.sum(np.logical_and(labels == 0, correct_proofs_with_skip_or_non_atomic_steps == 0))
		confusion_matrix_with_skip_and_non_atomic_steps[0, 1] += np.sum(np.logical_and(labels == 0, correct_proofs_with_skip_or_non_atomic_steps == 1))
		confusion_matrix_with_skip_and_non_atomic_steps[1, 0] += np.sum(np.logical_and(labels == 1, correct_proofs_with_skip_or_non_atomic_steps == 0))
		confusion_matrix_with_skip_and_non_atomic_steps[1, 1] += np.sum(np.logical_and(labels == 1, correct_proofs_with_skip_or_non_atomic_steps == 1))

		if color_code == "hop_count":
			if "1hop" in logfile:
				point_colors.append(colors[0])
			elif "3hop" in logfile:
				point_colors.append(colors[1])
			elif "5hop" in logfile:
				point_colors.append(colors[2])
			else:
				raise Exception("Unable to determine hop count.")
		elif color_code == "model_size":
			if "_textada001_" in logfile:
				point_colors.append(colors[0])
			elif "_textbabbage001_" in logfile:
				point_colors.append(colors[1])
			elif "_textcurie001_" in logfile:
				point_colors.append(colors[2])
			elif "_davinci_" in logfile:
				point_colors.append(colors[3])
			elif "_textdavinci001_" in logfile:
				point_colors.append(colors[4])
			elif "_textdavinci002_" in logfile:
				point_colors.append(colors[5])
			else:
				raise Exception("Unable to determine model size.")


	example_count = np.array(example_count)
	label_accuracy = np.array(label_accuracy)
	proof_accuracy = np.array(proof_accuracy)
	proof_accuracy_with_skip_steps = np.array(proof_accuracy_with_skip_steps)
	proof_accuracy_with_non_atomic_steps = np.array(proof_accuracy_with_non_atomic_steps)
	proof_accuracy_with_skip_steps_and_non_atomic_steps = np.array(proof_accuracy_with_skip_steps_and_non_atomic_steps)

	confusion_matrix /= np.sum(example_count)
	confusion_matrix_with_skip_steps /= np.sum(example_count)
	confusion_matrix_with_non_atomic_steps /= np.sum(example_count)
	confusion_matrix_with_skip_and_non_atomic_steps /= np.sum(example_count)

	print("Confusion matrices:")
	print(confusion_matrix)
	print(confusion_matrix_with_skip_steps)
	print(confusion_matrix_with_non_atomic_steps)
	print(confusion_matrix_with_skip_and_non_atomic_steps)

	(label_lower_bound, label_upper_bound) = wilson_conf_interval(label_accuracy, example_count)

	fig = plt.gcf()
	fig.set_size_inches(2.2, 2.2, forward=True)
	plt.plot([0, 1], [0, 1], color='black')
	(proof_lower_bound, proof_upper_bound) = wilson_conf_interval(proof_accuracy, example_count)
	plt.errorbar(label_accuracy, proof_accuracy, xerr=np.array((label_accuracy - label_lower_bound, label_upper_bound - label_accuracy)), yerr=np.array((proof_accuracy - proof_lower_bound, proof_upper_bound - proof_accuracy)), fmt='none', ecolor=(0.53,0.53,0.53), elinewidth=0.8)
	if len(point_colors) != 0:
		plt.scatter(label_accuracy, proof_accuracy, zorder=10, s=20.0, c=point_colors)
	else:
		plt.scatter(label_accuracy, proof_accuracy, zorder=10, s=20.0)
	plt.xlabel('label accuracy')
	plt.ylabel('strict proof accuracy')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	fig.savefig('label_vs_proof_accuracy.pdf', dpi=128, bbox_inches='tight')
	plt.clf()

	fig = plt.gcf()
	fig.set_size_inches(2.2, 2.2, forward=True)
	plt.plot([0, 1], [0, 1], color='black')
	(proof_lower_bound, proof_upper_bound) = wilson_conf_interval(proof_accuracy_with_skip_steps, example_count)
	plt.errorbar(label_accuracy, proof_accuracy_with_skip_steps, xerr=np.array((label_accuracy - label_lower_bound, label_upper_bound - label_accuracy)), yerr=np.array((proof_accuracy_with_skip_steps - proof_lower_bound, proof_upper_bound - proof_accuracy_with_skip_steps)), fmt='none', ecolor=(0.53,0.53,0.53), elinewidth=0.8)
	if len(point_colors) != 0:
		plt.scatter(label_accuracy, proof_accuracy_with_skip_steps, zorder=10, s=20.0, c=point_colors)
	else:
		plt.scatter(label_accuracy, proof_accuracy_with_skip_steps, zorder=10, s=20.0, c=colors[1])
	plt.xlabel('label accuracy')
	plt.ylabel('broad proof accuracy')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	fig.savefig('label_vs_proof_accuracy_with_skip_steps.pdf', dpi=128, bbox_inches='tight')
	plt.clf()

	fig = plt.gcf()
	fig.set_size_inches(2.2, 2.2, forward=True)
	plt.plot([0, 1], [0, 1], color='black')
	(proof_lower_bound, proof_upper_bound) = wilson_conf_interval(proof_accuracy_with_non_atomic_steps, example_count)
	plt.errorbar(label_accuracy, proof_accuracy_with_non_atomic_steps, xerr=np.array((label_accuracy - label_lower_bound, label_upper_bound - label_accuracy)), yerr=np.array((proof_accuracy_with_non_atomic_steps - proof_lower_bound, proof_upper_bound - proof_accuracy_with_non_atomic_steps)), fmt='none', ecolor=(0.53,0.53,0.53), elinewidth=0.8)
	if len(point_colors) != 0:
		plt.scatter(label_accuracy, proof_accuracy_with_non_atomic_steps, zorder=10, s=20.0, c=point_colors)
	else:
		plt.scatter(label_accuracy, proof_accuracy_with_non_atomic_steps, zorder=10, s=20.0, c=colors[2])
	plt.xlabel('label accuracy')
	plt.ylabel('``skip\'\' proof accuracy')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	fig.savefig('label_vs_proof_accuracy_with_non_atomic_steps.pdf', dpi=128, bbox_inches='tight')
	plt.clf()

	fig = plt.gcf()
	fig.set_size_inches(2.2, 2.2, forward=True)
	plt.plot([0, 1], [0, 1], color='black')
	(proof_lower_bound, proof_upper_bound) = wilson_conf_interval(proof_accuracy_with_skip_steps_and_non_atomic_steps, example_count)
	plt.errorbar(label_accuracy, proof_accuracy_with_skip_steps_and_non_atomic_steps, xerr=np.array((label_accuracy - label_lower_bound, label_upper_bound - label_accuracy)), yerr=np.array((proof_accuracy_with_skip_steps_and_non_atomic_steps - proof_lower_bound, proof_upper_bound - proof_accuracy_with_skip_steps_and_non_atomic_steps)), fmt='none', ecolor=(0.53,0.53,0.53), elinewidth=0.8)
	if len(point_colors) != 0:
		plt.scatter(label_accuracy, proof_accuracy_with_skip_steps_and_non_atomic_steps, zorder=10, s=20.0, c=point_colors)
	else:
		plt.scatter(label_accuracy, proof_accuracy_with_skip_steps_and_non_atomic_steps, zorder=10, s=20.0, c=colors[5])
	plt.xlabel('label accuracy')
	plt.ylabel('valid proof accuracy')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	fig.savefig('label_vs_proof_accuracy_with_skip_steps_and_non_atomic_steps.pdf', dpi=128, bbox_inches='tight')
	plt.clf()

	make_step_type_plot('Fictional ontology',
		['gpt_textdavinci002_1hop.log', 'gpt_textdavinci002_1hop_preorder.log', 'gpt_textdavinci002_3hop.log', 'gpt_textdavinci002_3hop_preorder.log', 'gpt_textdavinci002_5hop.log', 'gpt_textdavinci002_5hop_preorder.log'],
		['1 hop, bottom-up \n traversal direction', '1 hop, top-down \n traversal direction', '3 hops, bottom-up \n traversal direction', '3 hops, top-down \n traversal direction', '5 hops, bottom-up \n traversal direction', '5 hops, top-down \n traversal direction'],
		'textdavinci002_fictional_ontology_proof_accuracy.pdf', 'textdavinci002_fictional_ontology_first_error.pdf', 'textdavinci002_fictional_ontology_wrong_branch_lengths.pdf', first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.6)

	make_step_type_plot('False ontology',
		['gpt_textdavinci002_1hop_falseontology.log', 'gpt_textdavinci002_1hop_preorder_falseontology.log', 'gpt_textdavinci002_3hop_falseontology.log', 'gpt_textdavinci002_3hop_preorder_falseontology.log', 'gpt_textdavinci002_5hop_falseontology.log', 'gpt_textdavinci002_5hop_preorder_falseontology.log'],
		['1 hop, bottom-up \n traversal direction', '1 hop, top-down \n traversal direction', '3 hops, bottom-up \n traversal direction', '3 hops, top-down \n traversal direction', '5 hops, bottom-up \n traversal direction', '5 hops, top-down \n traversal direction'],
		'textdavinci002_false_ontology_proof_accuracy.pdf', 'textdavinci002_false_ontology_first_error.pdf', 'textdavinci002_false_ontology_wrong_branch_lengths.pdf', first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.6, show_ylabel=False, show_first_error_ylabel=False)

	make_step_type_plot('True ontology',
		['gpt_textdavinci002_1hop_trueontology.log', 'gpt_textdavinci002_1hop_preorder_trueontology.log', 'gpt_textdavinci002_3hop_trueontology.log', 'gpt_textdavinci002_3hop_preorder_trueontology.log', 'gpt_textdavinci002_5hop_trueontology.log', 'gpt_textdavinci002_5hop_preorder_trueontology.log'],
		['1 hop, bottom-up \n traversal direction', '1 hop, top-down \n traversal direction', '3 hops, bottom-up \n traversal direction', '3 hops, top-down \n traversal direction', '5 hops, bottom-up \n traversal direction', '5 hops, top-down \n traversal direction'],
		'textdavinci002_true_ontology_proof_accuracy.pdf', 'textdavinci002_true_ontology_first_error.pdf', 'textdavinci002_true_ontology_wrong_branch_lengths.pdf', first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.6, show_ylabel=False, show_first_error_ylabel=False)

	make_step_type_plot('Fictional ontology, 3 hops',
		['gpt_textada001_3hop_preorder.log', 'gpt_textbabbage001_3hop_preorder.log', 'gpt_textcurie001_3hop_preorder.log', 'gpt_davinci_3hop_preorder.log', 'gpt_textdavinci001_3hop_preorder.log', 'gpt_textdavinci002_3hop_preorder.log'],
		['\\texttt{text-ada-001}', '\\texttt{text-babbage-001}', '\\texttt{text-curie-001}', '\\texttt{davinci}', '\\texttt{text-davinci-001}', '\\texttt{text-davinci-002}'],
		'fictional_ontology_3hop_model_size.pdf', 'fictional_ontology_3hop_model_size_first_error.pdf', 'fictional_ontology_3hop_model_size_wrong_branch_lengths.pdf', figure_height=1.8, first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.6, show_first_error_ylabel=True, first_error_title='Fictional ontology, 3 hops, top-down traversal direction')

	make_step_type_plot('False ontology, 3 hops',
		['gpt_textada001_3hop_preorder_falseontology.log', 'gpt_textbabbage001_3hop_preorder_falseontology.log', 'gpt_textcurie001_3hop_preorder_falseontology.log', 'gpt_davinci_3hop_preorder_falseontology.log', 'gpt_textdavinci001_3hop_preorder_falseontology.log', 'gpt_textdavinci002_3hop_preorder_falseontology.log'],
		['\\texttt{text-ada-001}', '\\texttt{text-babbage-001}', '\\texttt{text-curie-001}', '\\texttt{davinci}', '\\texttt{text-davinci-001}', '\\texttt{text-davinci-002}'],
		'false_ontology_3hop_model_size.pdf', 'false_ontology_3hop_model_size_first_error.pdf', 'false_ontology_3hop_model_size_wrong_branch_lengths.pdf', figure_height=1.8, first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.6, show_ylabel=False, show_first_error_ylabel=False, first_error_title='False ontology, 3 hops, top-down traversal direction')

	make_step_type_plot('True ontology, 3 hops',
		['gpt_textada001_3hop_preorder_trueontology.log', 'gpt_textbabbage001_3hop_preorder_trueontology.log', 'gpt_textcurie001_3hop_preorder_trueontology.log', 'gpt_davinci_3hop_preorder_trueontology.log', 'gpt_textdavinci001_3hop_preorder_trueontology.log', 'gpt_textdavinci002_3hop_preorder_trueontology.log'],
		['\\texttt{text-ada-001}', '\\texttt{text-babbage-001}', '\\texttt{text-curie-001}', '\\texttt{davinci}', '\\texttt{text-davinci-001}', '\\texttt{text-davinci-002}'],
		'true_ontology_3hop_model_size.pdf', 'true_ontology_3hop_model_size_first_error.pdf', 'true_ontology_3hop_model_size_wrong_branch_lengths.pdf', figure_height=1.8, first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.6, show_ylabel=False, show_first_error_ylabel=False, first_error_title='True ontology, 3 hops, top-down traversal direction')

	make_step_type_plot('Fictional ontology, 1 hop',
		['gpt_textada001_1hop_preorder.log', 'gpt_textbabbage001_1hop_preorder.log', 'gpt_textcurie001_1hop_preorder.log', 'gpt_davinci_1hop_preorder.log', 'gpt_textdavinci001_1hop_preorder.log', 'gpt_textdavinci002_1hop_preorder.log'],
		['\\texttt{text-ada-001}', '\\texttt{text-babbage-001}', '\\texttt{text-curie-001}', '\\texttt{davinci}', '\\texttt{text-davinci-001}', '\\texttt{text-davinci-002}'],
		'fictional_ontology_1hop_model_size.pdf', 'fictional_ontology_1hop_model_size_first_error.pdf', 'fictional_ontology_1hop_model_size_wrong_branch_lengths.pdf', figure_height=1.8, first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.6, show_ylabel=False, show_first_error_ylabel=False, first_error_title='Fictional ontology, 1 hop, top-down traversal direction')

	make_step_type_plot('False ontology, 1 hop',
		['gpt_textada001_1hop_preorder_falseontology.log', 'gpt_textbabbage001_1hop_preorder_falseontology.log', 'gpt_textcurie001_1hop_preorder_falseontology.log', 'gpt_davinci_1hop_preorder_falseontology.log', 'gpt_textdavinci001_1hop_preorder_falseontology.log', 'gpt_textdavinci002_1hop_preorder_falseontology.log'],
		['\\texttt{text-ada-001}', '\\texttt{text-babbage-001}', '\\texttt{text-curie-001}', '\\texttt{davinci}', '\\texttt{text-davinci-001}', '\\texttt{text-davinci-002}'],
		'false_ontology_1hop_model_size.pdf', 'false_ontology_1hop_model_size_first_error.pdf', 'false_ontology_1hop_model_size_wrong_branch_lengths.pdf', figure_height=1.8, first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.6, show_ylabel=False, show_first_error_ylabel=False, first_error_title='False ontology, 1 hop, top-down traversal direction')

	make_step_type_plot('True ontology, 1 hop',
		['gpt_textada001_1hop_preorder_trueontology.log', 'gpt_textbabbage001_1hop_preorder_trueontology.log', 'gpt_textcurie001_1hop_preorder_trueontology.log', 'gpt_davinci_1hop_preorder_trueontology.log', 'gpt_textdavinci001_1hop_preorder_trueontology.log', 'gpt_textdavinci002_1hop_trueontology.log'],
		['\\texttt{text-ada-001}', '\\texttt{text-babbage-001}', '\\texttt{text-curie-001}', '\\texttt{davinci}', '\\texttt{text-davinci-001}', '\\texttt{text-davinci-002}'],
		'true_ontology_1hop_model_size.pdf', 'true_ontology_1hop_model_size_first_error.pdf', 'true_ontology_1hop_model_size_wrong_branch_lengths.pdf', figure_height=1.8, first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.6, show_ylabel=False, show_first_error_ylabel=False, first_error_title='True ontology, 1 hop, top-down traversal direction')
