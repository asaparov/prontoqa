import numpy as np
from run_experiment import parse_log
from sys import argv

# see https://www.mikulskibartosz.name/wilson-score-in-python-example/
def wilson_conf_interval(p, n, z=1.96):
	denominator = 1 + z*z/n
	center_adjusted_probability = p + z*z / (2*n)
	adjusted_standard_deviation = np.sqrt((p*(1 - p) + z*z / (4*n)) / n)

	lower_bound = (center_adjusted_probability - z*adjusted_standard_deviation) / denominator
	upper_bound = (center_adjusted_probability + z*adjusted_standard_deviation) / denominator
	return (lower_bound, upper_bound)


def analyze_log(logfile):
	with open(logfile, "r") as log:
		(_, _, results, _, _) = parse_log(log)

	# collect incorrect steps
	all_correct_steps = []
	all_correct_and_useful_steps = []
	all_redundant_steps = []
	all_unparseable_steps = []
	all_incorrect_steps = []
	proof_lengths = []
	contains_correct_proof = [0] * 8
	does_not_contain_correct_proof = [0] * 8
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

	wrong_branch_lengths = []

	question_id = 0
	correct_step_count = 0
	non_atomic_step_count = 0
	skip_step_count = 0
	invalid_step_count = 0
	for result in results:
		question_id += 1
		(label, correct_steps, correct_and_useful_steps, redundant_steps, unparseable_steps, wrong_branch_steps, useful_skip_steps, wrong_skip_steps, useful_non_atomic_steps, wrong_non_atomic_steps, invalid_steps, incorrect_steps, found_conclusion, found_conclusion_with_skip_steps, found_conclusion_with_non_atomic_steps) = result
		all_correct_steps.extend(correct_steps)
		correct_step_count += len(correct_steps)
		non_atomic_step_count += len(useful_non_atomic_steps) + len(wrong_non_atomic_steps)
		skip_step_count += len(useful_skip_steps) + len(wrong_skip_steps)
		invalid_step_count += len(incorrect_steps)
		all_correct_and_useful_steps.extend(correct_and_useful_steps)
		all_redundant_steps.extend(redundant_steps)
		all_unparseable_steps.extend(unparseable_steps)
		all_incorrect_steps.extend(incorrect_steps)
		if found_conclusion:
			contains_correct_proof[int(label)] += 1
		else:
			does_not_contain_correct_proof[int(label)] += 1

		found_conclusion_with_skip_or_non_atomic_steps = (found_conclusion_with_skip_steps or found_conclusion_with_non_atomic_steps)
		if found_conclusion_with_skip_steps:
			contains_correct_proof_with_skip_step += 1
		if found_conclusion_with_non_atomic_steps:
			contains_correct_proof_with_non_atomic_step += 1
		if found_conclusion_with_skip_or_non_atomic_steps:
			contains_correct_proof_with_skip_step_or_non_atomic_step += 1

		def increment_count(result_array):
			result_array[int(found_conclusion)] += 1
			result_array[2 + int(found_conclusion_with_skip_steps)] += 1
			result_array[4 + int(found_conclusion_with_non_atomic_steps)] += 1
			result_array[6 + int(found_conclusion_with_skip_or_non_atomic_steps)] += 1

		if len(wrong_branch_steps) != 0 and (len(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + invalid_steps) == 0 or min(wrong_branch_steps) <= min(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + invalid_steps)):
			increment_count(wrong_branch_first)
		if len(useful_skip_steps) != 0 and (len(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + invalid_steps) == 0 or min(useful_skip_steps) <= min(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + invalid_steps)):
			increment_count(useful_skip_step_first)
		if len(wrong_skip_steps) != 0 and (len(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + invalid_steps) == 0 or min(wrong_skip_steps) <= min(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + invalid_steps)):
			increment_count(wrong_skip_step_first)
		if len(useful_non_atomic_steps) != 0 and (len(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + invalid_steps) == 0 or min(useful_non_atomic_steps) <= min(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + invalid_steps)):
			increment_count(useful_non_atomic_step_first)
		if len(wrong_non_atomic_steps) != 0 and (len(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + invalid_steps) == 0 or min(wrong_non_atomic_steps) <= min(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + invalid_steps)):
			increment_count(wrong_non_atomic_step_first)
		if len(invalid_steps) != 0 and (len(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + invalid_steps) == 0 or min(invalid_steps) <= min(wrong_branch_steps + useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + invalid_steps)):
			increment_count(invalid_step_first)

		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			increment_count(contains_wrong_branch)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) != 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			increment_count(contains_useful_skip_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) != 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			increment_count(contains_wrong_skip_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			increment_count(contains_useful_non_atomic_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) != 0 and len(invalid_steps) == 0:
			increment_count(contains_wrong_non_atomic_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) != 0:
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
		if len(invalid_steps) != 0:
			increment_count(contains_any_invalid_step)

		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) != 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			increment_count(contains_wrong_branch_or_useful_skip_step)
		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) != 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			increment_count(contains_wrong_branch_or_wrong_skip_step)
		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			increment_count(contains_wrong_branch_or_useful_non_atomic_step)
		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) != 0 and len(invalid_steps) == 0:
			increment_count(contains_wrong_branch_or_wrong_non_atomic_step)
		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) != 0:
			increment_count(contains_wrong_branch_or_invalid_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) != 0 and len(wrong_skip_steps) != 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			increment_count(contains_useful_skip_or_wrong_skip_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) != 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			increment_count(contains_useful_skip_or_useful_non_atomic_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) != 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) != 0 and len(invalid_steps) == 0:
			increment_count(contains_useful_skip_or_wrong_non_atomic_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) != 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) != 0:
			increment_count(contains_useful_skip_or_invalid_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) != 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			increment_count(contains_wrong_skip_or_useful_non_atomic_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) != 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) != 0 and len(invalid_steps) == 0:
			increment_count(contains_wrong_skip_or_wrong_non_atomic_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) != 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) != 0:
			increment_count(contains_wrong_skip_or_invalid_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) != 0 and len(invalid_steps) == 0:
			increment_count(contains_useful_non_atomic_or_wrong_non_atomic_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) != 0:
			increment_count(contains_useful_non_atomic_or_invalid_step)
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) != 0 and len(invalid_steps) != 0:
			increment_count(contains_wrong_non_atomic_or_invalid_step)

		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) != 0 and len(invalid_steps) == 0:
			increment_count(contains_wrong_branch_or_non_atomic_step)
		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) != 0:
			increment_count(contains_wrong_branch_or_useful_non_atomic_or_invalid_step)
		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) != 0 and len(invalid_steps) != 0:
			increment_count(contains_wrong_branch_or_wrong_non_atomic_or_invalid_step)

		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) != 0 and len(invalid_steps) != 0:
			increment_count(contains_wrong_branch_or_non_atomic_or_invalid_step)

		if len(wrong_branch_steps) != 0 or len(useful_skip_steps) != 0 or len(wrong_skip_steps) != 0 or len(useful_non_atomic_steps) != 0 or len(wrong_non_atomic_steps) != 0 or len(invalid_steps) != 0:
			increment_count(contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step)
		if len(correct_steps + redundant_steps + incorrect_steps) == 0:
			proof_lengths.append(0)
		else:
			proof_lengths.append(max(correct_steps + redundant_steps + incorrect_steps) + 1)

		# count the number of steps after a wrong branch step before a correct step
		if found_conclusion_with_skip_or_non_atomic_steps and len(wrong_branch_steps) > 0:
			# find the first useful step after the wrong branch
			index = wrong_branch_steps[0]
			corrected_indices = [step - index for step in (useful_skip_steps + useful_non_atomic_steps + correct_and_useful_steps) if step > index]
			if len(corrected_indices) != 0:
				wrong_branch_lengths.append(min([step - index for step in (useful_skip_steps + useful_non_atomic_steps + correct_and_useful_steps) if step > index]))
		correct_labels += label

	return (len(results), proof_lengths, correct_labels, correct_step_count, non_atomic_step_count, skip_step_count, invalid_step_count, all_correct_and_useful_steps, all_redundant_steps, all_unparseable_steps, all_incorrect_steps, contains_correct_proof, does_not_contain_correct_proof, contains_wrong_branch, contains_useful_skip_step, contains_wrong_skip_step, contains_useful_non_atomic_step, contains_wrong_non_atomic_step, contains_invalid_step, contains_wrong_branch_or_useful_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_step, contains_wrong_branch_or_invalid_step, contains_wrong_branch_or_useful_non_atomic_or_invalid_step, contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step, contains_wrong_branch_or_useful_skip_step, contains_wrong_branch_or_wrong_skip_step, contains_useful_skip_or_wrong_skip_step, contains_useful_skip_or_useful_non_atomic_step, contains_useful_skip_or_wrong_non_atomic_step, contains_useful_skip_or_invalid_step, contains_wrong_skip_or_useful_non_atomic_step, contains_wrong_skip_or_wrong_non_atomic_step, contains_wrong_skip_or_invalid_step, contains_useful_non_atomic_or_wrong_non_atomic_step, contains_useful_non_atomic_or_invalid_step, contains_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_or_invalid_step, contains_correct_proof_with_skip_step, contains_correct_proof_with_non_atomic_step, contains_correct_proof_with_skip_step_or_non_atomic_step, wrong_branch_first, useful_skip_step_first, wrong_skip_step_first, useful_non_atomic_step_first, wrong_non_atomic_step_first, invalid_step_first, contains_any_wrong_branch, contains_any_useful_skip_step, contains_any_wrong_skip_step, contains_any_useful_non_atomic_step, contains_any_wrong_non_atomic_step, contains_any_invalid_step, wrong_branch_lengths)

if len(argv) > 1:
	(num_examples, proof_lengths, correct_labels, correct_step_count, non_atomic_step_count, skip_step_count, invalid_step_count, all_correct_and_useful_steps, all_redundant_steps, all_unparseable_steps, all_incorrect_steps, contains_correct_proof, does_not_contain_correct_proof, contains_wrong_branch, contains_useful_skip_step, contains_wrong_skip_step, contains_useful_non_atomic_step, contains_wrong_non_atomic_step, contains_invalid_step, contains_wrong_branch_or_useful_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_step, contains_wrong_branch_or_invalid_step, contains_wrong_branch_or_useful_non_atomic_or_invalid_step, contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step, contains_wrong_branch_or_useful_skip_step, contains_wrong_branch_or_wrong_skip_step, contains_useful_skip_or_wrong_skip_step, contains_useful_skip_or_useful_non_atomic_step, contains_useful_skip_or_wrong_non_atomic_step, contains_useful_skip_or_invalid_step, contains_wrong_skip_or_useful_non_atomic_step, contains_wrong_skip_or_wrong_non_atomic_step, contains_wrong_skip_or_invalid_step, contains_useful_non_atomic_or_wrong_non_atomic_step, contains_useful_non_atomic_or_invalid_step, contains_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_or_invalid_step, contains_correct_proof_with_skip_step, contains_correct_proof_with_non_atomic_step, contains_correct_proof_with_skip_step_or_non_atomic_step, wrong_branch_first, useful_skip_step_first, wrong_skip_step_first, useful_non_atomic_step_first, wrong_non_atomic_step_first, invalid_step_first, contains_any_wrong_branch, contains_any_useful_skip_step, contains_any_wrong_skip_step, contains_any_useful_non_atomic_step, contains_any_wrong_non_atomic_step, contains_any_invalid_step, wrong_branch_lengths) = analyze_log(argv[1])

	max_proof_length = max(proof_lengths)
	total_steps = np.sum(proof_lengths)
	print("Correct steps: {}".format(correct_step_count / total_steps))
	print("Non-atomic steps: {}".format(non_atomic_step_count / total_steps))
	print("Skip steps: {}".format(skip_step_count / total_steps))
	print("Invalid steps: {}".format(invalid_step_count / total_steps))
	print("Correct and useful steps: {}".format(len(all_correct_and_useful_steps) / total_steps))
	print("Redundant steps: {}".format(len(all_redundant_steps) / total_steps))
	print("Unparseable steps: {}".format(len(all_unparseable_steps) / total_steps))
	print("Incorrect steps: {}".format(len(all_incorrect_steps) / total_steps))

	correct_proofs = np.sum(contains_correct_proof)
	print("Proportion of proofs that contain the correct proof: {}".format(correct_proofs / num_examples))

	offset = 6
	print("Proportion of proofs with the correct label that contain the correct proof:   {}".format(contains_correct_proof[offset + 1] / correct_labels))
	print("Proportion of proofs with the incorrect label that contain the correct proof: {}".format(contains_correct_proof[offset + 0] / (num_examples - correct_labels)))
	print("Proportion of proofs with the correct label that do NOT contain the correct proof:   {}".format(does_not_contain_correct_proof[offset + 1] / correct_labels))
	print("Proportion of proofs with the incorrect label that do NOT contain the correct proof: {}".format(does_not_contain_correct_proof[offset + 0] / (num_examples - correct_labels)))

	print("Proportion of correct proofs that contain a \"useless branch\":          {}".format(contains_wrong_branch[offset + 1] / correct_proofs))
	print("Proportion of correct proofs that contain a \"useful skip step\":        {}".format(contains_useful_skip_step[offset + 1] / correct_proofs))
	print("Proportion of correct proofs that contain a \"useless skip step\":       {}".format(contains_wrong_skip_step[offset + 1] / correct_proofs))
	print("Proportion of correct proofs that contain a \"useful non-atomic step\":  {}".format(contains_useful_non_atomic_step[offset + 1] / correct_proofs))
	print("Proportion of correct proofs that contain a \"useless non-atomic step\": {}".format(contains_wrong_non_atomic_step[offset + 1] / correct_proofs))
	print("Proportion of correct proofs that contain an \"invalid step\":           {}".format(contains_invalid_step[offset + 1] / correct_proofs))
	print("Proportion of correct proofs that contain a \"useless branch\" AND \"useful non-atomic step\": {}".format(contains_wrong_branch_or_useful_non_atomic_step[offset + 1] / correct_proofs))
	print("Proportion of correct proofs that contain a \"useless branch\" AND \"useless non-atomic step\": {}".format(contains_wrong_branch_or_wrong_non_atomic_step[offset + 1] / correct_proofs))
	print("Proportion of correct proofs that contain a \"useless branch\" AND \"invalid step\": {}".format(contains_wrong_branch_or_invalid_step[offset + 1] / (num_examples - correct_proofs)))
	print("Proportion of correct proofs that contain a \"useless branch\" AND \"useful non-atomic step\" AND \"invalid step\": {}".format(contains_wrong_branch_or_useful_non_atomic_or_invalid_step[offset + 1] / correct_proofs))
	print("Proportion of correct proofs with ANY OF THE ABOVE:                    {}".format(contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step[offset + 1] / correct_proofs))
	print("Proportion of incorrect proofs that contain a \"useless branch\":          {}".format(contains_wrong_branch[offset + 0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs that contain a \"useful skip step\":        {}".format(contains_useful_skip_step[offset + 0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs that contain a \"useless skip step\":       {}".format(contains_wrong_skip_step[offset + 0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs that contain a \"useful non-atomic step\":  {}".format(contains_useful_non_atomic_step[offset + 0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs that contain a \"useless non-atomic step\": {}".format(contains_wrong_non_atomic_step[offset + 0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs that contain an \"invalid step\":           {}".format(contains_invalid_step[offset + 0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs that contain a \"useless branch\" AND \"useful non-atomic step\": {}".format(contains_wrong_branch_or_useful_non_atomic_step[offset + 0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs that contain a \"useless branch\" AND \"useless non-atomic step\": {}".format(contains_wrong_branch_or_wrong_non_atomic_step[offset + 0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs that contain a \"useless branch\" AND \"invalid step\": {}".format(contains_wrong_branch_or_invalid_step[offset + 0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs that contain a \"useless branch\" AND \"useful non-atomic step\" AND \"invalid step\": {}".format(contains_wrong_branch_or_useful_non_atomic_or_invalid_step[offset + 0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs with ANY OF THE ABOVE:                    {}".format(contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step[offset + 0] / (num_examples - correct_proofs)))
	print("Proportion of ALL proofs that would be correct if \"skip steps\" are considered correct: {}".format((correct_proofs + contains_correct_proof_with_skip_step) / num_examples))
	print("Proportion of ALL proofs that would be correct if \"non-atomic steps\" are considered correct: {}".format((correct_proofs + contains_correct_proof_with_non_atomic_step) / num_examples))
	print("Proportion of ALL proofs that would be correct if both \"skip steps\" and \"non-atomic steps\" are considered correct: {}".format((correct_proofs + contains_correct_proof_with_skip_step_or_non_atomic_step) / num_examples))
	print("Proportion of correct proofs where the \"useless branch\" is the first mistake: {}".format(wrong_branch_first[offset + 1] / correct_proofs))
	print("Proportion of incorrect proofs where the \"useless branch\" is the first mistake: {}".format(wrong_branch_first[offset + 0] / (num_examples - correct_proofs)))
	print("Proportion of correct proofs where the \"invalid step\" is the first mistake: {}".format(invalid_step_first[offset + 1] / correct_proofs))
	print("Proportion of incorrect proofs where the \"invalid step\" is the first mistake: {}".format(invalid_step_first[offset + 0] / (num_examples - correct_proofs)))
	print("wrong_branch_first: {}".format(wrong_branch_first))
	print("invalid_step_first: {}".format(invalid_step_first))

	print(contains_wrong_branch_or_useful_skip_step[offset + 1] / correct_proofs)
	print(contains_wrong_branch_or_wrong_skip_step[offset + 1] / correct_proofs)
	print(contains_useful_skip_or_wrong_skip_step[offset + 1] / correct_proofs)
	print(contains_useful_skip_or_useful_non_atomic_step[offset + 1] / correct_proofs)
	print(contains_useful_skip_or_wrong_non_atomic_step[offset + 1] / correct_proofs)
	print(contains_useful_skip_or_invalid_step[offset + 1] / correct_proofs)
	print(contains_wrong_skip_or_useful_non_atomic_step[offset + 1] / correct_proofs)
	print(contains_wrong_skip_or_wrong_non_atomic_step[offset + 1] / correct_proofs)
	print(contains_wrong_skip_or_invalid_step[offset + 1] / correct_proofs)
	print(contains_useful_non_atomic_or_wrong_non_atomic_step[offset + 1] / correct_proofs)
	print(contains_useful_non_atomic_or_invalid_step[offset + 1] / correct_proofs)
	print(contains_wrong_non_atomic_or_invalid_step[offset + 1] / correct_proofs)
	print(contains_wrong_branch_or_non_atomic_step[offset + 1] / correct_proofs)
	print(contains_wrong_branch_or_wrong_non_atomic_or_invalid_step[offset + 1] / correct_proofs)
	print(contains_wrong_branch_or_non_atomic_or_invalid_step[offset + 1] / correct_proofs)

	print(contains_wrong_branch_or_useful_skip_step[offset + 0] / (num_examples - correct_proofs))
	print(contains_wrong_branch_or_wrong_skip_step[offset + 0] / (num_examples - correct_proofs))
	print(contains_useful_skip_or_wrong_skip_step[offset + 0] / (num_examples - correct_proofs))
	print(contains_useful_skip_or_useful_non_atomic_step[offset + 0] / (num_examples - correct_proofs))
	print(contains_useful_skip_or_wrong_non_atomic_step[offset + 0] / (num_examples - correct_proofs))
	print(contains_useful_skip_or_invalid_step[offset + 0] / (num_examples - correct_proofs))
	print(contains_wrong_skip_or_useful_non_atomic_step[offset + 0] / (num_examples - correct_proofs))
	print(contains_wrong_skip_or_wrong_non_atomic_step[offset + 0] / (num_examples - correct_proofs))
	print(contains_wrong_skip_or_invalid_step[offset + 0] / (num_examples - correct_proofs))
	print(contains_useful_non_atomic_or_wrong_non_atomic_step[offset + 0] / (num_examples - correct_proofs))
	print(contains_useful_non_atomic_or_invalid_step[offset + 0] / (num_examples - correct_proofs))
	print(contains_wrong_non_atomic_or_invalid_step[offset + 0] / (num_examples - correct_proofs))
	print(contains_wrong_branch_or_non_atomic_step[offset + 0] / (num_examples - correct_proofs))
	print(contains_wrong_branch_or_wrong_non_atomic_or_invalid_step[offset + 0] / (num_examples - correct_proofs))
	print(contains_wrong_branch_or_non_atomic_or_invalid_step[offset + 0] / (num_examples - correct_proofs))

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

	def lighten_color(color, amount=0.5):
		import matplotlib.colors as mc
		import colorsys
		try:
			c = mc.cnames[color]
		except:
			c = color
		c = colorsys.rgb_to_hls(*mc.to_rgb(c))
		return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

	def make_step_type_plot(chart_title, filename_glob, group_labels, chart_filename, first_error_chart_filename, wrong_branch_lengths_filename, add_bar_legend=False, figure_height=2.4, first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.8, show_ylabel=True, show_first_error_ylabel=True, first_error_title=None):
		if type(filename_glob) == str:
			logfiles = glob.glob(filename_glob)
		elif type(filename_glob) == list:
			logfiles = filename_glob
		else:
			raise TypeError("'filename_glob' must be either a string or list of strings (i.e. filenames).")

		def get_count(result_array, index):
			if index < 2:
				return result_array[index]
			else:
				if index % 2 == 0:
					return result_array[0] - result_array[index + 1]
				else:
					return result_array[1] + result_array[index]

		correct_proofs = []
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
			(num_examples, proof_lengths, correct_labels, correct_step_count, non_atomic_step_count, skip_step_count, invalid_step_count, all_correct_and_useful_steps, all_redundant_steps, all_unparseable_steps, all_incorrect_steps, contains_correct_proof, does_not_contain_correct_proof, contains_wrong_branch, contains_useful_skip_step, contains_wrong_skip_step, contains_useful_non_atomic_step, contains_wrong_non_atomic_step, contains_invalid_step, contains_wrong_branch_or_useful_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_step, contains_wrong_branch_or_invalid_step, contains_wrong_branch_or_useful_non_atomic_or_invalid_step, contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step, contains_wrong_branch_or_useful_skip_step, contains_wrong_branch_or_wrong_skip_step, contains_useful_skip_or_wrong_skip_step, contains_useful_skip_or_useful_non_atomic_step, contains_useful_skip_or_wrong_non_atomic_step, contains_useful_skip_or_invalid_step, contains_wrong_skip_or_useful_non_atomic_step, contains_wrong_skip_or_wrong_non_atomic_step, contains_wrong_skip_or_invalid_step, contains_useful_non_atomic_or_wrong_non_atomic_step, contains_useful_non_atomic_or_invalid_step, contains_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_or_invalid_step, contains_correct_proof_with_skip_step, contains_correct_proof_with_non_atomic_step, contains_correct_proof_with_skip_step_or_non_atomic_step, wrong_branch_first, useful_skip_step_first, wrong_skip_step_first, useful_non_atomic_step_first, wrong_non_atomic_step_first, invalid_step_first, contains_any_wrong_branch, contains_any_useful_skip_step, contains_any_wrong_skip_step, contains_any_useful_non_atomic_step, contains_any_wrong_non_atomic_step, contains_any_invalid_step, wrong_branch_lengths) = analyze_log(logfile)

			correct_proof_count = np.sum(contains_correct_proof) + contains_correct_proof_with_skip_step_or_non_atomic_step
			example_count.append(num_examples)
			correct_proofs.append(correct_proof_count / num_examples)
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
		correct_proofs = np.array(correct_proofs)
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

		x1 = np.arange(len(correct_proofs))
		x2 = [x + bar_width for x in x1]
		x3 = [x + bar_width for x in x2]
		x4 = [x + bar_width for x in x3]
		x5 = [x + bar_width for x in x4]
		x6 = [x + bar_width for x in x5]

		fig = plt.gcf()
		fig.set_size_inches(10.0, figure_height, forward=True)
		plt.bar(x1, correct_proofs_wrong_branch, width=bar_width, color=lighten_color(colors[1], 1.3))
		plt.bar(x1, correct_proofs - correct_proofs_wrong_branch, width=bar_width, bottom=correct_proofs_wrong_branch, color=lighten_color(colors[1], 0.8))
		plt.bar(x1, incorrect_proofs_wrong_branch, width=bar_width, bottom=correct_proofs, color=lighten_color(colors[0], 1.3))
		plt.bar(x1, 1.0 - correct_proofs - incorrect_proofs_wrong_branch, width=bar_width, bottom=correct_proofs+incorrect_proofs_wrong_branch, color=lighten_color(colors[0], 0.8))

		plt.bar(x2, correct_proofs_useful_non_atomic_step, width=bar_width, color=lighten_color(colors[1], 1.3))
		plt.bar(x2, correct_proofs - correct_proofs_useful_non_atomic_step, width=bar_width, bottom=correct_proofs_useful_non_atomic_step, color=lighten_color(colors[1], 0.8))
		plt.bar(x2, incorrect_proofs_useful_non_atomic_step, width=bar_width, bottom=correct_proofs, color=lighten_color(colors[0], 1.3))
		plt.bar(x2, 1.0 - correct_proofs - incorrect_proofs_useful_non_atomic_step, width=bar_width, bottom=correct_proofs+incorrect_proofs_useful_non_atomic_step, color=lighten_color(colors[0], 0.8))

		plt.bar(x3, correct_proofs_wrong_non_atomic_step, width=bar_width, color=lighten_color(colors[1], 1.3))
		plt.bar(x3, correct_proofs - correct_proofs_wrong_non_atomic_step, width=bar_width, bottom=correct_proofs_wrong_non_atomic_step, color=lighten_color(colors[1], 0.8))
		plt.bar(x3, incorrect_proofs_wrong_non_atomic_step, width=bar_width, bottom=correct_proofs, color=lighten_color(colors[0], 1.3))
		plt.bar(x3, 1.0 - correct_proofs - incorrect_proofs_wrong_non_atomic_step, width=bar_width, bottom=correct_proofs+incorrect_proofs_wrong_non_atomic_step, color=lighten_color(colors[0], 0.8))

		plt.bar(x4, correct_proofs_useful_skip_step, width=bar_width, color=lighten_color(colors[1], 1.3))
		plt.bar(x4, correct_proofs - correct_proofs_useful_skip_step, width=bar_width, bottom=correct_proofs_useful_skip_step, color=lighten_color(colors[1], 0.8))
		plt.bar(x4, incorrect_proofs_useful_skip_step, width=bar_width, bottom=correct_proofs, color=lighten_color(colors[0], 1.3))
		plt.bar(x4, 1.0 - correct_proofs - incorrect_proofs_useful_skip_step, width=bar_width, bottom=correct_proofs+incorrect_proofs_useful_skip_step, color=lighten_color(colors[0], 0.8))

		plt.bar(x5, correct_proofs_wrong_skip_step, width=bar_width, color=lighten_color(colors[1], 1.3))
		plt.bar(x5, correct_proofs - correct_proofs_wrong_skip_step, width=bar_width, bottom=correct_proofs_wrong_skip_step, color=lighten_color(colors[1], 0.8))
		plt.bar(x5, incorrect_proofs_wrong_skip_step, width=bar_width, bottom=correct_proofs, color=lighten_color(colors[0], 1.3))
		plt.bar(x5, 1.0 - correct_proofs - incorrect_proofs_wrong_skip_step, width=bar_width, bottom=correct_proofs+incorrect_proofs_wrong_skip_step, color=lighten_color(colors[0], 0.8))

		plt.bar(x6, correct_proofs_invalid_step, width=bar_width, color=lighten_color(colors[1], 1.3))
		plt.bar(x6, correct_proofs - correct_proofs_invalid_step, width=bar_width, bottom=correct_proofs_invalid_step, color=lighten_color(colors[1], 0.8))
		plt.bar(x6, incorrect_proofs_invalid_step, width=bar_width, bottom=correct_proofs, color=lighten_color(colors[0], 1.3))
		plt.bar(x6, 1.0 - correct_proofs - incorrect_proofs_invalid_step, width=bar_width, bottom=correct_proofs+incorrect_proofs_invalid_step, color=lighten_color(colors[0], 0.8))

		# draw the error bars
		(lower_bound, upper_bound) = wilson_conf_interval(correct_proofs, example_count)
		plt.errorbar(x1 + (1.0 - bar_spacing - bar_width) / 2, correct_proofs, yerr=np.array((correct_proofs - lower_bound, upper_bound - correct_proofs)), fmt='none', ecolor=(0.0,0.0,0.0), capsize=3.0)

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
		plt.xlim([-bar_spacing, len(correct_proofs) - 0.12])
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
				proof_range = range(2, len(correct_proofs))
			else:
				fig.set_size_inches(10.0, wrong_branch_lengths_figure_height, forward=True)
				proof_range = range(len(correct_proofs))
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
				correct_proofs = correct_proofs[2:]
				incorrect_proofs_wrong_branch_first = incorrect_proofs_wrong_branch_first[2:]
				incorrect_proofs_useful_non_atomic_step_first = incorrect_proofs_useful_non_atomic_step_first[2:]
				incorrect_proofs_wrong_non_atomic_step_first = incorrect_proofs_wrong_non_atomic_step_first[2:]
				incorrect_proofs_useful_skip_step_first = incorrect_proofs_useful_skip_step_first[2:]
				incorrect_proofs_wrong_skip_step_first = incorrect_proofs_wrong_skip_step_first[2:]
				incorrect_proofs_invalid_step_first = incorrect_proofs_invalid_step_first[2:]
			else:
				fig.set_size_inches(10.0, first_error_figure_height, forward=True)
			incorrect_proofs = 1.0 - correct_proofs
			(lower_bound, upper_bound) = wilson_conf_interval(incorrect_proofs_wrong_branch_first/incorrect_proofs, example_count * incorrect_proofs)
			lower_bound[np.isnan(lower_bound)] = 0.0; upper_bound[np.isnan(upper_bound)] = 1.0
			err = np.array((incorrect_proofs_wrong_branch_first/incorrect_proofs - lower_bound, upper_bound - incorrect_proofs_wrong_branch_first/incorrect_proofs))
			plt.bar(x1, incorrect_proofs_wrong_branch_first/incorrect_proofs, width=bar_width, yerr=err, capsize=3.0, color=colors[5])
			(lower_bound, upper_bound) = wilson_conf_interval(incorrect_proofs_useful_non_atomic_step_first/incorrect_proofs, example_count * incorrect_proofs)
			lower_bound[np.isnan(lower_bound)] = 0.0; upper_bound[np.isnan(upper_bound)] = 1.0
			err = np.array((incorrect_proofs_useful_non_atomic_step_first/incorrect_proofs - lower_bound, upper_bound - incorrect_proofs_useful_non_atomic_step_first/incorrect_proofs))
			plt.bar(x2, incorrect_proofs_useful_non_atomic_step_first/incorrect_proofs, width=bar_width, yerr=err, capsize=3.0, color=lighten_color(colors[1], 0.8))
			(lower_bound, upper_bound) = wilson_conf_interval(incorrect_proofs_wrong_non_atomic_step_first/incorrect_proofs, example_count * incorrect_proofs)
			lower_bound[np.isnan(lower_bound)] = 0.0; upper_bound[np.isnan(upper_bound)] = 1.0
			err = np.array((incorrect_proofs_wrong_non_atomic_step_first/incorrect_proofs - lower_bound, upper_bound - incorrect_proofs_wrong_non_atomic_step_first/incorrect_proofs))
			plt.bar(x3, incorrect_proofs_wrong_non_atomic_step_first/incorrect_proofs, width=bar_width, yerr=err, capsize=3.0, color=lighten_color(colors[1], 1.2))
			(lower_bound, upper_bound) = wilson_conf_interval(incorrect_proofs_useful_skip_step_first/incorrect_proofs, example_count * incorrect_proofs)
			lower_bound[np.isnan(lower_bound)] = 0.0; upper_bound[np.isnan(upper_bound)] = 1.0
			err = np.array((incorrect_proofs_useful_skip_step_first/incorrect_proofs - lower_bound, upper_bound - incorrect_proofs_useful_skip_step_first/incorrect_proofs))
			plt.bar(x4, incorrect_proofs_useful_skip_step_first/incorrect_proofs, width=bar_width, yerr=err, capsize=3.0, color=lighten_color(colors[2], 0.8))
			(lower_bound, upper_bound) = wilson_conf_interval(incorrect_proofs_wrong_skip_step_first/incorrect_proofs, example_count * incorrect_proofs)
			lower_bound[np.isnan(lower_bound)] = 0.0; upper_bound[np.isnan(upper_bound)] = 1.0
			err = np.array((incorrect_proofs_wrong_skip_step_first/incorrect_proofs - lower_bound, upper_bound - incorrect_proofs_wrong_skip_step_first/incorrect_proofs))
			plt.bar(x5, incorrect_proofs_wrong_skip_step_first/incorrect_proofs, width=bar_width, yerr=err, capsize=3.0, color=lighten_color(colors[2], 1.2))
			(lower_bound, upper_bound) = wilson_conf_interval(incorrect_proofs_invalid_step_first/incorrect_proofs, example_count * incorrect_proofs)
			lower_bound[np.isnan(lower_bound)] = 0.0; upper_bound[np.isnan(upper_bound)] = 1.0
			err = np.array((incorrect_proofs_invalid_step_first/incorrect_proofs - lower_bound, upper_bound - incorrect_proofs_invalid_step_first/incorrect_proofs))
			plt.bar(x6, incorrect_proofs_invalid_step_first/incorrect_proofs, width=bar_width, yerr=err, capsize=3.0, color=colors[0])

			show_legend = False
			if show_legend:
				labels = ['strictly-valid atomic misleading steps', 'strictly-valid non-atomic correct steps', 'strictly-valid non-atomic misleading steps', 'broadly-valid correct steps', 'broadly-valid misleading steps', 'invalid steps']
				plt.legend(labels, loc='center left', bbox_to_anchor=(1.04, 0.5))

			ax = plt.gca()
			if show_first_error_ylabel:
				plt.ylabel('proportion of \n incorrect proofs')
			plt.xlim([-bar_spacing, len(correct_proofs) - 0.12])
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

	correct_proofs = []
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
	for logfile in logfiles:
		if "seed" in logfile:
			continue
		print('parsing "{}"'.format(logfile))
		(num_examples, proof_lengths, correct_labels, correct_step_count, non_atomic_step_count, skip_step_count, invalid_step_count, all_correct_and_useful_steps, all_redundant_steps, all_unparseable_steps, all_incorrect_steps, contains_correct_proof, does_not_contain_correct_proof, contains_wrong_branch, contains_useful_skip_step, contains_wrong_skip_step, contains_useful_non_atomic_step, contains_wrong_non_atomic_step, contains_invalid_step, contains_wrong_branch_or_useful_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_step, contains_wrong_branch_or_invalid_step, contains_wrong_branch_or_useful_non_atomic_or_invalid_step, contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step, contains_wrong_branch_or_useful_skip_step, contains_wrong_branch_or_wrong_skip_step, contains_useful_skip_or_wrong_skip_step, contains_useful_skip_or_useful_non_atomic_step, contains_useful_skip_or_wrong_non_atomic_step, contains_useful_skip_or_invalid_step, contains_wrong_skip_or_useful_non_atomic_step, contains_wrong_skip_or_wrong_non_atomic_step, contains_wrong_skip_or_invalid_step, contains_useful_non_atomic_or_wrong_non_atomic_step, contains_useful_non_atomic_or_invalid_step, contains_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_or_invalid_step, contains_correct_proof_with_skip_step, contains_correct_proof_with_non_atomic_step, contains_correct_proof_with_skip_step_or_non_atomic_step, wrong_branch_first, useful_skip_step_first, wrong_skip_step_first, useful_non_atomic_step_first, wrong_non_atomic_step_first, invalid_step_first, contains_any_wrong_branch, contains_any_useful_skip_step, contains_any_wrong_skip_step, contains_any_useful_non_atomic_step, contains_any_wrong_non_atomic_step, contains_any_invalid_step, wrong_branch_lengths) = analyze_log(logfile)
		print(np.sum(contains_wrong_branch_or_useful_skip_step + contains_wrong_branch_or_wrong_skip_step + contains_useful_skip_or_wrong_skip_step + contains_useful_skip_or_useful_non_atomic_step + contains_useful_skip_or_wrong_non_atomic_step + contains_useful_skip_or_invalid_step + contains_wrong_skip_or_useful_non_atomic_step + contains_wrong_skip_or_wrong_non_atomic_step + contains_wrong_skip_or_invalid_step + contains_useful_non_atomic_or_wrong_non_atomic_step + contains_useful_non_atomic_or_invalid_step + contains_wrong_non_atomic_or_invalid_step + contains_wrong_branch_or_non_atomic_step + contains_wrong_branch_or_wrong_non_atomic_or_invalid_step + contains_wrong_branch_or_non_atomic_or_invalid_step))

		correct_proof_count = np.sum(contains_correct_proof)
		example_count.append(num_examples)
		label_accuracy.append(correct_labels / num_examples)
		proof_accuracy.append(correct_proof_count / num_examples)
		proof_accuracy_with_skip_steps.append((correct_proof_count + contains_correct_proof_with_skip_step) / num_examples)
		proof_accuracy_with_non_atomic_steps.append((correct_proof_count + contains_correct_proof_with_non_atomic_step) / num_examples)
		proof_accuracy_with_skip_steps_and_non_atomic_steps.append((correct_proof_count + contains_correct_proof_with_skip_step_or_non_atomic_step) / num_examples)

	example_count = np.array(example_count)
	label_accuracy = np.array(label_accuracy)
	proof_accuracy = np.array(proof_accuracy)
	proof_accuracy_with_skip_steps = np.array(proof_accuracy_with_skip_steps)
	proof_accuracy_with_non_atomic_steps = np.array(proof_accuracy_with_non_atomic_steps)
	proof_accuracy_with_skip_steps_and_non_atomic_steps = np.array(proof_accuracy_with_skip_steps_and_non_atomic_steps)

	(label_lower_bound, label_upper_bound) = wilson_conf_interval(label_accuracy, example_count)

	plt.style.use('ggplot')
	colors = []
	for c in rcParams["axes.prop_cycle"]:
		colors.append(c['color'])

	fig = plt.gcf()
	fig.set_size_inches(2.2, 2.2, forward=True)
	plt.plot([0, 1], [0, 1], color='black')
	(proof_lower_bound, proof_upper_bound) = wilson_conf_interval(proof_accuracy, example_count)
	plt.errorbar(label_accuracy, proof_accuracy, xerr=np.array((label_accuracy - label_lower_bound, label_upper_bound - label_accuracy)), yerr=np.array((proof_accuracy - proof_lower_bound, proof_upper_bound - proof_accuracy)), fmt='none', ecolor=(0.53,0.53,0.53), elinewidth=0.8)
	plt.scatter(label_accuracy, proof_accuracy, zorder=10)
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
	plt.scatter(label_accuracy, proof_accuracy_with_skip_steps, zorder=10, c=colors[1])
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
	plt.scatter(label_accuracy, proof_accuracy_with_non_atomic_steps, zorder=10, c=colors[2])
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
	plt.scatter(label_accuracy, proof_accuracy_with_skip_steps_and_non_atomic_steps, zorder=10, c=colors[5])
	plt.xlabel('label accuracy')
	plt.ylabel('valid proof accuracy')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	fig.savefig('label_vs_proof_accuracy_with_skip_steps_and_non_atomic_steps.pdf', dpi=128, bbox_inches='tight')
	plt.clf()

	make_step_type_plot('Fictional ontology',
		['gpt_textdavinci002_1hop.log', 'gpt_textdavinci002_1hop_preorder.log', 'gpt_textdavinci002_3hop.log', 'gpt_textdavinci002_3hop_preorder.log', 'gpt_textdavinci002_5hop.log', 'gpt_textdavinci002_5hop_preorder.log'],
		['1 hop, bottom-up \n sentence ordering', '1 hop, top-down \n sentence ordering', '3 hops, bottom-up \n sentence ordering', '3 hops, top-down \n sentence ordering', '5 hops, bottom-up \n sentence ordering', '5 hops, top-down \n sentence ordering'],
		'textdavinci002_fictional_ontology_proof_accuracy.pdf', 'textdavinci002_fictional_ontology_first_error.pdf', 'textdavinci002_fictional_ontology_wrong_branch_lengths.pdf', first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.6)

	make_step_type_plot('False ontology',
		['gpt_textdavinci002_1hop_falseontology.log', 'gpt_textdavinci002_1hop_preorder_falseontology.log', 'gpt_textdavinci002_3hop_falseontology.log', 'gpt_textdavinci002_3hop_preorder_falseontology.log', 'gpt_textdavinci002_5hop_falseontology.log', 'gpt_textdavinci002_5hop_preorder_falseontology.log'],
		['1 hop, bottom-up \n sentence ordering', '1 hop, top-down \n sentence ordering', '3 hops, bottom-up \n sentence ordering', '3 hops, top-down \n sentence ordering', '5 hops, bottom-up \n sentence ordering', '5 hops, top-down \n sentence ordering'],
		'textdavinci002_false_ontology_proof_accuracy.pdf', 'textdavinci002_false_ontology_first_error.pdf', 'textdavinci002_false_ontology_wrong_branch_lengths.pdf', first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.6, show_ylabel=False, show_first_error_ylabel=False)

	make_step_type_plot('True ontology',
		['gpt_textdavinci002_1hop_trueontology.log', 'gpt_textdavinci002_1hop_preorder_trueontology.log', 'gpt_textdavinci002_3hop_trueontology.log', 'gpt_textdavinci002_3hop_preorder_trueontology.log', 'gpt_textdavinci002_5hop_trueontology.log', 'gpt_textdavinci002_5hop_preorder_trueontology.log'],
		['1 hop, bottom-up \n sentence ordering', '1 hop, top-down \n sentence ordering', '3 hops, bottom-up \n sentence ordering', '3 hops, top-down \n sentence ordering', '5 hops, bottom-up \n sentence ordering', '5 hops, top-down \n sentence ordering'],
		'textdavinci002_true_ontology_proof_accuracy.pdf', 'textdavinci002_true_ontology_first_error.pdf', 'textdavinci002_true_ontology_wrong_branch_lengths.pdf', first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.6, show_ylabel=False, show_first_error_ylabel=False)

	make_step_type_plot('Fictional ontology, 3 hops',
		['gpt_textada001_3hop_preorder.log', 'gpt_textbabbage001_3hop_preorder.log', 'gpt_textcurie001_3hop_preorder.log', 'gpt_davinci_3hop_preorder.log', 'gpt_textdavinci001_3hop_preorder.log', 'gpt_textdavinci002_3hop_preorder.log'],
		['\\texttt{text-ada-001}', '\\texttt{text-babbage-001}', '\\texttt{text-curie-001}', '\\texttt{davinci}', '\\texttt{text-davinci-001}', '\\texttt{text-davinci-002}'],
		'fictional_ontology_3hop_model_size.pdf', 'fictional_ontology_3hop_model_size_first_error.pdf', 'fictional_ontology_3hop_model_size_wrong_branch_lengths.pdf', figure_height=1.8, first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.6, show_first_error_ylabel=True, first_error_title='Fictional ontology, 3 hops, top-down sentence ordering')

	make_step_type_plot('False ontology, 3 hops',
		['gpt_textada001_3hop_preorder_falseontology.log', 'gpt_textbabbage001_3hop_preorder_falseontology.log', 'gpt_textcurie001_3hop_preorder_falseontology.log', 'gpt_davinci_3hop_preorder_falseontology.log', 'gpt_textdavinci001_3hop_preorder_falseontology.log', 'gpt_textdavinci002_3hop_preorder_falseontology.log'],
		['\\texttt{text-ada-001}', '\\texttt{text-babbage-001}', '\\texttt{text-curie-001}', '\\texttt{davinci}', '\\texttt{text-davinci-001}', '\\texttt{text-davinci-002}'],
		'false_ontology_3hop_model_size.pdf', 'false_ontology_3hop_model_size_first_error.pdf', 'false_ontology_3hop_model_size_wrong_branch_lengths.pdf', figure_height=1.8, first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.6, show_ylabel=False, show_first_error_ylabel=False, first_error_title='False ontology, 3 hops, top-down sentence ordering')

	make_step_type_plot('True ontology, 3 hops',
		['gpt_textada001_3hop_preorder_trueontology.log', 'gpt_textbabbage001_3hop_preorder_trueontology.log', 'gpt_textcurie001_3hop_preorder_trueontology.log', 'gpt_davinci_3hop_preorder_trueontology.log', 'gpt_textdavinci001_3hop_preorder_trueontology.log', 'gpt_textdavinci002_3hop_preorder_trueontology.log'],
		['\\texttt{text-ada-001}', '\\texttt{text-babbage-001}', '\\texttt{text-curie-001}', '\\texttt{davinci}', '\\texttt{text-davinci-001}', '\\texttt{text-davinci-002}'],
		'true_ontology_3hop_model_size.pdf', 'true_ontology_3hop_model_size_first_error.pdf', 'true_ontology_3hop_model_size_wrong_branch_lengths.pdf', figure_height=1.8, first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.6, show_ylabel=False, show_first_error_ylabel=False, first_error_title='True ontology, 3 hops, top-down sentence ordering')

	make_step_type_plot('Fictional ontology, 1 hop',
		['gpt_textada001_1hop_preorder.log', 'gpt_textbabbage001_1hop_preorder.log', 'gpt_textcurie001_1hop_preorder.log', 'gpt_davinci_1hop_preorder.log', 'gpt_textdavinci001_1hop_preorder.log', 'gpt_textdavinci002_1hop_preorder.log'],
		['\\texttt{text-ada-001}', '\\texttt{text-babbage-001}', '\\texttt{text-curie-001}', '\\texttt{davinci}', '\\texttt{text-davinci-001}', '\\texttt{text-davinci-002}'],
		'fictional_ontology_1hop_model_size.pdf', 'fictional_ontology_1hop_model_size_first_error.pdf', 'fictional_ontology_1hop_model_size_wrong_branch_lengths.pdf', figure_height=1.8, first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.6, show_ylabel=False, show_first_error_ylabel=False, first_error_title='Fictional ontology, 1 hop, top-down sentence ordering')

	make_step_type_plot('False ontology, 1 hop',
		['gpt_textada001_1hop_preorder_falseontology.log', 'gpt_textbabbage001_1hop_preorder_falseontology.log', 'gpt_textcurie001_1hop_preorder_falseontology.log', 'gpt_davinci_1hop_preorder_falseontology.log', 'gpt_textdavinci001_1hop_preorder_falseontology.log', 'gpt_textdavinci002_1hop_preorder_falseontology.log'],
		['\\texttt{text-ada-001}', '\\texttt{text-babbage-001}', '\\texttt{text-curie-001}', '\\texttt{davinci}', '\\texttt{text-davinci-001}', '\\texttt{text-davinci-002}'],
		'false_ontology_1hop_model_size.pdf', 'false_ontology_1hop_model_size_first_error.pdf', 'false_ontology_1hop_model_size_wrong_branch_lengths.pdf', figure_height=1.8, first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.6, show_ylabel=False, show_first_error_ylabel=False, first_error_title='False ontology, 1 hop, top-down sentence ordering')

	make_step_type_plot('True ontology, 1 hop',
		['gpt_textada001_1hop_preorder_trueontology.log', 'gpt_textbabbage001_1hop_preorder_trueontology.log', 'gpt_textcurie001_1hop_preorder_trueontology.log', 'gpt_davinci_1hop_preorder_trueontology.log', 'gpt_textdavinci001_1hop_preorder_trueontology.log', 'gpt_textdavinci002_1hop_trueontology.log'],
		['\\texttt{text-ada-001}', '\\texttt{text-babbage-001}', '\\texttt{text-curie-001}', '\\texttt{davinci}', '\\texttt{text-davinci-001}', '\\texttt{text-davinci-002}'],
		'true_ontology_1hop_model_size.pdf', 'true_ontology_1hop_model_size_first_error.pdf', 'true_ontology_1hop_model_size_wrong_branch_lengths.pdf', figure_height=1.8, first_error_figure_height=1.8, wrong_branch_lengths_figure_height=1.6, show_ylabel=False, show_first_error_ylabel=False, first_error_title='True ontology, 1 hop, top-down sentence ordering')
