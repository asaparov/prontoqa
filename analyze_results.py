import numpy as np
from main import parse_log
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
	wrong_branch_proofs = [0] * 8

	for result in results:
		(label, correct_steps, correct_and_useful_steps, redundant_steps, unparseable_steps, wrong_branch_steps, useful_skip_steps, wrong_skip_steps, useful_non_atomic_steps, wrong_non_atomic_steps, invalid_steps, incorrect_steps, found_conclusion, found_conclusion_with_skip_steps, found_conclusion_with_non_atomic_steps) = result
		all_correct_steps.extend(correct_steps)
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

		if len(wrong_branch_steps) != 0:
			increment_count(wrong_branch_proofs)
		if len(wrong_branch_steps) != 0 and (len(useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + invalid_steps) == 0 or min(wrong_branch_steps) <= min(useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + invalid_steps)):
			increment_count(wrong_branch_first)

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
		correct_labels += label

	return (len(results), proof_lengths, correct_labels, all_correct_and_useful_steps, all_redundant_steps, all_unparseable_steps, all_incorrect_steps, contains_correct_proof, does_not_contain_correct_proof, contains_wrong_branch, contains_useful_skip_step, contains_wrong_skip_step, contains_useful_non_atomic_step, contains_wrong_non_atomic_step, contains_invalid_step, contains_wrong_branch_or_useful_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_step, contains_wrong_branch_or_invalid_step, contains_wrong_branch_or_useful_non_atomic_or_invalid_step, contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step, contains_wrong_branch_or_useful_skip_step, contains_wrong_branch_or_wrong_skip_step, contains_useful_skip_or_wrong_skip_step, contains_useful_skip_or_useful_non_atomic_step, contains_useful_skip_or_wrong_non_atomic_step, contains_useful_skip_or_invalid_step, contains_wrong_skip_or_useful_non_atomic_step, contains_wrong_skip_or_wrong_non_atomic_step, contains_wrong_skip_or_invalid_step, contains_useful_non_atomic_or_wrong_non_atomic_step, contains_useful_non_atomic_or_invalid_step, contains_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_or_invalid_step, contains_correct_proof_with_skip_step, contains_correct_proof_with_non_atomic_step, contains_correct_proof_with_skip_step_or_non_atomic_step, wrong_branch_first, wrong_branch_proofs, contains_any_wrong_branch, contains_any_useful_skip_step, contains_any_wrong_skip_step, contains_any_useful_non_atomic_step, contains_any_wrong_non_atomic_step, contains_any_invalid_step)

if len(argv) > 1:
	(num_examples, proof_lengths, correct_labels, all_correct_and_useful_steps, all_redundant_steps, all_unparseable_steps, all_incorrect_steps, contains_correct_proof, does_not_contain_correct_proof, contains_wrong_branch, contains_useful_skip_step, contains_wrong_skip_step, contains_useful_non_atomic_step, contains_wrong_non_atomic_step, contains_invalid_step, contains_wrong_branch_or_useful_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_step, contains_wrong_branch_or_invalid_step, contains_wrong_branch_or_useful_non_atomic_or_invalid_step, contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step, contains_wrong_branch_or_useful_skip_step, contains_wrong_branch_or_wrong_skip_step, contains_useful_skip_or_wrong_skip_step, contains_useful_skip_or_useful_non_atomic_step, contains_useful_skip_or_wrong_non_atomic_step, contains_useful_skip_or_invalid_step, contains_wrong_skip_or_useful_non_atomic_step, contains_wrong_skip_or_wrong_non_atomic_step, contains_wrong_skip_or_invalid_step, contains_useful_non_atomic_or_wrong_non_atomic_step, contains_useful_non_atomic_or_invalid_step, contains_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_or_invalid_step, contains_correct_proof_with_skip_step, contains_correct_proof_with_non_atomic_step, contains_correct_proof_with_skip_step_or_non_atomic_step, wrong_branch_first, wrong_branch_proofs, contains_any_wrong_branch, contains_any_useful_skip_step, contains_any_wrong_skip_step, contains_any_useful_non_atomic_step, contains_any_wrong_non_atomic_step, contains_any_invalid_step) = analyze_log(argv[1])

	max_proof_length = max(proof_lengths)
	total_steps = np.sum(proof_lengths)
	print("Correct and useful steps: {}".format(len(all_correct_and_useful_steps) / total_steps))
	print("Redundant steps: {}".format(len(all_redundant_steps) / total_steps))
	print("Unparseable steps: {}".format(len(all_unparseable_steps) / total_steps))
	print("Incorrect steps: {}".format(len(all_incorrect_steps) / total_steps))

	correct_proofs = np.sum(contains_correct_proof)
	print("Proportion of proofs that contain the correct proof: {}".format(correct_proofs / num_examples))

	print("Proportion of proofs with the correct label that contain the correct proof:   {}".format(contains_correct_proof[1] / correct_labels))
	print("Proportion of proofs with the incorrect label that contain the correct proof: {}".format(contains_correct_proof[0] / (num_examples - correct_labels)))
	print("Proportion of proofs with the correct label that do NOT contain the correct proof:   {}".format(does_not_contain_correct_proof[1] / correct_labels))
	print("Proportion of proofs with the incorrect label that do NOT contain the correct proof: {}".format(does_not_contain_correct_proof[0] / (num_examples - correct_labels)))

	print("Proportion of correct proofs that contain a \"useless branch\":          {}".format(contains_wrong_branch[1] / correct_proofs))
	print("Proportion of correct proofs that contain a \"useful skip step\":        {}".format(contains_useful_skip_step[1] / correct_proofs))
	print("Proportion of correct proofs that contain a \"useless skip step\":       {}".format(contains_wrong_skip_step[1] / correct_proofs))
	print("Proportion of correct proofs that contain a \"useful non-atomic step\":  {}".format(contains_useful_non_atomic_step[1] / correct_proofs))
	print("Proportion of correct proofs that contain a \"useless non-atomic step\": {}".format(contains_wrong_non_atomic_step[1] / correct_proofs))
	print("Proportion of correct proofs that contain an \"invalid step\":           {}".format(contains_invalid_step[1] / correct_proofs))
	print("Proportion of correct proofs that contain a \"useless branch\" AND \"useful non-atomic step\": {}".format(contains_wrong_branch_or_useful_non_atomic_step[1] / correct_proofs))
	print("Proportion of correct proofs that contain a \"useless branch\" AND \"useless non-atomic step\": {}".format(contains_wrong_branch_or_wrong_non_atomic_step[1] / correct_proofs))
	print("Proportion of correct proofs that contain a \"useless branch\" AND \"invalid step\": {}".format(contains_wrong_branch_or_invalid_step[1] / (num_examples - correct_proofs)))
	print("Proportion of correct proofs that contain a \"useless branch\" AND \"useful non-atomic step\" AND \"invalid step\": {}".format(contains_wrong_branch_or_useful_non_atomic_or_invalid_step[1] / correct_proofs))
	print("Proportion of correct proofs with ANY OF THE ABOVE:                    {}".format(contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step[1] / correct_proofs))
	print("Proportion of incorrect proofs that contain a \"useless branch\":          {}".format(contains_wrong_branch[0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs that contain a \"useful skip step\":        {}".format(contains_useful_skip_step[0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs that contain a \"useless skip step\":       {}".format(contains_wrong_skip_step[0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs that contain a \"useful non-atomic step\":  {}".format(contains_useful_non_atomic_step[0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs that contain a \"useless non-atomic step\": {}".format(contains_wrong_non_atomic_step[0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs that contain an \"invalid step\":           {}".format(contains_invalid_step[0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs that contain a \"useless branch\" AND \"useful non-atomic step\": {}".format(contains_wrong_branch_or_useful_non_atomic_step[0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs that contain a \"useless branch\" AND \"useless non-atomic step\": {}".format(contains_wrong_branch_or_wrong_non_atomic_step[0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs that contain a \"useless branch\" AND \"invalid step\": {}".format(contains_wrong_branch_or_invalid_step[0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs that contain a \"useless branch\" AND \"useful non-atomic step\" AND \"invalid step\": {}".format(contains_wrong_branch_or_useful_non_atomic_or_invalid_step[0] / (num_examples - correct_proofs)))
	print("Proportion of incorrect proofs with ANY OF THE ABOVE:                    {}".format(contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step[0] / (num_examples - correct_proofs)))
	print("Proportion of ALL proofs that would be correct if \"skip steps\" are considered correct: {}".format((correct_proofs + contains_correct_proof_with_skip_step) / num_examples))
	print("Proportion of ALL proofs that would be correct if \"non-atomic steps\" are considered correct: {}".format((correct_proofs + contains_correct_proof_with_non_atomic_step) / num_examples))
	print("Proportion of ALL proofs that would be correct if both \"skip steps\" and \"non-atomic steps\" are considered correct: {}".format((correct_proofs + contains_correct_proof_with_skip_step_or_non_atomic_step) / num_examples))
	print("Proportion of correct proofs with a \"useless branch\" where the \"useless branch\" is the first mistake: {}".format('nan' if wrong_branch_proofs[1] == 0 else (wrong_branch_first[1] / wrong_branch_proofs[1])))
	print("Proportion of incorrect proofs with a \"useless branch\" where the \"useless branch\" is the first mistake: {}".format(wrong_branch_first[0] / wrong_branch_proofs[0]))
	print("Proportion of incorrect proofs where the \"useless branch\" is the first mistake: {}".format(wrong_branch_first[0] / (num_examples - correct_proofs)))

	print(contains_wrong_branch_or_useful_skip_step[1] / correct_proofs)
	print(contains_wrong_branch_or_wrong_skip_step[1] / correct_proofs)
	print(contains_useful_skip_or_wrong_skip_step[1] / correct_proofs)
	print(contains_useful_skip_or_useful_non_atomic_step[1] / correct_proofs)
	print(contains_useful_skip_or_wrong_non_atomic_step[1] / correct_proofs)
	print(contains_useful_skip_or_invalid_step[1] / correct_proofs)
	print(contains_wrong_skip_or_useful_non_atomic_step[1] / correct_proofs)
	print(contains_wrong_skip_or_wrong_non_atomic_step[1] / correct_proofs)
	print(contains_wrong_skip_or_invalid_step[1] / correct_proofs)
	print(contains_useful_non_atomic_or_wrong_non_atomic_step[1] / correct_proofs)
	print(contains_useful_non_atomic_or_invalid_step[1] / correct_proofs)
	print(contains_wrong_non_atomic_or_invalid_step[1] / correct_proofs)
	print(contains_wrong_branch_or_non_atomic_step[1] / correct_proofs)
	print(contains_wrong_branch_or_wrong_non_atomic_or_invalid_step[1] / correct_proofs)
	print(contains_wrong_branch_or_non_atomic_or_invalid_step[1] / correct_proofs)

	print(contains_wrong_branch_or_useful_skip_step[0] / (num_examples - correct_proofs))
	print(contains_wrong_branch_or_wrong_skip_step[0] / (num_examples - correct_proofs))
	print(contains_useful_skip_or_wrong_skip_step[0] / (num_examples - correct_proofs))
	print(contains_useful_skip_or_useful_non_atomic_step[0] / (num_examples - correct_proofs))
	print(contains_useful_skip_or_wrong_non_atomic_step[0] / (num_examples - correct_proofs))
	print(contains_useful_skip_or_invalid_step[0] / (num_examples - correct_proofs))
	print(contains_wrong_skip_or_useful_non_atomic_step[0] / (num_examples - correct_proofs))
	print(contains_wrong_skip_or_wrong_non_atomic_step[0] / (num_examples - correct_proofs))
	print(contains_wrong_skip_or_invalid_step[0] / (num_examples - correct_proofs))
	print(contains_useful_non_atomic_or_wrong_non_atomic_step[0] / (num_examples - correct_proofs))
	print(contains_useful_non_atomic_or_invalid_step[0] / (num_examples - correct_proofs))
	print(contains_wrong_non_atomic_or_invalid_step[0] / (num_examples - correct_proofs))
	print(contains_wrong_branch_or_non_atomic_step[0] / (num_examples - correct_proofs))
	print(contains_wrong_branch_or_wrong_non_atomic_or_invalid_step[0] / (num_examples - correct_proofs))
	print(contains_wrong_branch_or_non_atomic_or_invalid_step[0] / (num_examples - correct_proofs))

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

	def make_barplot(chart_title, filename_glob, group_labels, chart_filename, add_bar_legend=False):
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
				return result_array[index % 2] + result_array[index]

		experiment_names = []
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
		example_count = []
		offset = 6
		for logfile in logfiles:
			print('parsing "{}"'.format(logfile))
			(num_examples, proof_lengths, correct_labels, all_correct_and_useful_steps, all_redundant_steps, all_unparseable_steps, all_incorrect_steps, contains_correct_proof, does_not_contain_correct_proof, contains_wrong_branch, contains_useful_skip_step, contains_wrong_skip_step, contains_useful_non_atomic_step, contains_wrong_non_atomic_step, contains_invalid_step, contains_wrong_branch_or_useful_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_step, contains_wrong_branch_or_invalid_step, contains_wrong_branch_or_useful_non_atomic_or_invalid_step, contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step, contains_wrong_branch_or_useful_skip_step, contains_wrong_branch_or_wrong_skip_step, contains_useful_skip_or_wrong_skip_step, contains_useful_skip_or_useful_non_atomic_step, contains_useful_skip_or_wrong_non_atomic_step, contains_useful_skip_or_invalid_step, contains_wrong_skip_or_useful_non_atomic_step, contains_wrong_skip_or_wrong_non_atomic_step, contains_wrong_skip_or_invalid_step, contains_useful_non_atomic_or_wrong_non_atomic_step, contains_useful_non_atomic_or_invalid_step, contains_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_or_invalid_step, contains_correct_proof_with_skip_step, contains_correct_proof_with_non_atomic_step, contains_correct_proof_with_skip_step_or_non_atomic_step, wrong_branch_first, wrong_branch_proofs, contains_any_wrong_branch, contains_any_useful_skip_step, contains_any_wrong_skip_step, contains_any_useful_non_atomic_step, contains_any_wrong_non_atomic_step, contains_any_invalid_step) = analyze_log(logfile)
			experiment_names.append(logfile)

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

		bar_group_size = 6
		bar_spacing = 0.4
		bar_width = (1.0 - bar_spacing) / bar_group_size

		x1 = np.arange(len(experiment_names))
		x2 = [x + bar_width for x in x1]
		x3 = [x + bar_width for x in x2]
		x4 = [x + bar_width for x in x3]
		x5 = [x + bar_width for x in x4]
		x6 = [x + bar_width for x in x5]

		fig = plt.gcf()
		fig.set_size_inches(10.0, 2.4, forward=True)
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
		plt.errorbar(x1 + (1.0 - bar_spacing - bar_width) / 2, correct_proofs, yerr=np.array((correct_proofs - lower_bound, upper_bound - correct_proofs)), fmt='none', ecolor='#000', capsize=3.0)

		ax = plt.gca()
		if offset == 0:
			plt.ylabel('strict proof accuracy')
		elif offset == 2:
			plt.ylabel('proof accuracy if \n ``skip steps\'\' are correct')
		elif offset == 4:
			plt.ylabel('proof accuracy if \n non-atomic steps are correct')
		elif offset == 6:
			plt.ylabel('proof accuracy if \n both ``skip steps\'\' and \n non-atomic steps are correct')
		plt.ylim([0.0, 1.0])
		plt.title(chart_title)
		delta = (1.0 - bar_spacing) / (3 * bar_group_size)
		plt.xticks([x + ((1.0 - bar_spacing) / 2) - delta for x in x1], group_labels)
		plt.tick_params(axis='x', which='both', length=0)
		if add_bar_legend:
			plt.text(0.0 + 0*bar_width - delta, 1.0, "misleading branches", rotation=70, color=(0.3,0.3,0.3), fontsize=8.0, horizontalalignment='left', verticalalignment='bottom')
			plt.text(0.0 + 1*bar_width - delta, 1.0, "correct non-atomic steps", rotation=70, color=(0.3,0.3,0.3), fontsize=8.0, horizontalalignment='left', verticalalignment='bottom')
			plt.text(0.0 + 2*bar_width - delta, 1.0, "misleading non-atomic steps", rotation=70, color=(0.3,0.3,0.3), fontsize=8.0, horizontalalignment='left', verticalalignment='bottom')
			plt.text(0.0 + 3*bar_width - delta, 1.0, "correct ``skip steps''", rotation=70, color=(0.3,0.3,0.3), fontsize=8.0, horizontalalignment='left', verticalalignment='bottom')
			plt.text(0.0 + 4*bar_width - delta, 1.0, "misleading ``skip steps''", rotation=70, color=(0.3,0.3,0.3), fontsize=8.0, horizontalalignment='left', verticalalignment='bottom')
			plt.text(0.0 + 5*bar_width - delta, 1.0, "invalid steps", rotation=70, color=(0.3,0.3,0.3), fontsize=8.0, horizontalalignment='left', verticalalignment='bottom')
		fig.savefig(chart_filename, dpi=128, bbox_inches=Bbox([[0.3, -0.1], [10.0 - 0.3, 2.4]]))
		plt.clf()


	logfiles = glob.glob('gpt_*.log')
	example_count = []
	label_accuracy = []
	proof_accuracy = []
	proof_accuracy_with_skip_steps = []
	proof_accuracy_with_non_atomic_steps = []
	proof_accuracy_with_skip_steps_and_non_atomic_steps = []

	experiment_names = []
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
		print('parsing "{}"'.format(logfile))
		(num_examples, proof_lengths, correct_labels, all_correct_and_useful_steps, all_redundant_steps, all_unparseable_steps, all_incorrect_steps, contains_correct_proof, does_not_contain_correct_proof, contains_wrong_branch, contains_useful_skip_step, contains_wrong_skip_step, contains_useful_non_atomic_step, contains_wrong_non_atomic_step, contains_invalid_step, contains_wrong_branch_or_useful_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_step, contains_wrong_branch_or_invalid_step, contains_wrong_branch_or_useful_non_atomic_or_invalid_step, contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step, contains_wrong_branch_or_useful_skip_step, contains_wrong_branch_or_wrong_skip_step, contains_useful_skip_or_wrong_skip_step, contains_useful_skip_or_useful_non_atomic_step, contains_useful_skip_or_wrong_non_atomic_step, contains_useful_skip_or_invalid_step, contains_wrong_skip_or_useful_non_atomic_step, contains_wrong_skip_or_wrong_non_atomic_step, contains_wrong_skip_or_invalid_step, contains_useful_non_atomic_or_wrong_non_atomic_step, contains_useful_non_atomic_or_invalid_step, contains_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_or_invalid_step, contains_correct_proof_with_skip_step, contains_correct_proof_with_non_atomic_step, contains_correct_proof_with_skip_step_or_non_atomic_step, wrong_branch_first, wrong_branch_proofs, contains_any_wrong_branch, contains_any_useful_skip_step, contains_any_wrong_skip_step, contains_any_useful_non_atomic_step, contains_any_wrong_non_atomic_step, contains_any_invalid_step) = analyze_log(logfile)
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
	fig.set_size_inches(3.5, 3.5, forward=True)
	plt.plot([0, 1], [0, 1], color='black')
	(proof_lower_bound, proof_upper_bound) = wilson_conf_interval(proof_accuracy, example_count)
	plt.errorbar(label_accuracy, proof_accuracy, xerr=np.array((label_accuracy - label_lower_bound, label_upper_bound - label_accuracy)), yerr=np.array((proof_accuracy - proof_lower_bound, proof_upper_bound - proof_accuracy)), fmt='none', ecolor='#888', elinewidth=0.8)
	plt.scatter(label_accuracy, proof_accuracy, zorder=10)
	plt.xlabel('label accuracy')
	plt.ylabel('strict proof accuracy')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	fig.savefig('label_vs_proof_accuracy.pdf', dpi=128, bbox_inches='tight')
	plt.clf()

	fig = plt.gcf()
	fig.set_size_inches(3.5, 3.5, forward=True)
	plt.plot([0, 1], [0, 1], color='black')
	(proof_lower_bound, proof_upper_bound) = wilson_conf_interval(proof_accuracy_with_skip_steps, example_count)
	plt.errorbar(label_accuracy, proof_accuracy_with_skip_steps, xerr=np.array((label_accuracy - label_lower_bound, label_upper_bound - label_accuracy)), yerr=np.array((proof_accuracy_with_skip_steps - proof_lower_bound, proof_upper_bound - proof_accuracy_with_skip_steps)), fmt='none', ecolor='#888', elinewidth=0.8)
	plt.scatter(label_accuracy, proof_accuracy_with_skip_steps, zorder=10, c=colors[1])
	plt.xlabel('label accuracy')
	plt.ylabel('proof accuracy if \n ``skip steps\'\' are correct')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	fig.savefig('label_vs_proof_accuracy_with_skip_steps.pdf', dpi=128, bbox_inches='tight')
	plt.clf()

	fig = plt.gcf()
	fig.set_size_inches(3.5, 3.5, forward=True)
	plt.plot([0, 1], [0, 1], color='black')
	(proof_lower_bound, proof_upper_bound) = wilson_conf_interval(proof_accuracy_with_non_atomic_steps, example_count)
	plt.errorbar(label_accuracy, proof_accuracy_with_non_atomic_steps, xerr=np.array((label_accuracy - label_lower_bound, label_upper_bound - label_accuracy)), yerr=np.array((proof_accuracy_with_non_atomic_steps - proof_lower_bound, proof_upper_bound - proof_accuracy_with_non_atomic_steps)), fmt='none', ecolor='#888', elinewidth=0.8)
	plt.scatter(label_accuracy, proof_accuracy_with_non_atomic_steps, zorder=10, c=colors[2])
	plt.xlabel('label accuracy')
	plt.ylabel('proof accuracy if \n non-atomic steps are correct')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	fig.savefig('label_vs_proof_accuracy_with_non_atomic_steps.pdf', dpi=128, bbox_inches='tight')
	plt.clf()

	fig = plt.gcf()
	fig.set_size_inches(3.5, 3.5, forward=True)
	plt.plot([0, 1], [0, 1], color='black')
	(proof_lower_bound, proof_upper_bound) = wilson_conf_interval(proof_accuracy_with_skip_steps_and_non_atomic_steps, example_count)
	plt.errorbar(label_accuracy, proof_accuracy_with_skip_steps_and_non_atomic_steps, xerr=np.array((label_accuracy - label_lower_bound, label_upper_bound - label_accuracy)), yerr=np.array((proof_accuracy_with_skip_steps_and_non_atomic_steps - proof_lower_bound, proof_upper_bound - proof_accuracy_with_skip_steps_and_non_atomic_steps)), fmt='none', ecolor='#888', elinewidth=0.8)
	plt.scatter(label_accuracy, proof_accuracy_with_skip_steps_and_non_atomic_steps, zorder=10, c=colors[5])
	plt.xlabel('label accuracy')
	plt.ylabel('proof accuracy if \n both ``skip steps\'\' and \n non-atomic steps are correct')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	fig.savefig('label_vs_proof_accuracy_with_skip_steps_and_non_atomic_steps.pdf', dpi=128, bbox_inches='tight')
	plt.clf()

	make_barplot('Fictional ontology',
		['gpt_textdavinci002_1hop.log', 'gpt_textdavinci002_1hop_preorder.log', 'gpt_textdavinci002_3hop.log', 'gpt_textdavinci002_3hop_preorder.log', 'gpt_textdavinci002_5hop.log', 'gpt_textdavinci002_5hop_preorder.log'],
		['1 hop, bottom-up \n sentence ordering', '1 hop, top-down \n sentence ordering', '3 hops, bottom-up \n sentence ordering', '3 hops, top-down \n sentence ordering', '5 hops, bottom-up \n sentence ordering', '5 hops, top-down \n sentence ordering'],
		'textdavinci002_fictional_ontology_proof_accuracy.pdf')

	make_barplot('False ontology',
		['gpt_textdavinci002_1hop_falseontology.log', 'gpt_textdavinci002_1hop_preorder_falseontology.log', 'gpt_textdavinci002_3hop_falseontology.log', 'gpt_textdavinci002_3hop_preorder_falseontology.log', 'gpt_textdavinci002_5hop_falseontology.log', 'gpt_textdavinci002_5hop_preorder_falseontology.log'],
		['1 hop, bottom-up \n sentence ordering', '1 hop, top-down \n sentence ordering', '3 hops, bottom-up \n sentence ordering', '3 hops, top-down \n sentence ordering', '5 hops, bottom-up \n sentence ordering', '5 hops, top-down \n sentence ordering'],
		'textdavinci002_false_ontology_proof_accuracy.pdf')

	make_barplot('True ontology',
		['gpt_textdavinci002_1hop_trueontology.log', 'gpt_textdavinci002_1hop_preorder_realontology.log', 'gpt_textdavinci002_3hop_realontology.log', 'gpt_textdavinci002_3hop_preorder_realontology.log', 'gpt_textdavinci002_5hop_trueontology.log', 'gpt_textdavinci002_5hop_preorder_trueontology.log'],
		['1 hop, bottom-up \n sentence ordering', '1 hop, top-down \n sentence ordering', '3 hops, bottom-up \n sentence ordering', '3 hops, top-down \n sentence ordering', '5 hops, bottom-up \n sentence ordering', '5 hops, top-down \n sentence ordering'],
		'textdavinci002_true_ontology_proof_accuracy.pdf')
