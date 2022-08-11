import numpy as np
from main import parse_log
from sys import argv

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
	contains_correct_proof = [0, 0]
	does_not_contain_correct_proof = [0, 0]
	contains_wrong_branch = [0, 0]
	contains_useful_skip_step = [0, 0]
	contains_wrong_skip_step = [0, 0]
	contains_useful_non_atomic_step = [0, 0]
	contains_wrong_non_atomic_step = [0, 0]
	contains_invalid_step = [0, 0]
	contains_wrong_branch_or_useful_non_atomic_step = [0, 0]
	contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step = [0, 0]
	contains_correct_proof_with_skip_step = 0
	contains_correct_proof_with_non_atomic_step = 0
	contains_correct_proof_with_skip_step_or_non_atomic_step = 0
	correct_labels = 0

	contains_wrong_branch_or_useful_skip_step = [0, 0]
	contains_wrong_branch_or_wrong_skip_step = [0, 0]
	contains_wrong_branch_or_wrong_non_atomic_step = [0, 0]
	contains_wrong_branch_or_invalid_step = [0, 0]
	contains_useful_skip_or_wrong_skip_step = [0, 0]
	contains_useful_skip_or_useful_non_atomic_step = [0, 0]
	contains_useful_skip_or_wrong_non_atomic_step = [0, 0]
	contains_useful_skip_or_invalid_step = [0, 0]
	contains_wrong_skip_or_useful_non_atomic_step = [0, 0]
	contains_wrong_skip_or_wrong_non_atomic_step = [0, 0]
	contains_wrong_skip_or_invalid_step = [0, 0]
	contains_useful_non_atomic_or_wrong_non_atomic_step = [0, 0]
	contains_useful_non_atomic_or_invalid_step = [0, 0]
	contains_wrong_non_atomic_or_invalid_step = [0, 0]

	contains_wrong_branch_or_non_atomic_step = [0, 0]
	contains_wrong_branch_or_useful_non_atomic_or_invalid_step = [0, 0]
	contains_wrong_branch_or_wrong_non_atomic_or_invalid_step = [0, 0]

	contains_wrong_branch_or_non_atomic_or_invalid_step = [0, 0]

	wrong_branch_first = [0, 0]
	wrong_branch_proofs = [0, 0]

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

		if found_conclusion_with_skip_steps:
			contains_correct_proof_with_skip_step += 1
		if found_conclusion_with_non_atomic_steps:
			contains_correct_proof_with_non_atomic_step += 1
		if found_conclusion_with_skip_steps or found_conclusion_with_non_atomic_steps:
			contains_correct_proof_with_skip_step_or_non_atomic_step += 1

		if len(wrong_branch_steps) != 0:
			wrong_branch_proofs[int(found_conclusion)] += 1
		if len(wrong_branch_steps) != 0 and (len(useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + invalid_steps) == 0 or min(wrong_branch_steps) <= min(useful_skip_steps + wrong_skip_steps + useful_non_atomic_steps + wrong_non_atomic_steps + invalid_steps)):
			wrong_branch_first[int(found_conclusion)] += 1

		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			contains_wrong_branch[int(found_conclusion)] += 1
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) != 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			contains_useful_skip_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) != 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			contains_wrong_skip_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			contains_useful_non_atomic_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) != 0 and len(invalid_steps) == 0:
			contains_wrong_non_atomic_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) != 0:
			contains_invalid_step[int(found_conclusion)] += 1

		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) != 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			contains_wrong_branch_or_useful_skip_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) != 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			contains_wrong_branch_or_wrong_skip_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			contains_wrong_branch_or_useful_non_atomic_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) != 0 and len(invalid_steps) == 0:
			contains_wrong_branch_or_wrong_non_atomic_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) != 0:
			contains_wrong_branch_or_invalid_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) != 0 and len(wrong_skip_steps) != 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			contains_useful_skip_or_wrong_skip_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) != 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			contains_useful_skip_or_useful_non_atomic_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) != 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) != 0 and len(invalid_steps) == 0:
			contains_useful_skip_or_wrong_non_atomic_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) != 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) != 0:
			contains_useful_skip_or_invalid_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) != 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) == 0:
			contains_wrong_skip_or_useful_non_atomic_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) != 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) != 0 and len(invalid_steps) == 0:
			contains_wrong_skip_or_wrong_non_atomic_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) != 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) != 0:
			contains_wrong_skip_or_invalid_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) != 0 and len(invalid_steps) == 0:
			contains_useful_non_atomic_or_wrong_non_atomic_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) != 0:
			contains_useful_non_atomic_or_invalid_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) == 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) != 0 and len(invalid_steps) != 0:
			contains_wrong_non_atomic_or_invalid_step[int(found_conclusion)] += 1

		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) != 0 and len(invalid_steps) == 0:
			contains_wrong_branch_or_non_atomic_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) == 0 and len(invalid_steps) != 0:
			contains_wrong_branch_or_useful_non_atomic_or_invalid_step[int(found_conclusion)] += 1
		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) == 0 and len(wrong_non_atomic_steps) != 0 and len(invalid_steps) != 0:
			contains_wrong_branch_or_wrong_non_atomic_or_invalid_step[int(found_conclusion)] += 1

		if len(wrong_branch_steps) != 0 and len(useful_skip_steps) == 0 and len(wrong_skip_steps) == 0 and len(useful_non_atomic_steps) != 0 and len(wrong_non_atomic_steps) != 0 and len(invalid_steps) != 0:
			contains_wrong_branch_or_non_atomic_or_invalid_step[int(found_conclusion)] += 1

		if len(wrong_branch_steps) != 0 or len(useful_skip_steps) != 0 or len(wrong_skip_steps) != 0 or len(useful_non_atomic_steps) != 0 or len(wrong_non_atomic_steps) != 0 or len(invalid_steps) != 0:
			contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step[int(found_conclusion)] += 1
		if len(correct_steps + redundant_steps + incorrect_steps) == 0:
			proof_lengths.append(0)
		else:
			proof_lengths.append(max(correct_steps + redundant_steps + incorrect_steps) + 1)
		correct_labels += label

	return (len(results), proof_lengths, correct_labels, all_correct_and_useful_steps, all_redundant_steps, all_unparseable_steps, all_incorrect_steps, contains_correct_proof, does_not_contain_correct_proof, contains_wrong_branch, contains_useful_skip_step, contains_wrong_skip_step, contains_useful_non_atomic_step, contains_wrong_non_atomic_step, contains_invalid_step, contains_wrong_branch_or_useful_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_step, contains_wrong_branch_or_invalid_step, contains_wrong_branch_or_useful_non_atomic_or_invalid_step, contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step, contains_wrong_branch_or_useful_skip_step, contains_wrong_branch_or_wrong_skip_step, contains_useful_skip_or_wrong_skip_step, contains_useful_skip_or_useful_non_atomic_step, contains_useful_skip_or_wrong_non_atomic_step, contains_useful_skip_or_invalid_step, contains_wrong_skip_or_useful_non_atomic_step, contains_wrong_skip_or_wrong_non_atomic_step, contains_wrong_skip_or_invalid_step, contains_useful_non_atomic_or_wrong_non_atomic_step, contains_useful_non_atomic_or_invalid_step, contains_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_or_invalid_step, contains_correct_proof_with_skip_step, contains_correct_proof_with_non_atomic_step, contains_correct_proof_with_skip_step_or_non_atomic_step, wrong_branch_first, wrong_branch_proofs)

if len(argv) > 1:
	(num_examples, proof_lengths, correct_labels, all_correct_and_useful_steps, all_redundant_steps, all_unparseable_steps, all_incorrect_steps, contains_correct_proof, does_not_contain_correct_proof, contains_wrong_branch, contains_useful_skip_step, contains_wrong_skip_step, contains_useful_non_atomic_step, contains_wrong_non_atomic_step, contains_invalid_step, contains_wrong_branch_or_useful_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_step, contains_wrong_branch_or_invalid_step, contains_wrong_branch_or_useful_non_atomic_or_invalid_step, contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step, contains_wrong_branch_or_useful_skip_step, contains_wrong_branch_or_wrong_skip_step, contains_useful_skip_or_wrong_skip_step, contains_useful_skip_or_useful_non_atomic_step, contains_useful_skip_or_wrong_non_atomic_step, contains_useful_skip_or_invalid_step, contains_wrong_skip_or_useful_non_atomic_step, contains_wrong_skip_or_wrong_non_atomic_step, contains_wrong_skip_or_invalid_step, contains_useful_non_atomic_or_wrong_non_atomic_step, contains_useful_non_atomic_or_invalid_step, contains_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_or_invalid_step, contains_correct_proof_with_skip_step, contains_correct_proof_with_non_atomic_step, contains_correct_proof_with_skip_step_or_non_atomic_step, wrong_branch_first, wrong_branch_proofs) = analyze_log(argv[1])

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

	def lighten_color(color, amount=0.5):
		import matplotlib.colors as mc
		import colorsys
		try:
			c = mc.cnames[color]
		except:
			c = color
		c = colorsys.rgb_to_hls(*mc.to_rgb(c))
		return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

	def make_barplot(filename_glob, chart_filename):
		logfiles = glob.glob(filename_glob)

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
			(num_examples, proof_lengths, correct_labels, all_correct_and_useful_steps, all_redundant_steps, all_unparseable_steps, all_incorrect_steps, contains_correct_proof, does_not_contain_correct_proof, contains_wrong_branch, contains_useful_skip_step, contains_wrong_skip_step, contains_useful_non_atomic_step, contains_wrong_non_atomic_step, contains_invalid_step, contains_wrong_branch_or_useful_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_step, contains_wrong_branch_or_invalid_step, contains_wrong_branch_or_useful_non_atomic_or_invalid_step, contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step, contains_wrong_branch_or_useful_skip_step, contains_wrong_branch_or_wrong_skip_step, contains_useful_skip_or_wrong_skip_step, contains_useful_skip_or_useful_non_atomic_step, contains_useful_skip_or_wrong_non_atomic_step, contains_useful_skip_or_invalid_step, contains_wrong_skip_or_useful_non_atomic_step, contains_wrong_skip_or_wrong_non_atomic_step, contains_wrong_skip_or_invalid_step, contains_useful_non_atomic_or_wrong_non_atomic_step, contains_useful_non_atomic_or_invalid_step, contains_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_or_invalid_step, contains_correct_proof_with_skip_step, contains_correct_proof_with_non_atomic_step, contains_correct_proof_with_skip_step_or_non_atomic_step, wrong_branch_first, wrong_branch_proofs) = analyze_log(logfile)
			experiment_names.append(logfile)

			correct_proof_count = np.sum(contains_correct_proof)
			correct_proofs.append((correct_proof_count - contains_wrong_branch[1]) / num_examples)
			correct_proofs_wrong_branch.append(contains_wrong_branch[1] / num_examples)
			incorrect_proofs_wrong_branch.append(contains_wrong_branch[0] / num_examples)
			incorrect_proofs_useful_non_atomic_step.append(contains_useful_non_atomic_step[0] / num_examples)
			incorrect_proofs_wrong_non_atomic_step.append(contains_wrong_non_atomic_step[0] / num_examples)
			incorrect_proofs_useful_skip_step.append(contains_useful_skip_step[0] / num_examples)
			incorrect_proofs_wrong_skip_step.append(contains_wrong_skip_step[0] / num_examples)
			incorrect_proofs_invalid_step.append(contains_invalid_step[0] / num_examples)
			incorrect_proofs_wrong_branch_and_useful_non_atomic_step.append(contains_wrong_branch_or_useful_non_atomic_step[0] / num_examples)
			incorrect_proofs_wrong_branch_and_wrong_non_atomic_step.append(contains_wrong_branch_or_wrong_non_atomic_step[0] / num_examples)
			incorrect_proofs_wrong_branch_and_invalid_step.append(contains_wrong_branch_or_invalid_step[0] / num_examples)
			incorrect_proofs_wrong_branch_and_useful_non_atomic_step_and_invalid_step.append(contains_wrong_branch_or_useful_non_atomic_or_invalid_step[0] / num_examples)
			incorrect_proofs_other.append((num_examples - correct_proof_count - contains_wrong_branch[0] - contains_useful_non_atomic_step[0] - contains_wrong_non_atomic_step[0] - contains_useful_skip_step[0] - contains_wrong_skip_step[0] - contains_invalid_step[0] - contains_wrong_branch_or_useful_non_atomic_step[0] - contains_wrong_branch_or_wrong_non_atomic_step[0] - contains_wrong_branch_or_invalid_step[0] - contains_wrong_branch_or_useful_non_atomic_or_invalid_step[0]) / num_examples)

		correct_proofs = np.array(correct_proofs)
		correct_proofs_wrong_branch = np.array(correct_proofs_wrong_branch)
		incorrect_proofs_wrong_branch = np.array(incorrect_proofs_wrong_branch)
		incorrect_proofs_useful_non_atomic_step = np.array(incorrect_proofs_useful_non_atomic_step)
		incorrect_proofs_wrong_non_atomic_step = np.array(incorrect_proofs_wrong_non_atomic_step)
		incorrect_proofs_useful_skip_step = np.array(incorrect_proofs_useful_skip_step)
		incorrect_proofs_wrong_skip_step = np.array(incorrect_proofs_wrong_skip_step)
		incorrect_proofs_invalid_step = np.array(incorrect_proofs_invalid_step)
		incorrect_proofs_wrong_branch_and_useful_non_atomic_step = np.array(incorrect_proofs_wrong_branch_and_useful_non_atomic_step)
		incorrect_proofs_wrong_branch_and_wrong_non_atomic_step = np.array(incorrect_proofs_wrong_branch_and_wrong_non_atomic_step)
		incorrect_proofs_wrong_branch_and_invalid_step = np.array(incorrect_proofs_wrong_branch_and_invalid_step)
		incorrect_proofs_wrong_branch_and_useful_non_atomic_step_and_invalid_step = np.array(incorrect_proofs_wrong_branch_and_useful_non_atomic_step_and_invalid_step)
		incorrect_proofs_other = np.array(incorrect_proofs_other)

		fig = plt.gcf()
		fig.set_size_inches(8.5, 3.5, forward=True)
		plt.bar(experiment_names, correct_proofs, color=colors[5])
		plt.bar(experiment_names, correct_proofs_wrong_branch, bottom=correct_proofs, color=colors[5], hatch='///')
		plt.bar(experiment_names, incorrect_proofs_wrong_branch, bottom=correct_proofs+correct_proofs_wrong_branch, color=colors[0], hatch='///')
		plt.bar(experiment_names, incorrect_proofs_useful_non_atomic_step, bottom=correct_proofs+correct_proofs_wrong_branch+incorrect_proofs_wrong_branch, color=colors[0], hatch='\\\\\\')
		plt.bar(experiment_names, incorrect_proofs_wrong_non_atomic_step, bottom=correct_proofs+correct_proofs_wrong_branch+incorrect_proofs_wrong_branch+incorrect_proofs_useful_non_atomic_step, color=colors[0], hatch='\\\\\\')
		plt.bar(experiment_names, incorrect_proofs_useful_skip_step, bottom=correct_proofs+correct_proofs_wrong_branch+incorrect_proofs_wrong_branch+incorrect_proofs_useful_non_atomic_step+incorrect_proofs_wrong_non_atomic_step, color=colors[0], hatch='\\\\\\')
		plt.bar(experiment_names, incorrect_proofs_wrong_skip_step, bottom=correct_proofs+correct_proofs_wrong_branch+incorrect_proofs_wrong_branch+incorrect_proofs_useful_non_atomic_step+incorrect_proofs_wrong_non_atomic_step+incorrect_proofs_useful_skip_step, color=colors[0], hatch='\\\\\\')
		plt.bar(experiment_names, incorrect_proofs_invalid_step, bottom=correct_proofs+correct_proofs_wrong_branch+incorrect_proofs_wrong_branch+incorrect_proofs_useful_non_atomic_step+incorrect_proofs_wrong_non_atomic_step+incorrect_proofs_useful_skip_step+incorrect_proofs_wrong_skip_step, color=lighten_color(colors[0],1.3))
		plt.bar(experiment_names, incorrect_proofs_wrong_branch_and_useful_non_atomic_step, bottom=correct_proofs+correct_proofs_wrong_branch+incorrect_proofs_wrong_branch+incorrect_proofs_useful_non_atomic_step+incorrect_proofs_wrong_non_atomic_step+incorrect_proofs_useful_skip_step+incorrect_proofs_wrong_skip_step+incorrect_proofs_invalid_step, color=colors[0], hatch='xxx')
		plt.bar(experiment_names, incorrect_proofs_wrong_branch_and_wrong_non_atomic_step, bottom=correct_proofs+correct_proofs_wrong_branch+incorrect_proofs_wrong_branch+incorrect_proofs_useful_non_atomic_step+incorrect_proofs_wrong_non_atomic_step+incorrect_proofs_useful_skip_step+incorrect_proofs_wrong_skip_step+incorrect_proofs_invalid_step+incorrect_proofs_wrong_branch_and_useful_non_atomic_step, color=colors[0], hatch='xxx')
		plt.bar(experiment_names, incorrect_proofs_wrong_branch_and_invalid_step, bottom=correct_proofs+correct_proofs_wrong_branch+incorrect_proofs_wrong_branch+incorrect_proofs_useful_non_atomic_step+incorrect_proofs_wrong_non_atomic_step+incorrect_proofs_useful_skip_step+incorrect_proofs_wrong_skip_step+incorrect_proofs_invalid_step+incorrect_proofs_wrong_branch_and_useful_non_atomic_step+incorrect_proofs_wrong_branch_and_wrong_non_atomic_step, color=lighten_color(colors[0],1.3), hatch='///')
		plt.bar(experiment_names, incorrect_proofs_wrong_branch_and_useful_non_atomic_step_and_invalid_step, bottom=correct_proofs+correct_proofs_wrong_branch+incorrect_proofs_wrong_branch+incorrect_proofs_useful_non_atomic_step+incorrect_proofs_wrong_non_atomic_step+incorrect_proofs_useful_skip_step+incorrect_proofs_wrong_skip_step+incorrect_proofs_invalid_step+incorrect_proofs_wrong_branch_and_useful_non_atomic_step+incorrect_proofs_wrong_branch_and_wrong_non_atomic_step+incorrect_proofs_wrong_branch_and_invalid_step, color=lighten_color(colors[0],1.3), hatch='xxx')
		plt.bar(experiment_names, incorrect_proofs_other, bottom=correct_proofs+correct_proofs_wrong_branch+incorrect_proofs_wrong_branch+incorrect_proofs_useful_non_atomic_step+incorrect_proofs_wrong_non_atomic_step+incorrect_proofs_useful_skip_step+incorrect_proofs_wrong_skip_step+incorrect_proofs_invalid_step+incorrect_proofs_wrong_branch_and_useful_non_atomic_step+incorrect_proofs_wrong_branch_and_wrong_non_atomic_step+incorrect_proofs_wrong_branch_and_invalid_step+incorrect_proofs_wrong_branch_and_useful_non_atomic_step_and_invalid_step, color=colors[0])
		plt.ylabel('strict proof accuracy')
		plt.ylim([0.0, 1.0])
		fig.savefig(chart_filename, dpi=128, bbox_inches='tight')
		plt.clf()


	logfiles = glob.glob('gpt_*.log')
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
		(num_examples, proof_lengths, correct_labels, all_correct_and_useful_steps, all_redundant_steps, all_unparseable_steps, all_incorrect_steps, contains_correct_proof, does_not_contain_correct_proof, contains_wrong_branch, contains_useful_skip_step, contains_wrong_skip_step, contains_useful_non_atomic_step, contains_wrong_non_atomic_step, contains_invalid_step, contains_wrong_branch_or_useful_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_step, contains_wrong_branch_or_invalid_step, contains_wrong_branch_or_useful_non_atomic_or_invalid_step, contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step, contains_wrong_branch_or_useful_skip_step, contains_wrong_branch_or_wrong_skip_step, contains_useful_skip_or_wrong_skip_step, contains_useful_skip_or_useful_non_atomic_step, contains_useful_skip_or_wrong_non_atomic_step, contains_useful_skip_or_invalid_step, contains_wrong_skip_or_useful_non_atomic_step, contains_wrong_skip_or_wrong_non_atomic_step, contains_wrong_skip_or_invalid_step, contains_useful_non_atomic_or_wrong_non_atomic_step, contains_useful_non_atomic_or_invalid_step, contains_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_or_invalid_step, contains_correct_proof_with_skip_step, contains_correct_proof_with_non_atomic_step, contains_correct_proof_with_skip_step_or_non_atomic_step, wrong_branch_first, wrong_branch_proofs) = analyze_log(logfile)
		print(np.sum(contains_wrong_branch_or_useful_skip_step + contains_wrong_branch_or_wrong_skip_step + contains_useful_skip_or_wrong_skip_step + contains_useful_skip_or_useful_non_atomic_step + contains_useful_skip_or_wrong_non_atomic_step + contains_useful_skip_or_invalid_step + contains_wrong_skip_or_useful_non_atomic_step + contains_wrong_skip_or_wrong_non_atomic_step + contains_wrong_skip_or_invalid_step + contains_useful_non_atomic_or_wrong_non_atomic_step + contains_useful_non_atomic_or_invalid_step + contains_wrong_non_atomic_or_invalid_step + contains_wrong_branch_or_non_atomic_step + contains_wrong_branch_or_wrong_non_atomic_or_invalid_step + contains_wrong_branch_or_non_atomic_or_invalid_step))

		correct_proof_count = np.sum(contains_correct_proof)
		label_accuracy.append(correct_labels / num_examples)
		proof_accuracy.append(correct_proof_count / num_examples)
		proof_accuracy_with_skip_steps.append((correct_proof_count + contains_correct_proof_with_skip_step) / num_examples)
		proof_accuracy_with_non_atomic_steps.append((correct_proof_count + contains_correct_proof_with_non_atomic_step) / num_examples)
		proof_accuracy_with_skip_steps_and_non_atomic_steps.append((correct_proof_count + contains_correct_proof_with_skip_step_or_non_atomic_step) / num_examples)

	plt.style.use('ggplot')
	colors = []
	for c in rcParams["axes.prop_cycle"]:
		colors.append(c['color'])

	fig = plt.gcf()
	fig.set_size_inches(3.5, 3.5, forward=True)
	plt.plot([0, 1], [0, 1], color='black')
	plt.scatter(label_accuracy, proof_accuracy)
	plt.xlabel('label accuracy')
	plt.ylabel('strict proof accuracy')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	fig.savefig('label_vs_proof_accuracy.pdf', dpi=128, bbox_inches='tight')
	plt.clf()

	fig = plt.gcf()
	fig.set_size_inches(3.5, 3.5, forward=True)
	plt.plot([0, 1], [0, 1], color='black')
	plt.scatter(label_accuracy, proof_accuracy_with_skip_steps, c=colors[1])
	plt.xlabel('label accuracy')
	plt.ylabel('proof accuracy if \n "skip steps" are correct')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	fig.savefig('label_vs_proof_accuracy_with_skip_steps.pdf', dpi=128, bbox_inches='tight')
	plt.clf()

	fig = plt.gcf()
	fig.set_size_inches(3.5, 3.5, forward=True)
	plt.plot([0, 1], [0, 1], color='black')
	plt.scatter(label_accuracy, proof_accuracy_with_non_atomic_steps, c=colors[2])
	plt.xlabel('label accuracy')
	plt.ylabel('proof accuracy if \n non-atomic steps are correct')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	fig.savefig('label_vs_proof_accuracy_with_non_atomic_steps.pdf', dpi=128, bbox_inches='tight')
	plt.clf()

	fig = plt.gcf()
	fig.set_size_inches(3.5, 3.5, forward=True)
	plt.plot([0, 1], [0, 1], color='black')
	plt.scatter(label_accuracy, proof_accuracy_with_skip_steps_and_non_atomic_steps, c=colors[5])
	plt.xlabel('label accuracy')
	plt.ylabel('proof accuracy if \n both "skip steps" and \n non-atomic steps are correct')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	fig.savefig('label_vs_proof_accuracy_with_skip_steps_and_non_atomic_steps.pdf', dpi=128, bbox_inches='tight')
	plt.clf()

	make_barplot('gpt_textdavinci002_*.log', 'proof_accuracy_breakdown.png')