import argparse
import plotille
import numpy as np
from main import parse_log


parser = argparse.ArgumentParser()
parser.add_argument("log_file", type=str)
args = parser.parse_args()

with open(args.log_file, "r") as log:
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
does_not_contain_wrong_branch = [0, 0]
correct_labels = 0

for result in results:
	(label, correct_steps, correct_and_useful_steps, redundant_steps, unparseable_steps, wrong_branch_steps, incorrect_steps, found_conclusion) = result
	all_correct_steps.extend(correct_steps)
	all_correct_and_useful_steps.extend(correct_and_useful_steps)
	all_redundant_steps.extend(redundant_steps)
	all_unparseable_steps.extend(unparseable_steps)
	all_incorrect_steps.extend(incorrect_steps)
	if found_conclusion:
		contains_correct_proof[int(label)] += 1
	else:
		does_not_contain_correct_proof[int(label)] += 1
	if len(wrong_branch_steps) != 0:
		contains_wrong_branch[int(found_conclusion)] += 1
	else:
		does_not_contain_wrong_branch[int(found_conclusion)] += 1
	proof_lengths.append(max(correct_steps + redundant_steps + incorrect_steps) + 1)	
	correct_labels += label

max_proof_length = max(proof_lengths)
print("Proof lengths: (max: {})".format(max_proof_length))
print(plotille.histogram(proof_lengths, x_min=0, x_max=max_proof_length, y_min=0, y_max=120, height=20, bins=(max_proof_length + 1), lc=(80,80,255), color_mode='rgb'))

total_steps = np.sum(proof_lengths)
print("Correct steps: {}".format(len(all_correct_steps) / total_steps))
print(plotille.histogram(all_correct_steps, x_min=0, x_max=max_proof_length, y_min=0, y_max=120, height=20, bins=(max_proof_length + 1), lc=(80,80,255), color_mode='rgb'))

print("Correct and useful steps: {}".format(len(all_correct_and_useful_steps) / total_steps))
print(plotille.histogram(all_correct_and_useful_steps, x_min=0, x_max=max_proof_length, y_min=0, y_max=120, height=20, bins=(max_proof_length + 1), lc=(80,80,255), color_mode='rgb'))

print("Redundant steps: {}".format(len(all_redundant_steps) / total_steps))
print(plotille.histogram(all_redundant_steps, x_min=0, x_max=max_proof_length, y_min=0, y_max=120, height=20, bins=(max_proof_length + 1), lc=(80,80,255), color_mode='rgb'))

print("Unparseable steps: {}".format(len(all_unparseable_steps) / total_steps))
print(plotille.histogram(all_unparseable_steps, x_min=0, x_max=max_proof_length, y_min=0, y_max=120, height=20, bins=(max_proof_length + 1), lc=(80,80,255), color_mode='rgb'))

print("Incorrect steps: {}".format(len(all_incorrect_steps) / total_steps))
print(plotille.histogram(all_incorrect_steps, x_min=0, x_max=max_proof_length, y_min=0, y_max=120, height=20, bins=(max_proof_length + 1), lc=(80,80,255), color_mode='rgb'))

correct_proofs = np.sum(contains_correct_proof)
print("Proportion of proofs that contain the correct proof: {}".format(correct_proofs / len(results)))

print("Proportion of proofs with the correct label that contain the correct proof:   {}".format(contains_correct_proof[1] / correct_labels))
print("Proportion of proofs with the incorrect label that contain the correct proof: {}".format(contains_correct_proof[0] / (len(results) - correct_labels)))
print("Proportion of proofs with the correct label that do NOT contain the correct proof:   {}".format(does_not_contain_correct_proof[1] / correct_labels))
print("Proportion of proofs with the incorrect label that do NOT contain the correct proof: {}".format(does_not_contain_correct_proof[0] / (len(results) - correct_labels)))

print("Proportion of correct proofs that contain a \"wrong branch\":   {}".format(contains_wrong_branch[1] / correct_proofs))
print("Proportion of incorrect proofs that contain a \"wrong branch\": {}".format(contains_wrong_branch[0] / (len(results) - correct_proofs)))
print("Proportion of correct proofs that do NOT contain a \"wrong branch\":   {}".format(does_not_contain_wrong_branch[1] / correct_proofs))
print("Proportion of incorrect proofs that do NOT contain a \"wrong branch\": {}".format(does_not_contain_wrong_branch[0] / (len(results) - correct_proofs)))

print("Proportion of correct labels: {}".format(correct_labels / len(results)))
