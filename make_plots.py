import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.transforms import Bbox
from analyze_results import analyze_log, wilson_conf_interval, lighten_color

palm_results = {
	'1hop_AndElim_2testhops_random_noadj': 0.45,
	'1hop_AndElim_3testhops_random_noadj': 0.42,
	'1hop_AndIntro_2testhops_random_noadj': 0.0,
	'1hop_AndIntro_3testhops_random_noadj': 0.0,
	'1hop_OOD_OrElim_3proofwidth_random_noadj': 0.03,
	'1hop_OOD_ProofByContra_random_noadj': 0.04,
	'1hop_OrElim_3proofwidth_random_noadj': 0.48,
	'1hop_OrElim_3testwidth_random_noadj': 0.33,
	'1hop_OrElim_4proofwidth_random_noadj': 0.5,
	'1hop_OrElim_4testwidth_random_noadj': 0.46,
	'1hop_ProofByContra_random_noadj': 0.89,
	'1hop_ProofsOnly_2testhops_random_noadj': 0.64,
	'1hop_ProofsOnly_3testhops_random_noadj': 0.03,
	'1hop_ProofsOnly_4testhops_random_noadj': 0.05,
	'1hop_ProofsOnly_5testhops_random_noadj': 0.02,
	'2hop_AndElim_3proofwidth_random_noadj': 0.19,
	'2hop_AndElim_3testhops_random_noadj': 0.16,
	'2hop_AndElim_3testwidth_random_noadj': 0.17,
	'2hop_AndElim_4proofwidth_random_noadj': 0.03,
	'2hop_AndElim_4testwidth_random_noadj': 0.04,
	'2hop_AndElim_random_noadj': 0.34,
	'2hop_AndIntro_3proofwidth_random_noadj': 0.04,
	'2hop_AndIntro_3testhops_random_noadj': 0.07,
	'2hop_AndIntro_3testwidth_random_noadj': 0.03,
	'2hop_AndIntro_4proofwidth_random_noadj': 0.01,
	'2hop_AndIntro_4testwidth_random_noadj': 0.04,
	'2hop_AndIntro_5proofwidth_random_noadj': 0.02,
	'2hop_AndIntro_5testwidth_random_noadj': 0.04,
	'2hop_AndIntro_random_noadj': 0.08,
	'2hop_Composed_2ruletypes_random_noadj': 0.44,
	'2hop_Composed_4ruletypes_random_noadj': 0.33,
	'2hop_Composed_random_noadj': 0.49,
	'2hop_OOD_AndElim_3proofwidth_random_noadj': 0.35,
	'2hop_OOD_AndIntro_3proofwidth_random_noadj': 0.13,
	'2hop_OOD_Composed_2ruletypes_random_noadj': 0.44,
	'2hop_OOD_Composed_4ruletypes_random_noadj': 0.16,
	'2hop_OOD_Composed_random_noadj': 0.45,
	'2hop_OOD_ModusPonens_random_noadj': 0.26,
	'2hop_OOD_OrIntro_3proofwidth_random_noadj': 0.23,
	'2hop_OrIntro_3proofwidth_random_noadj': 0.26,
	'2hop_ProofsOnly_random_noadj': 0.47,
	'3hop_AndElim_random_noadj': 0.11,
	'3hop_AndIntro_random_noadj': 0.04,
	'3hop_Composed_random_noadj': 0.23,
	'3hop_ProofsOnly_random_noadj': 0.14,
	'4hop_Composed_random_noadj': 0.19,
	'4hop_OOD_Composed_random_noadj': 0.37,
	'4hop_ProofsOnly_random_noadj': 0.01,
	'5hop_ProofsOnly_random_noadj': 0.01,
	'1hop_AndElim_random_noadj': 1.0,
	'1hop_AndIntro_random_noadj': 0.98,
	'1hop_OrElim_3proofwidth_testrandom_nodistractor_testdistractor_noadj': 0.35,
	'1hop_ProofByContra_testrandom_nodistractor_testdistractor_noadj': 0.3,
	'1hop_ProofsOnly_random_noadj': 0.91,
	'2hop_AndElim_testrandom_nodistractor_testdistractor_noadj': 0.34,
	'2hop_AndIntro_testrandom_nodistractor_testdistractor_noadj': 0.34,
	'2hop_OrIntro_random_noadj': 0.27,
	'2hop_OrIntro_testrandom_nodistractor_testdistractor_noadj': 0.42,
	'2hop_ProofsOnly_testrandom_nodistractor_testdistractor_noadj': 0.36
}

missing_files = []

def get_proof_accuracy(logfile):
	if logfile.startswith('palm/'):
		filename = logfile[len('palm/'):-len('.log')]
		if filename not in palm_results:
			missing_files.append(logfile)
			return -100.0, 1.0e-16
		return (palm_results[filename], 100)
	try:
		(num_examples, proof_lengths, correct_labels, expected_label_true, correct_step_count, non_atomic_step_count, skip_step_count, invalid_step_count, all_correct_and_useful_steps, all_redundant_steps, all_unparseable_steps, all_incorrect_steps, contains_correct_proof, does_not_contain_correct_proof, contains_wrong_branch, contains_useful_skip_step, contains_wrong_skip_step, contains_useful_non_atomic_step, contains_wrong_non_atomic_step, contains_invalid_step, contains_wrong_branch_or_useful_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_step, contains_wrong_branch_or_invalid_step, contains_wrong_branch_or_useful_non_atomic_or_invalid_step, contains_wrong_branch_or_skip_step_or_non_atomic_step_or_invalid_step, contains_wrong_branch_or_useful_skip_step, contains_wrong_branch_or_wrong_skip_step, contains_useful_skip_or_wrong_skip_step, contains_useful_skip_or_useful_non_atomic_step, contains_useful_skip_or_wrong_non_atomic_step, contains_useful_skip_or_invalid_step, contains_wrong_skip_or_useful_non_atomic_step, contains_wrong_skip_or_wrong_non_atomic_step, contains_wrong_skip_or_invalid_step, contains_useful_non_atomic_or_wrong_non_atomic_step, contains_useful_non_atomic_or_invalid_step, contains_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_step, contains_wrong_branch_or_wrong_non_atomic_or_invalid_step, contains_wrong_branch_or_non_atomic_or_invalid_step, contains_correct_proof_with_skip_step, contains_correct_proof_with_non_atomic_step, contains_correct_proof_with_skip_step_or_non_atomic_step, wrong_branch_first, useful_skip_step_first, wrong_skip_step_first, useful_non_atomic_step_first, wrong_non_atomic_step_first, invalid_step_first, contains_any_wrong_branch, contains_any_useful_skip_step, contains_any_wrong_skip_step, contains_any_useful_non_atomic_step, contains_any_wrong_non_atomic_step, contains_any_invalid_step, wrong_branch_lengths, labels, correct_proofs, correct_proofs_with_skip_steps, correct_proofs_with_non_atomic_steps, correct_proofs_with_skip_or_non_atomic_steps, fully_correct_proofs, total_perplexity) = analyze_log(logfile)
	except FileNotFoundError:
		missing_files.append(logfile)
		return -100.0, 1.0e-16
	correct_proof_count = np.sum(contains_correct_proof)
	return ((correct_proof_count + contains_correct_proof_with_skip_step_or_non_atomic_step) / num_examples, num_examples)

def make_rule_plot(output_filename, filenames, group_labels, title=None, width=8.0, legend_pos=(1.0, 0.5), legend_cols=1):
	bar_group_size = 4
	bar_spacing = 0.2
	bar_width = (1.0 - bar_spacing) / bar_group_size

	x1 = np.arange(len(filenames))
	x2 = [x + bar_width for x in x1]
	x3 = [x + bar_width for x in x2]
	x4 = [x + bar_width for x in x3]

	accuracies = np.empty((4,len(filenames)))
	example_counts = np.empty((4,len(filenames)))
	for i in range(4):
		for j in range(len(filenames)):
			accuracies[i,j], example_counts[i,j] = get_proof_accuracy(filenames[j][i])

	(lower_bound, upper_bound) = wilson_conf_interval(accuracies, example_counts)
	lower_bound[np.isnan(lower_bound)] = 0.0; upper_bound[np.isnan(upper_bound)] = 1.0
	err = np.maximum(0, np.stack((accuracies - lower_bound, upper_bound - accuracies), axis=2))

	fig = plt.gcf()
	fig.set_size_inches(width, 1.8, forward=True)
	if title != None:
		plt.title(title, pad=-0.2, fontsize=11)
	plt.bar(x1, accuracies[0,:], width=bar_width, yerr=err[0,:,:].T, capsize=2.0, color=colors[0])
	plt.bar(x2, accuracies[1,:], width=bar_width, yerr=err[1,:,:].T, capsize=2.0, color=colors[1])
	plt.bar(x3, accuracies[2,:], width=bar_width, yerr=err[2,:,:].T, capsize=2.0, color=colors[5])
	plt.bar(x4, accuracies[3,:], width=bar_width, yerr=err[3,:,:].T, capsize=2.0, color=colors[2])

	labels = ['GPT-3.5', 'PaLM', 'LLaMA', 'FLAN-T5']
	if legend_pos == None:
		pass
	elif type(legend_pos) == str:
		plt.legend(labels, loc=legend_pos, fontsize=7.5, facecolor='white', edgecolor='white', ncols=legend_cols, columnspacing=1.0, handletextpad=0.5)
	else:
		plt.legend(labels, loc='center left', bbox_to_anchor=legend_pos, fontsize=8, facecolor='white', edgecolor='white', ncols=legend_cols, columnspacing=1.0, handletextpad=0.5)
	delta_width, delta_height = 0.0, 0.0
	if type(legend_pos) == tuple and legend_pos[0] >= 1.0:
		delta_width = 0.35
	elif type(legend_pos) == tuple and legend_pos[1] >= 1.0:
		delta_height = 0.25

	ax = plt.gca()
	plt.ylabel('proof accuracy')
	plt.xlim([-bar_spacing, len(filenames) - 0.12])
	plt.ylim([0.0, 1.0])
	delta = (1.0 - bar_spacing) / (2*bar_group_size)
	plt.xticks([x + ((1.0 - bar_spacing) / 2) - delta for x in x1], group_labels, fontsize=10)
	plt.tick_params(axis='x', which='both', length=0)
	xlabel_line_count = np.max([label.count('\n') for label in group_labels])
	fig.savefig(output_filename, dpi=128, bbox_inches=Bbox([[(width-5.5)/8, (xlabel_line_count + 1) * -0.1], [width + delta_width - 0.3, 1.8 + delta_height]]))
	plt.clf()
	return accuracies, lower_bound, upper_bound

def make_diff_plot(output_filename, filenames, group_labels, base_accuracies, base_lower_bound, base_upper_bound, title=None, width=8.0, legend_pos=(1.0, 0.5), legend_cols=1, ymin=-1.0, ymax=0.5):
	bar_group_size = 4
	bar_spacing = 0.2
	bar_width = (1.0 - bar_spacing) / bar_group_size

	x1 = np.arange(len(filenames))
	x2 = [x + bar_width for x in x1]
	x3 = [x + bar_width for x in x2]
	x4 = [x + bar_width for x in x3]

	accuracies = np.empty((4,len(filenames)))
	example_counts = np.empty((4,len(filenames)))
	for i in range(4):
		for j in range(len(filenames)):
			accuracies[i,j], example_counts[i,j] = get_proof_accuracy(filenames[j][i])

	(lower_bound, upper_bound) = wilson_conf_interval(accuracies, example_counts)
	lower_bound[np.isnan(lower_bound)] = 0.0; upper_bound[np.isnan(upper_bound)] = 1.0
	delta_accuracies = accuracies - base_accuracies
	delta_lower_bound = delta_accuracies - np.sqrt((accuracies - lower_bound)**2 + (base_upper_bound - base_accuracies)**2)
	delta_upper_bound = delta_accuracies + np.sqrt((upper_bound - accuracies)**2 + (base_accuracies - base_lower_bound)**2)
	err = np.maximum(0, np.stack((delta_accuracies - delta_lower_bound, delta_upper_bound - delta_accuracies), axis=2))

	fig = plt.gcf()
	fig.set_size_inches(width, 1.8, forward=True)
	if title != None:
		plt.title(title, pad=-0.2, fontsize=11)
	plt.plot([-bar_spacing, len(filenames) - 0.12], [0.0, 0.0], c='#555', linewidth=1, label='_nolegend_')
	plt.bar(x1, delta_accuracies[0,:], width=bar_width, yerr=err[0,:,:].T, capsize=2.0, color=colors[0])
	plt.bar(x2, delta_accuracies[1,:], width=bar_width, yerr=err[1,:,:].T, capsize=2.0, color=colors[1])
	plt.bar(x3, delta_accuracies[2,:], width=bar_width, yerr=err[2,:,:].T, capsize=2.0, color=colors[5])
	plt.bar(x4, delta_accuracies[3,:], width=bar_width, yerr=err[3,:,:].T, capsize=2.0, color=colors[2])

	labels = ['GPT-3.5', 'PaLM', 'LLaMA', 'FLAN-T5']
	if legend_pos == None:
		pass
	elif type(legend_pos) == str:
		plt.legend(labels, loc=legend_pos, fontsize=7.5, facecolor='white', edgecolor='white', ncols=legend_cols, columnspacing=1.0, handletextpad=0.5)
	else:
		plt.legend(labels, loc='center left', bbox_to_anchor=legend_pos, fontsize=8, facecolor='white', edgecolor='white', ncols=legend_cols, columnspacing=1.0, handletextpad=0.5)
	delta_width, delta_height = 0.0, 0.0
	if type(legend_pos) == tuple and legend_pos[0] >= 1.0:
		delta_width = 0.35
	elif type(legend_pos) == tuple and legend_pos[1] >= 1.0:
		delta_height = 0.25

	ax = plt.gca()
	plt.ylabel('$\Delta$ proof accuracy')
	plt.xlim([-bar_spacing, len(filenames) - 0.12])
	plt.ylim([ymin, ymax])
	delta = (1.0 - bar_spacing) / (2*bar_group_size)
	plt.xticks([x + ((1.0 - bar_spacing) / 2) - delta for x in x1], group_labels, fontsize=10)
	plt.tick_params(axis='x', which='both', length=0)
	xlabel_line_count = np.max([label.count('\n') for label in group_labels])
	fig.savefig(output_filename, dpi=128, bbox_inches=Bbox([[(width-5.5)/8, (xlabel_line_count + 1) * -0.1], [width + delta_width - 0.3, 1.8 + delta_height]]))
	plt.clf()

plt.rcParams.update({
	"text.usetex": True,
	"font.family": "serif",
	"font.serif": ["Palatino"],
})

plt.style.use('ggplot')
colors = []
for c in rcParams["axes.prop_cycle"]:
	colors.append(c['color'])
# colors are red, blue, purple, grey, yellow, green, pink

rule_accuracies, rule_lower_bound, rule_upper_bound = make_rule_plot('rule_accuracies.pdf',
		[
		['gpt_textdavinci003_2hop_ProofsOnly_random_noadj.log',
		 'palm/2hop_ProofsOnly_random_noadj.log',
		 'llama/2hop_ProofsOnly_random_noadj.json',
		 'flan-t5/latest/2hop_ProofsOnly_random_noadj.json'],
		['gpt_textdavinci003_2hop_AndIntro_3proofwidth_random_noadj.log',
		 'palm/2hop_AndIntro_3proofwidth_random_noadj.log',
		 'llama/2hop_AndIntro_3proofwidth_random_noadj.json',
		 'flan-t5/latest/2hop_AndIntro_3proofwidth_random_noadj.json'],
		['gpt_textdavinci003_2hop_AndElim_3proofwidth_random_noadj.log',
		 'palm/2hop_AndElim_3proofwidth_random_noadj.log',
		 'llama/2hop_AndElim_3proofwidth_random_noadj.json',
		 'flan-t5/latest/2hop_AndElim_3proofwidth_random_noadj.json'],
		['gpt_textdavinci003_2hop_OrIntro_3proofwidth_random_noadj.log',
		 'palm/2hop_OrIntro_3proofwidth_random_noadj.log',
		 'llama/2hop_OrIntro_3proofwidth_random_noadj.json',
		 'flan-t5/latest/2hop_OrIntro_3proofwidth_random_noadj.json'],
		['gpt_textdavinci003_1hop_OrElim_3proofwidth_random_noadj.log',
		 'palm/1hop_OrElim_3proofwidth_random_noadj.log',
		 'llama/1hop_OrElim_3proofwidth_random_noadj.json',
		 'flan-t5/latest/1hop_OrElim_3proofwidth_random_noadj.json'],
		['gpt_textdavinci003_1hop_ProofByContra_random_noadj.log',
		 'palm/1hop_ProofByContra_random_noadj.log',
		 'llama/1hop_ProofByContra_random_noadj.json',
		 'flan-t5/latest/1hop_ProofByContra_random_noadj.json']
		],
		['implication\nelimination', 'conjunction\nintroduction', 'conjunction\nelimination', 'disjunction\nintroduction', 'disjunction\nelimination', 'proof by\ncontradiction'],
		title="ID: Rule generalization")
make_diff_plot('ood_accuracies.pdf', [
		['gpt_textdavinci003_2hop_OOD_ModusPonens_random_noadj.log',
		 'palm/2hop_OOD_ModusPonens_random_noadj.log',
		 'llama/2hop_OOD_ModusPonens_random_noadj.json',
		 'flan-t5/latest/2hop_OOD_ModusPonens_random_noadj.json'],
		['gpt_textdavinci003_2hop_OOD_AndIntro_3proofwidth_random_noadj.log',
		 'palm/2hop_OOD_AndIntro_3proofwidth_random_noadj.log',
		 'llama/2hop_OOD_AndIntro_3proofwidth_random_noadj.json',
		 'flan-t5/latest/2hop_OOD_AndIntro_3proofwidth_random_noadj.json'],
		['gpt_textdavinci003_2hop_OOD_AndElim_3proofwidth_random_noadj.log',
		 'palm/2hop_OOD_AndElim_3proofwidth_random_noadj.log',
		 'llama/2hop_OOD_AndElim_3proofwidth_random_noadj.json',
		 'flan-t5/latest/2hop_OOD_AndElim_3proofwidth_random_noadj.json'],
		['gpt_textdavinci003_2hop_OOD_OrIntro_3proofwidth_random_noadj.log',
		 'palm/2hop_OOD_OrIntro_3proofwidth_random_noadj.log',
		 'llama/2hop_OOD_OrIntro_3proofwidth_random_noadj.json',
		 'flan-t5/latest/2hop_OOD_OrIntro_3proofwidth_random_noadj.json'],
		['gpt_textdavinci003_1hop_OOD_OrElim_3proofwidth_random_noadj.log',
		 'palm/1hop_OOD_OrElim_3proofwidth_random_noadj.log',
		 'llama/1hop_OOD_OrElim_3proofwidth_random_noadj.json',
		 'flan-t5/latest/1hop_OOD_OrElim_3proofwidth_random_noadj.json'],
		['gpt_textdavinci003_1hop_OOD_ProofByContra_random_noadj.log',
		 'palm/1hop_OOD_ProofByContra_random_noadj.log',
		 'llama/1hop_OOD_ProofByContra_random_noadj.json',
		 'flan-t5/latest/1hop_OOD_ProofByContra_random_noadj.json']
		],
		['implication\nelimination', 'conjunction\nintroduction', 'conjunction\nelimination', 'disjunction\nintroduction', 'disjunction\nelimination', 'proof by\ncontradiction'],
		rule_accuracies, rule_lower_bound, rule_upper_bound,
		title="OOD: Rule generalization")

compositional_accuracies, compositional_lower_bound, compositional_upper_bound = make_rule_plot('compositional_id_accuracies.pdf',
		[
		['gpt_textdavinci003_2hop_Composed_2ruletypes_random_noadj.log',
		 'palm/2hop_Composed_2ruletypes_random_noadj.log',
		 'llama/2hop_Composed_2ruletypes_random_noadj.json',
		 'flan-t5/latest/2hop_Composed_2ruletypes_random_noadj.json'],
		['gpt_textdavinci003_2hop_Composed_random_noadj.log',
		 'palm/2hop_Composed_random_noadj.log',
		 'llama/2hop_Composed_random_noadj.json',
		 'flan-t5/latest/2hop_Composed_random_noadj.json'],
		['gpt_textdavinci003_2hop_Composed_4ruletypes_random_noadj.log',
		 'palm/2hop_Composed_4ruletypes_random_noadj.log',
		 'llama/2hop_Composed_4ruletypes_random_noadj.json',
		 'flan-t5/latest/2hop_Composed_4ruletypes_random_noadj.json'],
		['gpt_textdavinci003_4hop_Composed_random_noadj.log',
		 'palm/4hop_Composed_random_noadj.log',
		 'llama/4hop_Composed_random_noadj.json',
		 'flan-t5/latest/4hop_Composed_random_noadj.json']
		],
		['min depth: 2\nrule types: 2', 'min depth: 2\nrule types: 3', 'min depth: 2\nrule types: 4', 'min depth: 4\nrule types: 3'],
		title="ID: Compositional examples", width=5.0, legend_pos='upper right', legend_cols=4)
make_diff_plot('compositional_ood_accuracies.pdf', [
		['gpt_textdavinci003_2hop_OOD_Composed_2ruletypes_random_noadj.log',
		 'palm/2hop_OOD_Composed_2ruletypes_random_noadj.log',
		 'llama/2hop_OOD_Composed_2ruletypes_random_noadj.json',
		 'flan-t5/latest/2hop_OOD_Composed_2ruletypes_random_noadj.json'],
		['gpt_textdavinci003_2hop_OOD_Composed_random_noadj.log',
		 'palm/2hop_OOD_Composed_random_noadj.log',
		 'llama/2hop_OOD_Composed_random_noadj.json',
		 'flan-t5/latest/2hop_OOD_Composed_random_noadj.json'],
		['gpt_textdavinci003_2hop_OOD_Composed_4ruletypes_random_noadj.log',
		 'palm/2hop_OOD_Composed_4ruletypes_random_noadj.log',
		 'llama/2hop_OOD_Composed_4ruletypes_random_noadj.json',
		 'flan-t5/latest/2hop_OOD_Composed_4ruletypes_random_noadj.json'],
		['gpt_textdavinci003_4hop_OOD_Composed_random_noadj.log',
		 'palm/4hop_OOD_Composed_random_noadj.log',
		 'llama/4hop_OOD_Composed_random_noadj.json',
		 'flan-t5/latest/4hop_OOD_Composed_random_noadj.json']
		],
		['min depth: 2\nrule types: 2', 'min depth: 2\nrule types: 3', 'min depth: 2\nrule types: 4', 'min depth: 4\nrule types: 3'],
		compositional_accuracies, compositional_lower_bound, compositional_upper_bound,
		title="OOD: Compositional examples", width=5.0, legend_pos=None, ymin=-0.5)

distractor_accuracies, distractor_lower_bound, distractor_upper_bound = make_rule_plot('distractor_accuracies.pdf',
		[
		['gpt_textdavinci003_2hop_ProofsOnly_random_noadj.log',
		 'palm/2hop_ProofsOnly_random_noadj.log',
		 'llama/2hop_ProofsOnly_random_noadj.json',
		 'flan-t5/latest/2hop_ProofsOnly_random_noadj.json'],
		['gpt_textdavinci003_2hop_AndIntro_random_noadj.log',
		 'palm/2hop_AndIntro_random_noadj.log',
		 'llama/2hop_AndIntro_random_noadj.json',
		 'flan-t5/latest/2hop_AndIntro_random_noadj.json'],
		['gpt_textdavinci003_2hop_AndElim_random_noadj.log',
		 'palm/2hop_AndElim_random_noadj.log',
		 'llama/2hop_AndElim_random_noadj.json',
		 'flan-t5/latest/2hop_AndElim_random_noadj.json'],
		['gpt_textdavinci003_2hop_OrIntro_random_noadj.log',
		 'palm/2hop_OrIntro_random_noadj.log',
		 'llama/2hop_OrIntro_random_noadj.json',
		 'flan-t5/latest/2hop_OrIntro_random_noadj.json'],
		['gpt_textdavinci003_1hop_OrElim_3proofwidth_random_noadj.log',
		 'palm/1hop_OrElim_3proofwidth_random_noadj.log',
		 'llama/1hop_OrElim_3proofwidth_random_noadj.json',
		 'flan-t5/latest/1hop_OrElim_3proofwidth_random_noadj.json'],
		['gpt_textdavinci003_1hop_ProofByContra_random_noadj.log',
		 'palm/1hop_ProofByContra_random_noadj.log',
		 'llama/1hop_ProofByContra_random_noadj.json',
		 'flan-t5/latest/1hop_ProofByContra_random_noadj.json']
		],
		['implication\nelimination', 'conjunction\nintroduction', 'conjunction\nelimination', 'disjunction\nintroduction', 'disjunction\nelimination', 'proof by\ncontradiction'],
		title="ID: In-context examples with distractors")
make_diff_plot('nodistractor_accuracies.pdf', [
		['gpt_textdavinci003_2hop_ProofsOnly_testrandom_nodistractor_testdistractor_noadj.log',
		 'palm/2hop_ProofsOnly_testrandom_nodistractor_testdistractor_noadj.log',
		 'llama/2hop_ProofsOnly_testrandom_nodistractor_testdistractor_noadj.json',
		 'flan-t5/latest/2hop_ProofsOnly_testrandom_nodistractor_testdistractor_noadj.json'],
		['gpt_textdavinci003_2hop_AndIntro_testrandom_nodistractor_testdistractor_noadj.log',
		 'palm/2hop_AndIntro_testrandom_nodistractor_testdistractor_noadj.log',
		 'llama/2hop_AndIntro_testrandom_nodistractor_testdistractor_noadj.json',
		 'flan-t5/latest/2hop_AndIntro_testrandom_nodistractor_testdistractor_noadj.json'],
		['gpt_textdavinci003_2hop_AndElim_testrandom_nodistractor_testdistractor_noadj.log',
		 'palm/2hop_AndElim_testrandom_nodistractor_testdistractor_noadj.log',
		 'llama/2hop_AndElim_testrandom_nodistractor_testdistractor_noadj.json',
		 'flan-t5/latest/2hop_AndElim_testrandom_nodistractor_testdistractor_noadj.json'],
		['gpt_textdavinci003_2hop_OrIntro_testrandom_nodistractor_testdistractor_noadj.log',
		 'palm/2hop_OrIntro_testrandom_nodistractor_testdistractor_noadj.log',
		 'llama/2hop_OrIntro_testrandom_nodistractor_testdistractor_noadj.json',
		 'flan-t5/latest/2hop_OrIntro_testrandom_nodistractor_testdistractor_noadj.json'],
		['gpt_textdavinci003_1hop_OrElim_3proofwidth_testrandom_nodistractor_testdistractor_noadj.log',
		 'palm/1hop_OrElim_3proofwidth_testrandom_nodistractor_testdistractor_noadj.log',
		 'llama/1hop_OrElim_3proofwidth_testrandom_nodistractor_testdistractor_noadj.json',
		 'flan-t5/latest/1hop_OrElim_3proofwidth_testrandom_nodistractor_testdistractor_noadj.json'],
		['gpt_textdavinci003_1hop_ProofByContra_testrandom_nodistractor_testdistractor_noadj.log',
		 'palm/1hop_ProofByContra_testrandom_nodistractor_testdistractor_noadj.log',
		 'llama/1hop_ProofByContra_testrandom_nodistractor_testdistractor_noadj.json',
		 'flan-t5/latest/1hop_ProofByContra_testrandom_nodistractor_testdistractor_noadj.json']
		],
		['implication\nelimination', 'conjunction\nintroduction', 'conjunction\nelimination', 'disjunction\nintroduction', 'disjunction\nelimination', 'proof by\ncontradiction'],
		distractor_accuracies, distractor_lower_bound, distractor_upper_bound,
		title="OOD: In-context examples without distractors")

depth1_accuracies, depth1_lower_bound, depth1_upper_bound = make_rule_plot('1hop_accuracies.pdf',
		[
		['gpt_textdavinci003_1hop_ProofsOnly_random_noadj.log',
		 'palm/1hop_ProofsOnly_random_noadj.log',
		 'llama/1hop_ProofsOnly_random_noadj.json',
		 'flan-t5/latest/1hop_ProofsOnly_random_noadj.json'],
		['gpt_textdavinci003_1hop_AndIntro_random_noadj.log',
		 'palm/1hop_AndIntro_random_noadj.log',
		 'llama/1hop_AndIntro_random_noadj.json',
		 'flan-t5/latest/1hop_AndIntro_random_noadj.json'],
		['gpt_textdavinci003_1hop_AndElim_random_noadj.log',
		 'palm/1hop_AndElim_random_noadj.log',
		 'llama/1hop_AndElim_random_noadj.json',
		 'flan-t5/latest/1hop_AndElim_random_noadj.json']
		],
		['implication\nelimination', 'conjunction\nintroduction', 'conjunction\nelimination'],
		title="ID: Train depth = 1, test depth = 1", width=4.0, legend_pos=(-0.05, 1.25), legend_cols=4)
make_diff_plot('1hop_accuracies_2testhops.pdf', [
		['gpt_textdavinci003_1hop_ProofsOnly_2testhops_random_noadj.log',
		 'palm/1hop_ProofsOnly_2testhops_random_noadj.log',
		 'llama/1hop_ProofsOnly_2testhops_random_noadj.json',
		 'flan-t5/latest/1hop_ProofsOnly_2testhops_random_noadj.json'],
		['gpt_textdavinci003_1hop_AndIntro_2testhops_random_noadj.log',
		 'palm/1hop_AndIntro_2testhops_random_noadj.log',
		 'llama/1hop_AndIntro_2testhops_random_noadj.json',
		 'flan-t5/latest/1hop_AndIntro_2testhops_random_noadj.json'],
		['gpt_textdavinci003_1hop_AndElim_2testhops_random_noadj.log',
		 'palm/1hop_AndElim_2testhops_random_noadj.log',
		 'llama/1hop_AndElim_2testhops_random_noadj.json',
		 'flan-t5/latest/1hop_AndElim_2testhops_random_noadj.json']
		],
		['implication\nelimination', 'conjunction\nintroduction', 'conjunction\nelimination'],
		depth1_accuracies, depth1_lower_bound, depth1_upper_bound,
		title="OOD: Train depth = 1, test depth = 2", width=4.0, legend_pos=None)
make_diff_plot('1hop_accuracies_3testhops.pdf', [
		['gpt_textdavinci003_1hop_ProofsOnly_3testhops_random_noadj.log',
		 'palm/1hop_ProofsOnly_3testhops_random_noadj.log',
		 'llama/1hop_ProofsOnly_3testhops_random_noadj.json',
		 'flan-t5/latest/1hop_ProofsOnly_3testhops_random_noadj.json'],
		['gpt_textdavinci003_1hop_AndIntro_3testhops_random_noadj.log',
		 'palm/1hop_AndIntro_3testhops_random_noadj.log',
		 'llama/1hop_AndIntro_3testhops_random_noadj.json',
		 'flan-t5/latest/1hop_AndIntro_3testhops_random_noadj.json'],
		['gpt_textdavinci003_1hop_AndElim_3testhops_random_noadj.log',
		 'palm/1hop_AndElim_3testhops_random_noadj.log',
		 'llama/1hop_AndElim_3testhops_random_noadj.json',
		 'flan-t5/latest/1hop_AndElim_3testhops_random_noadj.json']
		],
		['implication\nelimination', 'conjunction\nintroduction', 'conjunction\nelimination'],
		depth1_accuracies, depth1_lower_bound, depth1_upper_bound,
		title="OOD: Train depth = 1, test depth = 3", width=4.0, legend_pos=None)

depth2_accuracies, depth2_lower_bound, depth2_upper_bound = make_rule_plot('2hop_accuracies.pdf',
		[
		['gpt_textdavinci003_2hop_AndIntro_random_noadj.log',
		 'palm/2hop_AndIntro_random_noadj.log',
		 'llama/2hop_AndIntro_random_noadj.json',
		 'flan-t5/latest/2hop_AndIntro_random_noadj.json'],
		['gpt_textdavinci003_2hop_AndElim_random_noadj.log',
		 'palm/2hop_AndElim_random_noadj.log',
		 'llama/2hop_AndElim_random_noadj.json',
		 'flan-t5/latest/2hop_AndElim_random_noadj.json']
		],
		['conjunction\nintroduction', 'conjunction\nelimination'],
		title="ID: Train depth = 2, test depth = 2", width=3.2, legend_pos=None)
make_diff_plot('2hop_accuracies_3testhops.pdf', [
		['gpt_textdavinci003_2hop_AndIntro_3testhops_random_noadj.log',
		 'palm/2hop_AndIntro_3testhops_random_noadj.log',
		 'llama/2hop_AndIntro_3testhops_random_noadj.json',
		 'flan-t5/latest/2hop_AndIntro_3testhops_random_noadj.json'],
		['gpt_textdavinci003_2hop_AndElim_3testhops_random_noadj.log',
		 'palm/2hop_AndElim_3testhops_random_noadj.log',
		 'llama/2hop_AndElim_3testhops_random_noadj.json',
		 'flan-t5/latest/2hop_AndElim_3testhops_random_noadj.json']
		],
		['conjunction\nintroduction', 'conjunction\nelimination'],
		depth2_accuracies, depth2_lower_bound, depth2_upper_bound,
		title="OOD: Train depth = 2, test depth = 3", width=3.2, legend_pos=None)

width_accuracies, width_lower_bound, width_upper_bound = make_rule_plot('width_accuracies.pdf',
		[
		['gpt_textdavinci003_2hop_AndIntro_random_noadj.log',
		 'palm/2hop_AndIntro_random_noadj.log',
		 'llama/2hop_AndIntro_random_noadj.json',
		 'flan-t5/latest/2hop_AndIntro_random_noadj.json'],
		['gpt_textdavinci003_2hop_AndElim_random_noadj.log',
		 'palm/2hop_AndElim_random_noadj.log',
		 'llama/2hop_AndElim_random_noadj.json',
		 'flan-t5/latest/2hop_AndElim_random_noadj.json']
		],
		['conjunction\nintroduction', 'conjunction\nelimination'],
		title="ID: Train width = 2, test width = 2", width=3.2, legend_pos=(-0.25, 1.25), legend_cols=4)
make_diff_plot('width_accuracies_3testwidth.pdf', [
		['gpt_textdavinci003_2hop_AndIntro_3testwidth_random_noadj.log',
		 'palm/2hop_AndIntro_3testwidth_random_noadj.log',
		 'llama/2hop_AndIntro_3testwidth_random_noadj.json',
		 'flan-t5/latest/2hop_AndIntro_3testwidth_random_noadj.json'],
		['gpt_textdavinci003_2hop_AndElim_3testwidth_random_noadj.log',
		 'palm/2hop_AndElim_3testwidth_random_noadj.log',
		 'llama/2hop_AndElim_3testwidth_random_noadj.json',
		 'flan-t5/latest/2hop_AndElim_3testwidth_random_noadj.json']
		],
		['conjunction\nintroduction', 'conjunction\nelimination'],
		width_accuracies, width_lower_bound, width_upper_bound,
		title="OOD: Train width = 2, test width = 3", width=3.2, legend_pos=None)
make_diff_plot('width_accuracies_4testwidth.pdf', [
		['gpt_textdavinci003_2hop_AndIntro_4testwidth_random_noadj.log',
		 'palm/2hop_AndIntro_4testwidth_random_noadj.log',
		 'llama/2hop_AndIntro_4testwidth_random_noadj.json',
		 'flan-t5/latest/2hop_AndIntro_4testwidth_random_noadj.json'],
		['gpt_textdavinci003_2hop_AndElim_4testwidth_random_noadj.log',
		 'palm/2hop_AndElim_4testwidth_random_noadj.log',
		 'llama/2hop_AndElim_4testwidth_random_noadj.json',
		 'flan-t5/latest/2hop_AndElim_4testwidth_random_noadj.json']
		],
		['conjunction\nintroduction', 'conjunction\nelimination'],
		width_accuracies, width_lower_bound, width_upper_bound,
		title="OOD: Train width = 2, test width = 4", width=3.2, legend_pos=None)

if len(missing_files) != 0:
	print('[WARNING] The following files are missing:\n  ' + '\n  '.join(missing_files))
