import fol
from theory import *
import numpy as np
from enum import Enum
from random import choice, random
import itertools

class ProofStepType(Enum):
	AXIOM = 0
	UNIVERSAL_INSTANTIATION = 1 # given ![x]:(f(x) -> g(x)) and f(a), conclude g(a)
	CONJUNCTION_INTRODUCTION = 2 # given A_1, ..., A_n, conclude A_1 & ... & A_n
	CONJUNCTION_ELIMINATION = 3 # given A_1 & ... & A_n, conclude A_i
	DISJUNCTION_INTRODUCTION = 4 # given A_i, conclude A_1 | ... | A_n
	DISJUNCTION_ELIMINATION = 5 # given A or B, A proves C, B proves C, conclude C

class ProofStep(object):
	def __init__(self, step_type, premises, conclusion):
		self.step_type = step_type
		self.premises = premises
		self.conclusion = conclusion

def get_nodes(theory, level=0):
	nodes = [(theory, level)]
	for child in theory.children:
		nodes.extend(get_nodes(child, level + 1))
	return nodes

def generate_membership_question(theory, entity_name, num_deduction_steps=None, generate_questions_about_types=True, generate_questions_about_properties=True, deduction_rule="ModusPonens", use_dfs=True):
	nodes = get_nodes(theory)
	max_level = 0
	for _, level in nodes:
		max_level = max(max_level, level)
	sufficiently_deep_nodes = []
	for node, level in nodes:
		if level + 1 >= num_deduction_steps:
			sufficiently_deep_nodes.append(node)
	if len(sufficiently_deep_nodes) == 0:
		return (None, None, None, None, None)
	start = choice(sufficiently_deep_nodes)

	if deduction_rule == "AndElim":
		if len(node.properties) == 1 and len(node.negated_properties) == 0:
			other = fol.FOLFuncApplication(node.properties[0], [fol.FOLConstant(entity_name)])
		elif len(node.properties) == 0 and len(node.negated_properties) == 1:
			other = fol.FOLNot(fol.FOLFuncApplication(node.negated_properties[0], [fol.FOLConstant(entity_name)]))
		else:
			raise ValueError("Expected exactly one defining property.")
		start_formula = fol.FOLAnd([other, fol.FOLFuncApplication(start.name, [fol.FOLConstant(entity_name)])])
	else:
		start_formula = fol.FOLFuncApplication(start.name, [fol.FOLConstant(entity_name)])
	premises = [start_formula]
	start_axiom = ProofStep(ProofStepType.AXIOM, [], start_formula)
	current_node = start
	current_step = start_axiom
	current_num_steps = 1

	branch_points = []
	while True:
		if deduction_rule == "AndIntro":
			if current_node.parent == None:
				if not generate_questions_about_types:
					return (None, None, None, None, None)
				break
			subsumption_formula = get_subsumption_formula(current_node, deduction_rule)
			other_axiom = fol.substitute(subsumption_formula.operand.antecedent.operands[0], fol.FOLVariable(1), fol.FOLConstant(entity_name))
			other_axiom_step = ProofStep(ProofStepType.AXIOM, [], other_axiom)
			premises.append(other_axiom)
			intro_step_conclusion = fol.FOLAnd([other_axiom, current_step.conclusion])
			current_step = ProofStep(ProofStepType.CONJUNCTION_INTRODUCTION, [other_axiom_step, current_step], intro_step_conclusion)

			current_num_steps += 1
			if generate_questions_about_types and ((num_deduction_steps != None and current_num_steps == num_deduction_steps) or (num_deduction_steps == None and random() < 0.3)):
				break

			subsumption_step = ProofStep(ProofStepType.AXIOM, [], subsumption_formula)
			conclusion = fol.FOLFuncApplication(current_node.parent.name, [fol.FOLConstant(entity_name)])
			next_step = ProofStep(ProofStepType.UNIVERSAL_INSTANTIATION, [subsumption_step, current_step], conclusion)

			current_node = current_node.parent
			current_step = next_step

		elif deduction_rule == "AndElim":
			if current_node.parent == None:
				if not generate_questions_about_types:
					return (None, None, None, None, None)
				break
			elim_step_conclusion = current_step.conclusion.operands[1]
			current_step = ProofStep(ProofStepType.CONJUNCTION_ELIMINATION, [current_step], elim_step_conclusion)

			current_num_steps += 1
			if generate_questions_about_types and ((num_deduction_steps != None and current_num_steps == num_deduction_steps) or (num_deduction_steps == None and random() < 0.3)):
				break

			subsumption_formula = get_subsumption_formula(current_node, deduction_rule)
			subsumption_step = ProofStep(ProofStepType.AXIOM, [], subsumption_formula)
			conclusion = fol.substitute(subsumption_formula.operand.consequent, fol.FOLVariable(1), fol.FOLConstant(entity_name))
			next_step = ProofStep(ProofStepType.UNIVERSAL_INSTANTIATION, [subsumption_step, current_step], conclusion)

			current_node = current_node.parent
			current_step = next_step

		elif deduction_rule == "OrIntro":
			if current_node.parent == None:
				if not generate_questions_about_types:
					return (None, None, None, None, None)
				break
			subsumption_formula = get_subsumption_formula(current_node, deduction_rule)
			other_axiom = fol.substitute(subsumption_formula.operand.antecedent.operands[0], fol.FOLVariable(1), fol.FOLConstant(entity_name))
			intro_step_conclusion = fol.FOLOr([other_axiom, current_step.conclusion])
			current_step = ProofStep(ProofStepType.DISJUNCTION_INTRODUCTION, [current_step], intro_step_conclusion)

			current_num_steps += 1
			if generate_questions_about_types and ((num_deduction_steps != None and current_num_steps == num_deduction_steps) or (num_deduction_steps == None and random() < 0.3)):
				break

			subsumption_step = ProofStep(ProofStepType.AXIOM, [], subsumption_formula)
			conclusion = fol.FOLFuncApplication(current_node.parent.name, [fol.FOLConstant(entity_name)])
			next_step = ProofStep(ProofStepType.UNIVERSAL_INSTANTIATION, [subsumption_step, current_step], conclusion)

			current_node = current_node.parent
			current_step = next_step

		else:
			property_question_weight = 0.3 if generate_questions_about_properties else 0.0
			negated_property_question_weight = 0.7 if generate_questions_about_properties else 0.0
			probabilities = [1] + [property_question_weight] * len(current_node.properties) + [negated_property_question_weight] * len(current_node.negated_properties)
			probabilities = np.array(probabilities) / np.sum(probabilities)
			index = np.random.choice(1 + len(current_node.properties) + len(current_node.negated_properties), p=probabilities)
			if (1 if current_node.parent != None else 0) + len(current_node.properties) + len(current_node.negated_properties) > 1:
				branch_points.append(current_node)
			if index == 0:
				if current_node.parent == None:
					if not generate_questions_about_types:
						return (None, None, None, None, None)
					break
				subsumption_formula = get_subsumption_formula(current_node, deduction_rule)
				subsumption_step = ProofStep(ProofStepType.AXIOM, [], subsumption_formula)
				conclusion = fol.FOLFuncApplication(current_node.parent.name, [fol.FOLConstant(entity_name)])
				next_step = ProofStep(ProofStepType.UNIVERSAL_INSTANTIATION, [subsumption_step, current_step], conclusion)

				current_node = current_node.parent
				current_step = next_step
				current_num_steps += 1
				if generate_questions_about_types and ((num_deduction_steps != None and current_num_steps == num_deduction_steps) or (num_deduction_steps == None and random() < 0.3)):
					break

			elif index == 1:
				index -= 1
				properties_formulas = []
				get_properties_formula(properties_formulas, current_node, deduction_rule)
				properties_formula = properties_formulas[0]
				properties_step = ProofStep(ProofStepType.AXIOM, [], properties_formula)
				conclusion = fol.substitute(properties_formula.operand.consequent, fol.FOLVariable(1), fol.FOLConstant(entity_name))
				next_step = ProofStep(ProofStepType.UNIVERSAL_INSTANTIATION, [properties_step, current_step], conclusion)
				current_step = next_step
				current_num_steps += 1

				if type(conclusion) == fol.FOLAnd:
					current_step = ProofStep(ProofStepType.CONJUNCTION_ELIMINATION, [current_step], fol.FOLFuncApplication(current_node.properties[index], [fol.FOLConstant(entity_name)]))
					current_num_steps += 1
				break

			else:
				index -= 1 + len(current_node.properties)
				properties_formulas = []
				get_properties_formula(properties_formulas, current_node, deduction_rule)
				properties_formula = properties_formulas[0]
				properties_step = ProofStep(ProofStepType.AXIOM, [], properties_formula)
				conclusion = fol.substitute(properties_formula.operand.consequent, fol.FOLVariable(1), fol.FOLConstant(entity_name))
				next_step = ProofStep(ProofStepType.UNIVERSAL_INSTANTIATION, [properties_step, current_step], conclusion)
				current_step = next_step
				current_num_steps += 1

				if type(conclusion) == fol.FOLAnd:
					current_step = ProofStep(ProofStepType.CONJUNCTION_ELIMINATION, [current_step], fol.FOLNot(fol.FOLFuncApplication(current_node.properties[index], [fol.FOLConstant(entity_name)])))
					current_num_steps += 1
				break

	linearized_proof = linearize_proof_steps(current_step)
	if use_dfs and len(branch_points) > 0:
		branch_index = 0
		for step in linearized_proof:
			if type(step.conclusion) == fol.FOLFuncApplication and step.conclusion.function == branch_points[-1].name:
				break
			branch_index += 1
		dfs_node = branch_points[-1]
		dfs_step = linearized_proof[branch_index]

		#import pdb; pdb.set_trace()
		dfs_proof = []
		while True:
			probabilities = [1 if dfs_node.parent != None else 0] + [1] * len(dfs_node.properties) + [1] * len(dfs_node.negated_properties)
			if np.sum(probabilities) == 0:
				break
			probabilities = np.array(probabilities) / np.sum(probabilities)
			index = np.random.choice(1 + len(dfs_node.properties) + len(dfs_node.negated_properties), p=probabilities)
			if index == 0:
				subsumption_formula = get_subsumption_formula(dfs_node, deduction_rule)
				subsumption_step = ProofStep(ProofStepType.AXIOM, [], subsumption_formula)
				dfs_proof.append(subsumption_step)
				conclusion = fol.FOLFuncApplication(dfs_node.parent.name, [fol.FOLConstant(entity_name)])
				next_step = ProofStep(ProofStepType.UNIVERSAL_INSTANTIATION, [subsumption_step, dfs_step], conclusion)

				dfs_node = dfs_node.parent
				dfs_step = next_step
				dfs_proof.append(dfs_step)

			elif index == 1:
				index -= 1
				properties_formulas = []
				get_properties_formula(properties_formulas, dfs_node, deduction_rule)
				properties_formula = properties_formulas[0]
				properties_step = ProofStep(ProofStepType.AXIOM, [], properties_formula)
				dfs_proof.append(properties_step)
				conclusion = fol.substitute(properties_formula.operand.consequent, fol.FOLVariable(1), fol.FOLConstant(entity_name))
				next_step = ProofStep(ProofStepType.UNIVERSAL_INSTANTIATION, [properties_step, dfs_step], conclusion)
				dfs_step = next_step
				dfs_proof.append(dfs_step)

				if type(conclusion) == fol.FOLAnd:
					dfs_step = ProofStep(ProofStepType.CONJUNCTION_ELIMINATION, [dfs_step], fol.FOLFuncApplication(dfs_node.properties[index], [fol.FOLConstant(entity_name)]))
					dfs_proof.append(dfs_step)
				break

			else:
				index -= 1 + len(dfs_node.properties)
				properties_formulas = []
				get_properties_formula(properties_formulas, dfs_node, deduction_rule)
				properties_formula = properties_formulas[0]
				properties_step = ProofStep(ProofStepType.AXIOM, [], properties_formula)
				dfs_proof.append(properties_step)
				conclusion = fol.substitute(properties_formula.operand.consequent, fol.FOLVariable(1), fol.FOLConstant(entity_name))
				next_step = ProofStep(ProofStepType.UNIVERSAL_INSTANTIATION, [properties_step, dfs_step], conclusion)
				dfs_step = next_step
				dfs_proof.append(dfs_step)

				if type(conclusion) == fol.FOLAnd:
					dfs_step = ProofStep(ProofStepType.CONJUNCTION_ELIMINATION, [dfs_step], fol.FOLNot(fol.FOLFuncApplication(dfs_node.properties[index], [fol.FOLConstant(entity_name)])))
					dfs_proof.append(dfs_step)
				break

		if dfs_proof[0] != linearized_proof[branch_index + 1]:
			linearized_proof[(branch_index + 1):(branch_index + 1)] = dfs_proof

	return (premises, current_step.conclusion, current_step, current_num_steps, linearized_proof)

def linearize_proof_steps(last_step):
	if last_step.step_type == ProofStepType.AXIOM:
		return [last_step]
	elif last_step.step_type == ProofStepType.UNIVERSAL_INSTANTIATION:
		return linearize_proof_steps(last_step.premises[1]) + linearize_proof_steps(last_step.premises[0]) + [last_step]
	elif last_step.step_type == ProofStepType.CONJUNCTION_INTRODUCTION:
		return list(itertools.chain.from_iterable([linearize_proof_steps(premise) for premise in reversed(last_step.premises)])) + [last_step]
	else:
		return list(itertools.chain.from_iterable([linearize_proof_steps(premise) for premise in last_step.premises])) + [last_step]
