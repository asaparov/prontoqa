import fol
from theory import *
import numpy as np
from enum import Enum
from random import choice, random

class ProofStepType(Enum):
	AXIOM = 0
	UNIVERSAL_INSTANTIATION = 1 # given ![x]:(f(x) -> g(x)) and f(a), conclude g(a)
	CONJUNCTION_ELIMINATION = 2 # given A_1 & ... & A_n, conclude A_i

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

def generate_membership_question(theory, entity_name, num_deduction_steps=None, generate_questions_about_types=True, generate_questions_about_properties=True):
	nodes = get_nodes(theory)
	max_level = 0
	for _, level in nodes:
		max_level = max(max_level, level)
	sufficiently_deep_nodes = []
	for node, level in nodes:
		if level >= num_deduction_steps:
			sufficiently_deep_nodes.append(node)
	if len(sufficiently_deep_nodes) == 0:
		return (None, None, None, None)
	start = choice(sufficiently_deep_nodes)

	start_formula = fol.FOLFuncApplication(start.name, [fol.FOLConstant(entity_name)])
	start_axiom = ProofStep(ProofStepType.AXIOM, [], start_formula)
	current_node = start
	current_step = start_axiom
	current_num_steps = 1
	while True:
		property_question_weight = 0.3 if generate_questions_about_properties else 0.0
		negated_property_question_weight = 0.7 if generate_questions_about_properties else 0.0
		probabilities = [1] + [property_question_weight] * len(current_node.properties) + [negated_property_question_weight] * len(current_node.negated_properties)
		probabilities = np.array(probabilities) / np.sum(probabilities)
		index = np.random.choice(1 + len(current_node.properties) + len(current_node.negated_properties), p=probabilities)
		if index == 0:
			if current_node.parent == None:
				if not generate_questions_about_types:
					return (None, None, None, None)
				break
			subsumption_formula = get_subsumption_formula(current_node)
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
			properties_formula = get_properties_formula(current_node)
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
			properties_formula = get_properties_formula(current_node)
			properties_step = ProofStep(ProofStepType.AXIOM, [], properties_formula)
			conclusion = fol.substitute(properties_formula.operand.consequent, fol.FOLVariable(1), fol.FOLConstant(entity_name))
			next_step = ProofStep(ProofStepType.UNIVERSAL_INSTANTIATION, [properties_step, current_step], conclusion)
			current_step = next_step
			current_num_steps += 1

			if type(conclusion) == fol.FOLAnd:
				current_step = ProofStep(ProofStepType.CONJUNCTION_ELIMINATION, [current_step], fol.FOLNot(fol.FOLFuncApplication(current_node.properties[index], [fol.FOLConstant(entity_name)])))
				current_num_steps += 1
			break

	return (start_formula, current_step.conclusion, current_step, current_num_steps)

def get_proof_intermediate_formulas(last_step):
	if last_step.step_type == ProofStepType.AXIOM:
		return [last_step.conclusion]
	elif last_step.step_type == ProofStepType.CONJUNCTION_ELIMINATION:
		return get_proof_intermediate_formulas(last_step.premises[0]) + [last_step.conclusion]
	elif last_step.step_type == ProofStepType.UNIVERSAL_INSTANTIATION:
		return get_proof_intermediate_formulas(last_step.premises[1]) + get_proof_intermediate_formulas(last_step.premises[0]) + [last_step.conclusion]
	else:
		raise Exception("get_proof_intermediate_formulas ERROR: Unsupported proof step type.")
