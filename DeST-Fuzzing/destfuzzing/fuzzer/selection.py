import random
import math
import logging
import numpy as np
from destfuzzing.fuzzer.mutator import (
    MutateRandomSinglePolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten,
    OpenAIMutatorChangeStyle, OpenAIMutatorPolish, RewardBasedMutatePolicy,
    ALL_OPERATOR_NAMES, RADICAL_EXPLORATION_OPERATORS, CONSERVATIVE_INFILTRATION_OPERATORS)

from destfuzzing.fuzzer import destfuzzing, PromptNode


class SelectPolicy:
    """Base class for node selection policies."""
    def __init__(self, fuzzer: destfuzzing = None):
        self.fuzzer = fuzzer

    def select(self) -> PromptNode:
        raise NotImplementedError(
            "SelectPolicy must implement select method.")

    def update(self, prompt_nodes: 'list[PromptNode]'):
        pass


class RoundRobinSelectPolicy(SelectPolicy):
    def __init__(self, fuzzer: destfuzzing = None):
        super().__init__(fuzzer)
        self.index: int = 0

    def select(self) -> PromptNode:
        if not self.fuzzer or not self.fuzzer.prompt_nodes:
            return None
        seed = self.fuzzer.prompt_nodes[self.index % len(self.fuzzer.prompt_nodes)]
        seed.visited_num += 1
        return seed

    def update(self, prompt_nodes: 'list[PromptNode]'):
        self.index = (self.index + 1) % len(self.fuzzer.prompt_nodes) if self.fuzzer and self.fuzzer.prompt_nodes else 0


class RandomSelectPolicy(SelectPolicy):
    def __init__(self, fuzzer: destfuzzing = None):
        super().__init__(fuzzer)

    def select(self) -> PromptNode:
        if not self.fuzzer or not self.fuzzer.prompt_nodes:
            return None
        seed = random.choice(self.fuzzer.prompt_nodes)
        seed.visited_num += 1
        return seed


class UCBSelectPolicy(SelectPolicy):
    def __init__(self,
                 explore_coeff: float = 1.0,
                 fuzzer: destfuzzing = None):
        super().__init__(fuzzer)
        self.step = 0
        self.last_choice_index = None
        self.explore_coeff = explore_coeff
        self.rewards = []

    def select(self) -> PromptNode:
        if not self.fuzzer or not self.fuzzer.prompt_nodes:
            return None
            
        if len(self.fuzzer.prompt_nodes) > len(self.rewards):
            self.rewards.extend(
                [0 for _ in range(len(self.fuzzer.prompt_nodes) - len(self.rewards))])

        self.step += 1
        scores = np.zeros(len(self.fuzzer.prompt_nodes))
        for i, prompt_node in enumerate(self.fuzzer.prompt_nodes):
            smooth_visited_num = prompt_node.visited_num + 1
            scores[i] = self.rewards[i] / smooth_visited_num + \
                        self.explore_coeff * \
                        np.sqrt(2 * np.log(self.step) / smooth_visited_num)

        self.last_choice_index = np.argmax(scores)
        self.fuzzer.prompt_nodes[self.last_choice_index].visited_num += 1
        return self.fuzzer.prompt_nodes[self.last_choice_index]

    def update(self, prompt_nodes: 'list[PromptNode]'):
        succ_num_2 = sum([prompt_node.results.count(2) 
                          for prompt_node in prompt_nodes if prompt_node.results])
        succ_num_3 = sum([prompt_node.results.count(3) 
                          for prompt_node in prompt_nodes if prompt_node.results])
        succ_num = (succ_num_2 * 1.0) + (succ_num_3 * 1.5)
        if self.last_choice_index is not None and self.fuzzer and self.fuzzer.questions:
            self.rewards[self.last_choice_index] += succ_num / len(self.fuzzer.questions)


class MCTSExploreSelectPolicy(SelectPolicy):
    """Legacy MCTS selection policy with exploration path tracking."""
    def __init__(self, fuzzer: destfuzzing = None, ratio=0.5, alpha=0.1, beta=0.2):
        super().__init__(fuzzer)
        self.step = 0
        self.mctc_select_path: 'list[PromptNode]' = []
        self.last_choice_index = None
        self.rewards = []
        self.ratio = ratio
        self.alpha = alpha
        self.beta = beta

    def select(self) -> PromptNode:
        self.step += 1
        if len(self.fuzzer.prompt_nodes) > len(self.rewards):
            self.rewards.extend(
                [0 for _ in range(len(self.fuzzer.prompt_nodes) - len(self.rewards))])

        self.mctc_select_path.clear()
        cur = max(
            self.fuzzer.initial_prompts_nodes,
            key=lambda pn:
            self.rewards[pn.index] / (pn.visited_num + 1) +
            self.ratio * np.sqrt(2 * np.log(self.step) /
                                 (pn.visited_num + 0.01))
        )
        self.mctc_select_path.append(cur)

        while len(cur.child) > 0:
            if np.random.rand() < self.alpha:
                break
            cur = max(
                cur.child,
                key=lambda pn:
                self.rewards[pn.index] / (pn.visited_num + 1) +
                self.ratio * np.sqrt(2 * np.log(self.step) /
                                     (pn.visited_num + 0.01))
            )
            self.mctc_select_path.append(cur)

        for pn in self.mctc_select_path:
            pn.visited_num += 1

        self.last_choice_index = cur.index
        return cur

    def update(self, prompt_nodes: 'list[PromptNode]'):
        succ_num_2 = sum([prompt_node.results.count(2) 
                          for prompt_node in prompt_nodes if prompt_node.results])
        succ_num_3 = sum([prompt_node.results.count(3) 
                          for prompt_node in prompt_nodes if prompt_node.results])
        total_reward = (succ_num_2 * 1.0) + (succ_num_3 * 1.5)

        last_choice_node = self.fuzzer.prompt_nodes[self.last_choice_index]
        for prompt_node in reversed(self.mctc_select_path):
            reward = total_reward / (len(self.fuzzer.questions)
                                 * len(prompt_nodes))
            self.rewards[prompt_node.index] += reward * \
                max(self.beta, (1 - 0.1 * last_choice_node.level))
            prompt_node.reward = self.rewards[prompt_node.index]


class EnhancedMCTSSelectPolicy(SelectPolicy):
    """DeST-Fuzzing Stability-Controlled Tree Search selection policy.
    
    Implements Eq 17 for node selection with boundary potential, uncertainty penalty,
    and dynamic intermediate expansion. Uses the new PromptNode fields (W, N, uncertainty, etc.)
    for stability-controlled scoring.
    """
    def __init__(self, fuzzer: destfuzzing = None, ratio=3.0, alpha=0.3, beta=0.2,
                 reward_weights=(1.0, 1.5), uncertainty_gamma=0.3):
        super().__init__(fuzzer)
        self.step = 0
        self.search_path = []  
        self.last_choice_index = None
        self.rewards = []  
        self.uncertainties = []  
        self.visit_counts = []  

        self.ratio = ratio
        self.alpha = alpha
        self.beta = beta
        self.reward_weights = reward_weights
        self.uncertainty_gamma = uncertainty_gamma

    def select(self) -> PromptNode:
        """DeST-Fuzzing node selection using stability-controlled UCB score (Eq 17)."""
        if not self.fuzzer or not self.fuzzer.prompt_nodes:
            return None

        self.step += 1

        # Ensure tracking arrays are sized correctly
        current_node_count = len(self.fuzzer.prompt_nodes)
        if current_node_count > len(self.rewards):
            extend_count = current_node_count - len(self.rewards)
            self.rewards.extend([0.0] * extend_count)
            self.uncertainties.extend([1.0] * extend_count) 
            self.visit_counts.extend([0] * extend_count)

        self.search_path.clear()

        # Start from initial seed nodes
        current_node = self._select_best_child(self.fuzzer.initial_prompts_nodes)
        if current_node:
            self.search_path.append(current_node)

        # Descend through children
        while current_node and current_node.child:
            if random.random() < self.alpha:
                break
            current_node = self._select_best_child(current_node.child)
            if current_node:
                self.search_path.append(current_node)

        if not current_node:
            current_node = random.choice(self.fuzzer.prompt_nodes)

        # Dynamic intermediate expansion (Eq 15-16)
        if current_node and not current_node.child and random.random() < self.alpha:
            new_node = self._expand_node(current_node)
            if new_node:
                current_node.child.append(new_node)
                self.fuzzer.prompt_nodes.append(new_node)
                self.rewards.append(0.0)
                self.uncertainties.append(1.0)
                self.visit_counts.append(0)
                current_node = new_node
                self.search_path.append(current_node)

        # Update visit counts along search path
        for node in self.search_path:
            node.visited_num += 1
            idx = node.index
            if idx is not None and idx < len(self.visit_counts):
                self.visit_counts[idx] += 1

        self.last_choice_index = current_node.index if current_node else None
        return current_node

    def _select_best_child(self, nodes):
        """Select best child using DeST-Fuzzing stability-controlled score.
        
        Score = W̄_t(v) + c_v * sqrt(log(|V|+1)/(N(v)+1)) - gamma * Ũ_v  (Eq 17)
        """
        if not nodes:
            return None

        scores = []
        for node in nodes:
            idx = node.index
            if idx is None or idx >= len(self.rewards):
                scores.append(float('inf'))
                continue
                
            # Use DeST-Fuzzing fields if available
            if hasattr(node, 'W') and hasattr(node, 'N') and node.N > 0:
                exploitation = node.W / node.N  # Mean value from backups
                exploration = self.ratio * math.sqrt(
                    2 * math.log(self.step + 1) / node.N
                )
                uncertainty_penalty = self.uncertainty_gamma * node.uncertainty
                total_score = exploitation + exploration - uncertainty_penalty
            elif self.visit_counts[idx] > 0:
                # Legacy fallback
                exploitation = self.rewards[idx] / self.visit_counts[idx]
                exploration = self.ratio * np.sqrt(
                    2 * np.log(self.step) / self.visit_counts[idx]
                )
                uncertainty_bonus = self.uncertainty_gamma * self.uncertainties[idx]
                total_score = exploitation + exploration + uncertainty_bonus
            else:
                total_score = float('inf')
                
            scores.append(total_score)

        return nodes[np.argmax(scores)]

    def _expand_node(self, parent_node):
        """Create a new node by mutating the parent template."""
        try:
            new_index = len(self.fuzzer.prompt_nodes)
            mutated_prompts = self._mutate_prompt(parent_node.prompt)
            if isinstance(mutated_prompts, list) and len(mutated_prompts) > 0:
                new_prompt = mutated_prompts[0]
            else:
                new_prompt = mutated_prompts

            new_prompt_node = PromptNode(
                fuzzer=self.fuzzer,
                prompt=new_prompt,
                parent=parent_node
            )
            new_prompt_node.index = new_index
            return new_prompt_node
        except Exception as e:
            logging.warning(f"Node expansion failed: {e}")
            return None

    def _mutate_prompt(self, prompt):
        """Apply a random mutation operator to generate new prompt."""
        from destfuzzing.llm import OpenAILLM
        from destfuzzing.fuzzer.mutator import (
            MutateRandomSinglePolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand,
            OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten,
            OpenAIMutatorChangeStyle, OpenAIMutatorPolish)

        openai_model = OpenAILLM('gpt-3.5-turbo', '')
        mutate_policy = MutateRandomSinglePolicy([
            OpenAIMutatorChangeStyle(openai_model, temperature=0.5),
            OpenAIMutatorGenerateSimilar(openai_model, temperature=0.5),
            OpenAIMutatorShorten(openai_model, temperature=0.5),
            OpenAIMutatorRephrase(openai_model, temperature=0.5),
            OpenAIMutatorExpand(openai_model, temperature=0.5),
            OpenAIMutatorCrossOver(openai_model, temperature=0.5),
            OpenAIMutatorPolish(openai_model, temperature=0.5),
            ],
            concatentate=True,
        )
        mutate_policy.fuzzer = self.fuzzer
        mutator = random.choice(mutate_policy.mutators)
        
        prompt_text = prompt.prompt if hasattr(prompt, 'prompt') else prompt
        results = mutator.mutate_single(prompt_text)
        if mutate_policy.concatentate:
            results = [result + prompt_text for result in results]

        return results[0] if results else prompt_text

    def update(self, prompt_nodes: 'list[PromptNode]'):
        """Update rewards and uncertainties after evaluation (backward compatible)."""
        if not self.search_path:
            return

        # Compute rewards using DeST-Fuzzing fields if available
        succ_num_2 = sum([
            prompt_node.results.count(2)
            for prompt_node in prompt_nodes
            if hasattr(prompt_node, 'results') and prompt_node.results
        ])
        succ_num_3 = sum([
            prompt_node.results.count(3)
            for prompt_node in prompt_nodes
            if hasattr(prompt_node, 'results') and prompt_node.results
        ])

        total_reward = (succ_num_2 * self.reward_weights[0] +
                        succ_num_3 * self.reward_weights[1])
        total_questions = len(self.fuzzer.questions)
        if total_questions == 0 or len(prompt_nodes) == 0:
            base_reward = 0
        else:
            base_reward = total_reward / (total_questions * len(prompt_nodes))

        last_choice_node = None
        if self.last_choice_index is not None and self.last_choice_index < len(self.fuzzer.prompt_nodes):
            last_choice_node = self.fuzzer.prompt_nodes[self.last_choice_index]

        for i, prompt_node in enumerate(reversed(self.search_path)):
            idx = prompt_node.index
            if idx is None or idx >= len(self.rewards):
                continue

            level_penalty = 1.0
            if last_choice_node:
                level_penalty = max(self.beta, (1 - 0.1 * last_choice_node.level))

            adjusted_reward = base_reward * level_penalty
            self.rewards[idx] += adjusted_reward

            # Update uncertainty tracking
            if self.visit_counts[idx] > 0:
                reward_change = abs(adjusted_reward) / (self.visit_counts[idx] + 1)
                self.uncertainties[idx] = (
                        self.uncertainty_gamma * reward_change +
                        (1 - self.uncertainty_gamma) * self.uncertainties[idx]
                )

            prompt_node.reward = self.rewards[idx]

    def get_policy_info(self):
        if not self.rewards:
            return "Policy not initialized"

        avg_reward = np.mean(self.rewards)
        avg_uncertainty = np.mean(self.uncertainties)
        active_nodes = len([r for r in self.rewards if r > 0])

        return (f"EnhancedMCTS - Nodes: {len(self.rewards)}, "
                f"Active: {active_nodes}, Avg Reward: {avg_reward:.3f}, "
                f"Avg Uncertainty: {avg_uncertainty:.3f}")


class EXP3SelectPolicy(SelectPolicy):
    def __init__(self,
                 gamma: float = 0.05,
                 alpha: float = 25,
                 fuzzer: destfuzzing = None):
        super().__init__(fuzzer)

        self.energy = self.fuzzer.energy
        self.gamma = gamma
        self.alpha = alpha
        self.last_choice_index = None
        self.weights = [1. for _ in range(len(self.fuzzer.prompt_nodes))]
        self.probs = [0. for _ in range(len(self.fuzzer.prompt_nodes))]

    def select(self) -> PromptNode:
        if len(self.fuzzer.prompt_nodes) > len(self.weights):
            self.weights.extend(
                [1. for _ in range(len(self.fuzzer.prompt_nodes) - len(self.weights))])

        np_weights = np.array(self.weights)
        probs = (1 - self.gamma) * np_weights / np_weights.sum() + \
                self.gamma / len(self.fuzzer.prompt_nodes)

        self.last_choice_index = np.random.choice(
            len(self.fuzzer.prompt_nodes), p=probs)

        self.fuzzer.prompt_nodes[self.last_choice_index].visited_num += 1
        self.probs[self.last_choice_index] = probs[self.last_choice_index]

        return self.fuzzer.prompt_nodes[self.last_choice_index]

    def update(self, prompt_nodes: 'list[PromptNode]'):
        succ_num = sum([prompt_node.num_jailbreak
                        for prompt_node in prompt_nodes])

        r = 1 - succ_num / len(prompt_nodes)
        x = -1 * r / self.probs[self.last_choice_index]
        self.weights[self.last_choice_index] *= np.exp(
            self.alpha * x / len(self.fuzzer.prompt_nodes))