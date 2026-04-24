import random
import numpy as np
from gptfuzzer.fuzzer.mutator import (
    MutateRandomSinglePolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten, RewardBasedMutatePolicy)

from gptfuzzer.fuzzer import GPTFuzzer, PromptNode


class SelectPolicy:
    def __init__(self, fuzzer: GPTFuzzer):
        self.fuzzer = fuzzer

    def select(self) -> PromptNode:
        raise NotImplementedError(
            "SelectPolicy must implement select method.")

    def update(self, prompt_nodes: 'list[PromptNode]'):
        pass


class RoundRobinSelectPolicy(SelectPolicy):
    def __init__(self, fuzzer: GPTFuzzer = None):
        super().__init__(fuzzer)
        self.index: int = 0

    def select(self) -> PromptNode:
        seed = self.fuzzer.prompt_nodes[self.index]
        seed.visited_num += 1
        return seed

    def update(self, prompt_nodes: 'list[PromptNode]'):
        self.index = (self.index - 1 + len(self.fuzzer.prompt_nodes)
                      ) % len(self.fuzzer.prompt_nodes)


class RandomSelectPolicy(SelectPolicy):
    def __init__(self, fuzzer: GPTFuzzer = None):
        super().__init__(fuzzer)

    def select(self) -> PromptNode:
        seed = random.choice(self.fuzzer.prompt_nodes)
        seed.visited_num += 1
        return seed


class UCBSelectPolicy(SelectPolicy):
    def __init__(self,
                 explore_coeff: float = 1.0,
                 fuzzer: GPTFuzzer = None):
        super().__init__(fuzzer)

        self.step = 0
        self.last_choice_index = None
        self.explore_coeff = explore_coeff
        if self.fuzzer is not None and hasattr(self.fuzzer, 'prompt_nodes'):
            self.rewards = [0 for _ in range(len(self.fuzzer.prompt_nodes))]
        else:
            self.rewards = []

    def select(self) -> PromptNode:
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
        self.rewards[self.last_choice_index] += \
            succ_num / len(self.fuzzer.questions)


class MCTSExploreSelectPolicy(SelectPolicy):
    def __init__(self, fuzzer: GPTFuzzer = None, ratio=0.5, alpha=0.1, beta=0.2):
        super().__init__(fuzzer)

        self.step = 0
        self.mctc_select_path: 'list[PromptNode]' = []
        self.last_choice_index = None
        self.rewards = []
        self.ratio = ratio  # balance between exploration and exploitation
        self.alpha = alpha  # penalty for level
        self.beta = beta  # minimal reward after penalty

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

    def update(self, prompt_nodes: 'list[PromptNode]'):
        succ_num = sum([prompt_node.num_jailbreak
                        for prompt_node in prompt_nodes])

        last_choice_node = self.fuzzer.prompt_nodes[self.last_choice_index]
        for prompt_node in reversed(self.mctc_select_path):
            reward = succ_num / (len(self.fuzzer.questions)
                                 * len(prompt_nodes))
            self.rewards[prompt_node.index] += reward * \
                                               max(self.beta, (1 - 0.1 * last_choice_node.level))


class EnhancedMCTSSelectPolicy(SelectPolicy):
    def __init__(self, fuzzer: GPTFuzzer = None, ratio=3.0, alpha=0.3, beta=0.2,
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
        self.reward_weights = reward_weights  # (weight_for_2, weight_for_3)
        self.uncertainty_gamma = uncertainty_gamma

    def select(self) -> PromptNode:

        self.step += 1

        current_node_count = len(self.fuzzer.prompt_nodes)
        if current_node_count > len(self.rewards):
            extend_count = current_node_count - len(self.rewards)
            self.rewards.extend([0.0] * extend_count)
            self.uncertainties.extend([1.0] * extend_count) 
            self.visit_counts.extend([0] * extend_count)

        self.search_path.clear()

        current_node = self._select_best_child(self.fuzzer.initial_prompts_nodes)
        if current_node:
            self.search_path.append(current_node)

        while current_node and current_node.child:
            if random.random() < self.alpha:
                break

            current_node = self._select_best_child(current_node.child)
            if current_node:
                self.search_path.append(current_node)

        if not current_node:
            current_node = random.choice(self.fuzzer.prompt_nodes)

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

        for node in self.search_path:
            node.visited_num += 1
            idx = node.index
            if idx < len(self.visit_counts):
                self.visit_counts[idx] += 1

        self.last_choice_index = current_node.index
        return current_node

    def _select_best_child(self, nodes):

        if not nodes:
            return None

        scores = []
        for node in nodes:
            idx = node.index
            if idx >= len(self.rewards) or self.visit_counts[idx] == 0:
                scores.append(float('inf'))
            else:
                exploitation = self.rewards[idx] / self.visit_counts[idx]
                exploration = self.ratio * np.sqrt(
                    2 * np.log(self.step) / self.visit_counts[idx]
                )

                uncertainty_bonus = self.uncertainty_gamma * self.uncertainties[idx]

                total_score = exploitation + exploration + uncertainty_bonus
                scores.append(total_score)

        return nodes[np.argmax(scores)]

    def _expand_node(self, parent_node):
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
            print(f"节点扩展失败: {e}")
            return None

    def _mutate_prompt(self, prompt):
        from gptfuzzer.llm import OpenAILLM
        from gptfuzzer.fuzzer.mutator import MutateRandomSinglePolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand, \
            OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten,OpenAIMutatorChangeStyle

        openai_model = OpenAILLM('gpt-3.5-turbo', '')
        mutate_policy = MutateRandomSinglePolicy([
            OpenAIMutatorChangeStyle(openai_model, temperature=0.0),
            OpenAIMutatorGenerateSimilar(openai_model, temperature=0.0),
            OpenAIMutatorShorten(openai_model, temperature=0.0),
            OpenAIMutatorRephrase(openai_model, temperature=0.0),
            OpenAIMutatorExpand(openai_model, temperature=0.0),
            OpenAIMutatorCrossOver(openai_model, temperature=0.0)
            ],
            concatentate=True,
        )
        mutate_policy.fuzzer = self.fuzzer
        mutator = random.choice(mutate_policy.mutators)
        if hasattr(prompt, 'prompt'):
            prompt_text = prompt.prompt
        else:
            prompt_text = prompt

        results = mutator.mutate_single(prompt_text)
        if mutate_policy.concatentate:
            results = [result + prompt_text for result in results]

        if results:
            return results[0]
        else:
            return prompt_text

    def update(self, prompt_nodes: 'list[PromptNode]'):
        if not self.search_path:
            return

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
            if idx >= len(self.rewards):
                continue

            level_penalty = 1.0
            if last_choice_node:
                level_penalty = max(self.beta, (1 - 0.1 * last_choice_node.level))

            adjusted_reward = base_reward * level_penalty

            old_reward = self.rewards[idx]
            self.rewards[idx] += adjusted_reward

            if self.visit_counts[idx] > 0:
                reward_change = abs(adjusted_reward) / (self.visit_counts[idx] + 1)
                self.uncertainties[idx] = (
                        self.uncertainty_gamma * reward_change +
                        (1 - self.uncertainty_gamma) * self.uncertainties[idx]
                )

            prompt_node.reward = self.rewards[idx]

    def get_policy_info(self):
        if not self.rewards:
            return "策略未初始化"

        avg_reward = np.mean(self.rewards)
        avg_uncertainty = np.mean(self.uncertainties)
        active_nodes = len([r for r in self.rewards if r > 0])

        return (f"EnhancedMCTS - 节点数: {len(self.rewards)}, "
                f"活跃节点: {active_nodes}, 平均奖励: {avg_reward:.3f}, "
                f"平均不确定性: {avg_uncertainty:.3f}")


class EXP3SelectPolicy(SelectPolicy):
    def __init__(self,
                 gamma: float = 0.05,
                 alpha: float = 25,
                 fuzzer: GPTFuzzer = None):
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