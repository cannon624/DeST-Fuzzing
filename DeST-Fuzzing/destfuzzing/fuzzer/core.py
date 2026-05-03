import logging
import time
import csv
import math
import numpy as np
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mutator import Mutator, MutatePolicy
    from .selection import SelectPolicy

from destfuzzing.llm import LLM, LocalLLM, OpenAILLM
from destfuzzing.utils.template import synthesis_message
from destfuzzing.utils.predict import Predictor, compute_boundary_potential, compute_defense_state, compute_response_uncertainty
import warnings


class PromptNode:
    """DeST-Fuzzing template tree node (Eq 12).
    
    v = (T_v, s_v, Phi_v, U_v, N_v, W_v, d_v)
    """
    def __init__(self,
                 fuzzer: 'destfuzzing',
                 prompt: str,
                 response: str = None,
                 results: 'list[int]' = None,
                 parent: 'PromptNode' = None,
                 mutator: 'Mutator' = None):
        self.fuzzer: 'destfuzzing' = fuzzer
        self.prompt: str = prompt
        self.response: str = response
        self.results: 'list[int]' = results  # K state labels from repeated evaluation
        self.visited_num = 0
        self.reward = 0.0

        self.parent: 'PromptNode' = parent
        self.mutator: 'Mutator' = mutator
        self.child: 'list[PromptNode]' = []
        self.level: int = 0 if parent is None else parent.level + 1

        self._index: int = None

        # DeST-Fuzzing fields (Eq 12)
        self.defense_state: int = 0          # s_v: most frequent defense state
        self.boundary_potential: float = 0.0  # Phi_v: boundary potential
        self.uncertainty: float = 1.0         # U_v: normalized response-state uncertainty
        self.W: float = 0.0                   # W_v: accumulated value from backups
        self.N: int = 0                       # N_v: backup count

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index
        if self.parent is not None:
            self.parent.child.append(self)

    @property
    def num_jailbreak(self):
        return sum(1 for result in self.results if result in [2, 3]) if self.results else 0

    @property
    def num_reject(self):
        return sum(1 for result in self.results if result in [0, 1]) if self.results else 0

    @property
    def num_query(self):
        return len(self.results) if self.results else 0

    def set_state_from_labels(self, state_labels: 'list[int]'):
        """Compute and store defense state, boundary potential, and uncertainty from K observations."""
        self.results = state_labels
        self.defense_state = compute_defense_state(state_labels)
        self.boundary_potential = compute_boundary_potential(state_labels)
        self.uncertainty = compute_response_uncertainty(state_labels)


class destfuzzing:
    """DeST-Fuzzing: Defense-State Transition Optimization for Stable Jailbreak Discovery.
    
    Implements Algorithm 1 from the paper. Key parameters:
    - K: number of repeated target-model calls for state estimation (sampling number)
    - B: query budget (max total queries)
    - H: maximum tree depth
    - lambda: uncertainty penalty in transition reward
    - alpha: attenuation strength for uncertainty gate
    - c_v: exploration constant for node selection
    - c_a: exploration constant for operator selection
    - gamma: uncertainty penalty coefficient in node selection
    - beta: smoothing constant for transition kernel
    - n_min: minimum visits for dynamic intermediate expansion
    - delta_die: improvement margin for dynamic intermediate expansion
    """
    def __init__(self,
                 questions: 'list[str]',
                 target: 'LLM',
                 predictor: 'Predictor',
                 initial_seed: 'list[str]',
                 mutate_policy: 'MutatePolicy',
                 select_policy: 'SelectPolicy',
                 max_query: int = -1,
                 max_jailbreak: int = -1,
                 max_reject: int = -1,
                 max_iteration: int = -1,
                 energy: int = 1,
                 result_file: str = None,
                 generate_in_batch: bool = False,
                 openai_model = None,
                 # DeST-Fuzzing specific parameters
                 K: int = 3,                # Sampling number for repeated evaluation
                 H: int = 6,                # Maximum tree depth
                 lambda_uncertainty: float = 0.3,  # Uncertainty penalty in transition reward
                 alpha_gate: float = 2.0,   # Attenuation strength for uncertainty gate
                 c_v: float = 1.2,          # Exploration constant for node selection
                 c_a: float = 0.7,          # Exploration constant for operator selection
                 gamma_uncertainty: float = 0.3,  # Uncertainty penalty in node selection
                 beta_smooth: float = 0.1,  # Smoothing constant for transition kernel
                 n_min: int = 2,            # Minimum visits for dynamic intermediate expansion
                 delta_die: float = 0.02,   # Improvement margin for dynamic intermediate expansion
                 ):

        self.openai_model = openai_model
        self.questions: 'list[str]' = questions
        self.target: LLM = target
        self.predictor = predictor
        self.prompt_nodes: 'list[PromptNode]' = [
            PromptNode(self, prompt) for prompt in initial_seed
        ]
        self.initial_prompts_nodes = self.prompt_nodes.copy()

        for i, prompt_node in enumerate(self.prompt_nodes):
            prompt_node.index = i

        self.mutate_policy = mutate_policy
        self.select_policy = select_policy

        self.current_query: int = 0
        self.current_jailbreak: int = 0
        self.current_reject: int = 0
        self.current_iteration: int = 0

        self.max_query: int = max_query
        self.max_jailbreak: int = max_jailbreak
        self.max_reject: int = max_reject
        self.max_iteration: int = max_iteration

        self.energy: int = energy
        if result_file is None:
            result_file = f'results-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.csv'

        self.raw_fp = open(result_file, 'w', buffering=1, encoding='utf-8')
        self.writter = csv.writer(self.raw_fp)
        self.writter.writerow(
            ['question', 'prompt', 'response', 'parent', 'results', 'defense_state', 'boundary_potential', 'uncertainty'])

        self.generate_in_batch = False
        if len(self.questions) > 0 and generate_in_batch is True:
            self.generate_in_batch = True
            if isinstance(self.target, LocalLLM):
                warnings.warn("IMPORTANT! Hugging face inference with batch generation has the problem of consistency due to pad tokens. We do not suggest doing so and you may experience (1) degraded output quality due to long padding tokens, (2) inconsistent responses due to different number of padding tokens during reproduction. You should turn off generate_in_batch or use vllm batch inference.")
        
        # DeST-Fuzzing parameters
        self.K: int = K
        self.H: int = H
        self.lambda_uncertainty: float = lambda_uncertainty
        self.alpha_gate: float = alpha_gate
        self.c_v: float = c_v
        self.c_a: float = c_a
        self.gamma_uncertainty: float = gamma_uncertainty
        self.beta_smooth: float = beta_smooth
        self.n_min: int = n_min
        self.delta_die: float = delta_die
        
        # Transition tracking (Eq 8-11)
        # C(s, a, s'): number of observed transitions from state s to s' under operator a
        self.transition_counts: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # N_op(s, a): total number of operator a selections under state s
        self.operator_counts: dict = defaultdict(lambda: defaultdict(int))
        # N_state(s): total number of operator selections under state s
        self.state_counts: dict = defaultdict(int)
        # G(s, a): state-conditioned operator utility
        self.operator_utilities: dict = defaultdict(lambda: defaultdict(float))
        # Accumulated transition rewards for g_bar computation
        self.transition_rewards: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        
        # Output candidate set (templates that achieved state 3)
        self.output_candidates: list = []
        self.question_summaries: list = []
        
        self.setup()

    def setup(self):
        self.mutate_policy.fuzzer = self
        self.select_policy.fuzzer = self
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s %(message)s', datefmt='[%H:%M:%S]')

    def is_stop(self):
        checks = [
            ('max_query', 'current_query'),
            ('max_jailbreak', 'current_jailbreak'),
            ('max_reject', 'current_reject'),
            ('max_iteration', 'current_iteration'),
        ]
        return any(getattr(self, max_attr) != -1 and getattr(self, curr_attr) >= getattr(self, max_attr) for max_attr, curr_attr in checks)

    # ============================================================
    # DeST-Fuzzing: Algorithm 1 - Main Search Loop
    # ============================================================
    
    def run(self):
        """Execute DeST-Fuzzing Algorithm 1 for each question.
        
        Algorithm 1:
        1. Initialize tree, transition counts, utilities, query counter
        2. For each seed template, evaluate K times, estimate state stats, insert seed node
        3. While q + K <= B:
           a. Construct expandable candidate set C_t (Eq 16)
           b. Select parent node v_t (Eq 17)
           c. Select mutation operator a_t (Eq 19)
           d. Generate child template, evaluate K times
           e. Insert child node, compute reward, apply uncertainty gate
           f. Update tree values, transition counts, operator utilities
        4. Select best template T* by maximizing Phi - lambda * U
        """
        logging.info("DeST-Fuzzing started!")
        
        total_questions = len(self.questions)
        jailbroken_questions = 0
        self.question_summaries = []
        
        try:
            for q_idx, question in enumerate(self.questions):
                logging.info(f"Processing question {q_idx+1}/{total_questions}: {question}")
                
                # Store original question list and set single question
                original_questions = self.questions
                self.questions = [question]
                
                # Reset tree and tracking for this question
                self._reset_for_question()
                
                # Step 1-2: Initialize seed nodes with K evaluations
                self._initialize_seed_nodes(question)
                
                # Step 3: Main search loop
                query_budget = self.max_query if self.max_query != -1 else 300
                
                while self.current_query + self.K <= query_budget:
                    # Step 3a: Construct expandable candidate set C_t (Eq 16)
                    expandable = self._get_expandable_nodes()
                    if not expandable:
                        logging.info("No expandable nodes remaining.")
                        break
                    
                    # Step 3b: Select parent node v_t (Eq 17)
                    parent_node = self._select_parent_node(expandable)
                    
                    # Step 3c: Select mutation operator a_t (Eq 19)
                    operator = self._select_mutation_operator(parent_node)
                    
                    # Step 3d: Generate child template
                    child_prompt = self._apply_operator(operator, parent_node.prompt)
                    if child_prompt is None:
                        continue
                    
                    # Step 3e: Evaluate child template K times
                    child_node = self._evaluate_template(child_prompt, question, parent_node, operator)
                    
                    # Step 3f: Compute transition reward (Eq 20)
                    delta_phi = child_node.boundary_potential - parent_node.boundary_potential
                    r_t = delta_phi - self.lambda_uncertainty * child_node.uncertainty
                    
                    # Step 3g: Apply uncertainty gate (Eq 21)
                    omega_t = math.exp(-self.alpha_gate * child_node.uncertainty)
                    r_tilde = omega_t * max(r_t, 0) + min(r_t, 0)
                    
                    # Step 3h: Update tree values (Eq 22) - backup to ancestors
                    self._backup_value(child_node, r_tilde)
                    
                    # Step 3i: Update transition counts and operator utilities
                    self._update_transition_stats(parent_node, operator, child_node, delta_phi)
                    
                    # Log progress
                    if child_node.defense_state == 3:
                        self.output_candidates.append(child_node)
                        logging.info(f"State 3 reached! Potential={child_node.boundary_potential:.3f}, Uncertainty={child_node.uncertainty:.3f}")
                    
                    self.current_query += self.K
                    
                    if child_node.num_jailbreak > 0:
                        self.current_jailbreak += child_node.num_jailbreak
                    
                    self.log()
                    self.current_iteration += 1
                    
                    if self.is_stop():
                        break
                
                # Step 4: Select best template T*
                best_template = self._select_best_template()
                if best_template:
                    logging.info(f"Best template for question {q_idx+1}: state={best_template.defense_state}, "
                               f"potential={best_template.boundary_potential:.3f}, uncertainty={best_template.uncertainty:.3f}")
                    
                    # Write all nodes to CSV
                    self.write_prompt_nodes_to_csv()
                    
                    question_has_jailbreak = any(c.defense_state == 3 for c in self.output_candidates)
                    self.question_summaries.append({
                        'question': question,
                        'best_prompt': best_template.prompt,
                        'defense_state': best_template.defense_state,
                        'boundary_potential': best_template.boundary_potential,
                        'uncertainty': best_template.uncertainty,
                        'queries': self.current_query,
                        'nodes': len(self.prompt_nodes),
                        'jailbreak_found': question_has_jailbreak,
                    })
                    
                    if question_has_jailbreak:
                        jailbroken_questions += 1
                        logging.info(f"Jailbreak found for question {q_idx+1}")
                    else:
                        logging.info(f"No jailbreak found for question {q_idx+1}")
                
                # Restore original questions
                self.questions = original_questions
            
            logging.info(f"DeST-Fuzzing completed. Jailbroken {jailbroken_questions}/{total_questions} questions.")
            
        except KeyboardInterrupt:
            logging.info("DeST-Fuzzing interrupted by user!")
        finally:
            logging.info("DeST-Fuzzing finished!")
            self.raw_fp.close()
    
    def _reset_for_question(self):
        """Reset tree and tracking state for a new question."""
        # Re-initialize prompt nodes from seeds
        self.prompt_nodes = [PromptNode(self, prompt) for prompt in 
                            [n.prompt for n in self.initial_prompts_nodes]]
        for i, pn in enumerate(self.prompt_nodes):
            pn.index = i
        
        # Reset transition tracking
        self.transition_counts.clear()
        self.operator_counts.clear()
        self.state_counts.clear()
        self.operator_utilities.clear()
        self.transition_rewards.clear()
        self.output_candidates = []
        
        # Reset query counters
        self.current_query = 0
        self.current_jailbreak = 0
        self.current_reject = 0
        self.current_iteration = 0
    
    def _initialize_seed_nodes(self, question: str):
        """Step 1-2 of Algorithm 1: Evaluate each seed template K times and initialize nodes."""
        for seed_node in self.prompt_nodes:
            if self.current_query + self.K > (self.max_query if self.max_query != -1 else 300):
                break
            
            responses = []
            for _ in range(self.K):
                message = synthesis_message(question, seed_node.prompt)
                if message is None:
                    responses.append("")
                    continue
                response = self.target.generate(message)
                responses.append(response[0] if isinstance(response, list) else response)
            
            state_labels = self.predictor.predict(responses)
            seed_node.set_state_from_labels(state_labels)
            seed_node.response = responses
            
            # Initialize tree stats
            seed_node.N = 0
            seed_node.W = 0.0
            
            self.current_query += self.K
            
            if seed_node.defense_state == 3:
                self.output_candidates.append(seed_node)
            
            self.writter.writerow([
                question, seed_node.prompt, str(responses)[:200],
                seed_node.parent.index if seed_node.parent else None,
                state_labels, seed_node.defense_state,
                seed_node.boundary_potential, seed_node.uncertainty
            ])
    
    def _get_expandable_nodes(self) -> 'list[PromptNode]':
        """Construct expandable candidate set C_t (Eq 15-16).
        
        C_t = {v in V_t : s_v < 3, d_v < H, |Ch(v)| = 0 OR chi_t(v) = 1}
        
        where chi_t(v) = 1[|Ch(v)| > 0, N_t(v) >= n_min, max_{u in Ch(v)} W̄_t(u) <= W̄_t(v) + delta_die]
        """
        expandable = []
        for node in self.prompt_nodes:
            # Skip full acceptance nodes
            if node.defense_state == 3:
                continue
            # Enforce depth budget
            if node.level >= self.H:
                continue
            # Leaf node or qualified intermediate node
            if len(node.child) == 0:
                expandable.append(node)
            elif node.N >= self.n_min:
                # Dynamic intermediate expansion check
                best_child_mean = max(
                    (c.W / (c.N + 1)) for c in node.child
                ) if node.child else 0.0
                parent_mean = node.W / (node.N + 1)
                if best_child_mean <= parent_mean + self.delta_die:
                    expandable.append(node)
        return expandable
    
    def _select_parent_node(self, expandable: 'list[PromptNode]') -> 'PromptNode':
        """Select parent node v_t using stability-controlled score (Eq 17).
        
        v_t = argmax_{v in C_t} [W̄_t(v) + c_v * sqrt(log(|V_t|+1) / (N_t(v)+1)) - gamma * Ũ_v]
        """
        best_node = None
        best_score = float('-inf')
        log_V = math.log(len(self.prompt_nodes) + 1)
        
        for node in expandable:
            W_bar = node.W / (node.N + 1)  # Eq 14
            exploration = self.c_v * math.sqrt(log_V / (node.N + 1))
            uncertainty_penalty = self.gamma_uncertainty * node.uncertainty
            score = W_bar + exploration - uncertainty_penalty
            
            if score > best_score:
                best_score = score
                best_node = node
        
        if best_node is None:
            # Fallback: pick a random node
            import random
            best_node = random.choice(self.prompt_nodes) if self.prompt_nodes else None
        
        best_node.visited_num += 1
        return best_node
    
    def _select_mutation_operator(self, parent_node: 'PromptNode') -> str:
        """Select mutation operator a_t using state-conditioned UCB (Eq 18-19).
        
        a_t = argmax_{a in A} [G_t(s_vt, a) + c_a * sqrt(log(N_state(s_vt)+1) / (N_op(s_vt, a)+1))]
        """
        s = parent_node.defense_state
        operator_names = self.mutate_policy.get_operator_names()
        
        if not operator_names:
            # Fallback
            import random
            return random.choice(['Rephrase', 'Shorten', 'Polish', 'Expand', 'CrossOver', 'ChangeStyle', 'GenerateSimilar'])
        
        best_op = None
        best_score = float('-inf')
        N_state = self.state_counts.get(s, 0)
        log_state = math.log(N_state + 1)
        
        for op_name in operator_names:
            G = self.operator_utilities[s].get(op_name, 0.0)
            N_op = self.operator_counts[s].get(op_name, 0)
            exploration = self.c_a * math.sqrt(log_state / (N_op + 1))
            score = G + exploration
            
            if score > best_score:
                best_score = score
                best_op = op_name
        
        return best_op if best_op else operator_names[0]
    
    def _apply_operator(self, operator_name: str, parent_prompt: str) -> str:
        """Apply the named mutation operator to generate a child template."""
        try:
            mutator = None
            if hasattr(self.mutate_policy, 'get_mutator_by_name'):
                mutator = self.mutate_policy.get_mutator_by_name(operator_name)
            if mutator is None:
                logging.warning(f"Operator {operator_name} is not configured in the mutate policy.")
                return None
            mutator.fuzzer = self
            result = mutator.mutate_single(parent_prompt)
            if isinstance(result, list) and len(result) > 0:
                return result[0]
            return result
        except Exception as e:
            logging.warning(f"Operator {operator_name} failed: {e}")
            return None
    
    def _evaluate_template(self, prompt: str, question: str, parent: 'PromptNode', operator_name: str) -> 'PromptNode':
        """Evaluate a child template K times and create a PromptNode."""
        responses = []
        for _ in range(self.K):
            message = synthesis_message(question, prompt)
            if message is None:
                responses.append("")
                continue
            response = self.target.generate(message)
            responses.append(response[0] if isinstance(response, list) else response)
        
        state_labels = self.predictor.predict(responses)
        
        child_node = PromptNode(self, prompt, parent=parent)
        child_node.set_state_from_labels(state_labels)
        child_node.response = responses
        child_node.index = len(self.prompt_nodes)
        self.prompt_nodes.append(child_node)
        
        # Set mutator name for logging
        child_node.mutator_name = operator_name
        
        # Write to result CSV
        question_text = self.questions[0] if len(self.questions) == 1 else str(self.questions)
        self.writter.writerow([
            question_text, prompt, str(responses)[:200],
            parent.index, state_labels,
            child_node.defense_state, child_node.boundary_potential, child_node.uncertainty
        ])
        
        return child_node
    
    def _backup_value(self, child_node: 'PromptNode', r_tilde: float):
        """Backpropagate adjusted reward to child and ancestors (Eq 22).
        
        For node v in ancestor chain of u including u:
            W(v) <- W(v) + r_tilde
            N(v) <- N(v) + 1
        """
        current = child_node
        while current is not None:
            current.W += r_tilde
            current.N += 1
            current = current.parent
    
    def _update_transition_stats(self, parent_node: 'PromptNode', operator_name: str,
                                  child_node: 'PromptNode', delta_phi: float):
        """Update transition counts and operator utilities (Eq 8-11)."""
        s = parent_node.defense_state
        a = operator_name
        s_prime = child_node.defense_state
        
        # Increment transition count C(s, a, s')
        self.transition_counts[s][a][s_prime] = self.transition_counts[s][a].get(s_prime, 0) + 1
        
        # Increment operator count N_op(s, a)
        self.operator_counts[s][a] = self.operator_counts[s].get(a, 0) + 1
        
        # Increment state count N_state(s)
        self.state_counts[s] = self.state_counts.get(s, 0) + 1
        
        # Accumulate transition reward for g_bar computation
        reward_with_penalty = delta_phi - self.lambda_uncertainty * child_node.uncertainty
        self.transition_rewards[s][a][s_prime] = self.transition_rewards[s][a].get(s_prime, 0.0) + reward_with_penalty
        
        # Recompute operator utilities G(s, a) for all operators under state s
        self._recompute_operator_utilities(s)
    
    def _recompute_operator_utilities(self, s: int):
        """Recompute state-conditioned operator utilities G_t(s, a) (Eq 9-11).
        
        K̂_t(s'|s,a) = (C_t(s,a,s') + beta) / (N_op_t(s,a) + |S|*beta)
        ḡ_t(s,a,s') = sum of delta_phi rewards / (C_t(s,a,s') + epsilon)
        G_t(s,a) = sum_{s'} K̂_t(s'|s,a) * ḡ_t(s,a,s')
        """
        NUM_STATES = 4
        EPSILON = 1e-6
        
        for a in self.operator_counts[s]:
            C_sa = self.operator_counts[s][a]  # N_op(s,a)
            if C_sa == 0:
                continue
            
            G = 0.0
            for s_prime in range(NUM_STATES):
                C_sas = self.transition_counts[s][a].get(s_prime, 0)
                # Empirical transition kernel (Eq 9)
                kernel = (C_sas + self.beta_smooth) / (C_sa + NUM_STATES * self.beta_smooth)
                
                # Average transition reward (Eq 10)
                total_reward = self.transition_rewards[s][a].get(s_prime, 0.0)
                g_bar = total_reward / (C_sas + EPSILON)
                
                G += kernel * g_bar
            
            self.operator_utilities[s][a] = G
    
    def _select_best_template(self) -> 'PromptNode':
        """Select best template T* by maximizing Phi - lambda_out * U over output candidates.
        
        If no state-3 candidates, returns the highest-potential node.
        """
        valid_nodes = [n for n in self.prompt_nodes if n.results is not None]
        if not valid_nodes:
            return None
        candidates = [n for n in valid_nodes if n.defense_state == 3]
        if not candidates:
            candidates = valid_nodes
        
        best_node = None
        best_score = float('-inf')
        for node in candidates:
            score = node.boundary_potential - self.lambda_uncertainty * node.uncertainty
            if score > best_score:
                best_score = score
                best_node = node
        
        return best_node

    def log(self):
        logging.info(
            f"Iteration {self.current_iteration}: {self.current_jailbreak} jailbreaks, "
            f"{self.current_reject} rejects, {self.current_query} queries")

    def write_prompt_nodes_to_csv(self):
        import csv
        import time
        
        nodes_file = f'prompt_nodes_details-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.csv'
        
        with open(nodes_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'index', 
                'prompt', 
                'response', 
                'results', 
                'visited_num',
                'reward',
                'parent_index', 
                'mutator_type',
                'mutator_name',
                'child_count',
                'level',
                'defense_state',
                'boundary_potential',
                'uncertainty',
                'W',
                'N',
            ])
            
            sorted_nodes = sorted(self.prompt_nodes, key=lambda x: x.index if x.index is not None else -1)
            
            for node in sorted_nodes:
                parent_index = node.parent.index if node.parent is not None and node.parent.index is not None else None
                mutator_type = type(node.mutator).__name__ if node.mutator is not None else None
                mutator_name = getattr(node, 'mutator_name', None)
                
                writer.writerow([
                    node.index,
                    node.prompt,
                    node.response,
                    node.results,
                    node.visited_num,
                    node.reward,
                    parent_index,
                    mutator_type,
                    mutator_name,
                    len(node.child),
                    node.level,
                    node.defense_state,
                    node.boundary_potential,
                    node.uncertainty,
                    node.W,
                    node.N,
                ])
        
        logging.info(f"Prompt nodes details written to {nodes_file}")
