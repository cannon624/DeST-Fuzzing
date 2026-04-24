import logging
import time
import csv

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mutator import Mutator, MutatePolicy
    from .selection import SelectPolicy

from gptfuzzer.llm import LLM, LocalLLM, OpenAILLM
from gptfuzzer.utils.template import synthesis_message
from gptfuzzer.utils.predict import Predictor
import warnings


class PromptNode:
    def __init__(self,
                 fuzzer: 'GPTFuzzer',
                 prompt: str,
                 response: str = None,
                 results: 'list[int]' = None,
                 parent: 'PromptNode' = None,
                 mutator: 'Mutator' = None):
        self.fuzzer: 'GPTFuzzer' = fuzzer
        self.prompt: str = prompt
        self.response: str = response
        self.results: 'list[int]' = results
        self.visited_num = 0
        self.reward = 0.0 

        self.parent: 'PromptNode' = parent
        self.mutator: 'Mutator' = mutator
        self.child: 'list[PromptNode]' = []
        self.level: int = 0 if parent is None else parent.level + 1

        self._index: int = None

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
        return sum(1 for result in self.results if result in [2, 3])

    @property
    def num_reject(self):
        return sum(1 for result in self.results if result in [0, 1])


    @property
    def num_query(self):
        return len(self.results)


class GPTFuzzer:
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
                 openai_model = OpenAILLM('gpt-3.5-turbo', '')
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
            ['question', 'prompt', 'response', 'parent', 'results'])

        self.generate_in_batch = False
        if len(self.questions) > 0 and generate_in_batch is True:
            self.generate_in_batch = True
            if isinstance(self.target, LocalLLM):
                warnings.warn("IMPORTANT! Hugging face inference with batch generation has the problem of consistency due to pad tokens. We do not suggest doing so and you may experience (1) degraded output quality due to long padding tokens, (2) inconsistent responses due to different number of padding tokens during reproduction. You should turn off generate_in_batch or use vllm batch inference.")
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


    
    def run(self):
        logging.info("Fuzzing started with question iteration!")
        
        total_questions = len(self.questions)
        jailbroken_questions = 0

        initial_global_query = self.current_query
        initial_global_jailbreak = self.current_jailbreak
        initial_global_reject = self.current_reject
        initial_global_iteration = self.current_iteration
        
        try:
            for q_idx, question in enumerate(self.questions):
                logging.info(f"Processing question {q_idx+1}/{total_questions}: {question}")
                
                question_query = 0
                question_jailbreak = 0
                question_iteration = 0
                max_question_iterations = self.max_iteration if self.max_iteration != -1 else 1000
                max_question_queries = 100
                
                original_questions = self.questions
                self.questions = [question]
                
                found_successful_result = False
                
                while (not found_successful_result and 
                       question_query < max_question_queries and 
                       question_iteration < max_question_iterations):
                    
                    seed = self.select_policy.select()
                    mutated_results = self.mutate_policy.mutate_single(seed)
                    self.evaluate(mutated_results,flag=False)
                    
                    for result in mutated_results:
                        question_query += result.num_query
                        question_jailbreak += result.num_jailbreak

                        if result.num_jailbreak > 0: 
                            found_successful_result = True
                    
                    self.update(mutated_results)
                    
                    question_iteration += 1

                    self.current_query = initial_global_query + question_query
                    self.current_jailbreak = initial_global_jailbreak + question_jailbreak
                    self.current_reject = initial_global_reject + (question_query - question_jailbreak)
                    self.current_iteration = initial_global_iteration + question_iteration
                    
                    self.log()
                    
                    if self.is_stop():
                        break
                
                initial_global_query = self.current_query
                initial_global_jailbreak = self.current_jailbreak
                initial_global_reject = self.current_reject
                initial_global_iteration = self.current_iteration

                self.questions = original_questions
                
                if question_jailbreak > 0:
                    jailbroken_questions += 1
                    logging.info(f"Jailbreak found for question {q_idx+1}")
                else:
                    logging.info(f"No jailbreak found for question {q_idx+1} within limits")
            
            self.write_prompt_nodes_to_csv()

            logging.info(f"Question iteration completed. Jailbroken {jailbroken_questions}/{total_questions} questions.")
            
        except KeyboardInterrupt:
            logging.info(f"Fuzzing interrupted by user!")
        finally:
            logging.info("Fuzzing finished!")
            self.raw_fp.close()
    
    """def run(self):
        logging.info("Fuzzing started!")
        try:
            for q_idx, question in enumerate(self.questions):
                logging.info(f"Testing question {q_idx}: {question}")
                
                for idx, seed in enumerate(self.initial_prompts_nodes):
                    if self.max_query != -1 and self.current_query >= self.max_query:
                        break
                        
                    prompt_node = PromptNode(self, seed.prompt)
                    self.evaluate_single(prompt_node, question)
                    self.current_query += 1

                    self.writter.writerow([question, prompt_node.prompt,
                                           prompt_node.response, seed.index, prompt_node.results])
                    
                    if prompt_node.num_jailbreak > 0:
                        self.current_jailbreak += 1
                        logging.info(f"Jailbreak found for question {q_idx} with template {idx}")
                    else:
                        self.current_reject += 1
                        logging.info(f"Template {idx} failed for question {q_idx}")
                        
                    self.log()
                
        except KeyboardInterrupt:
            logging.info("Fuzzing interrupted by user!")

        logging.info("Fuzzing finished!")
        self.raw_fp.close()"""

    def evaluate_single(self, prompt_node: 'PromptNode', question: str):
        message = synthesis_message(question, prompt_node.prompt)
        if message is None:  
            prompt_node.response = []
            prompt_node.results = []
            return
            
        if not self.generate_in_batch:
            response = self.target.generate(message)
            response = response[0] if isinstance(response, list) else response
        else:
            response = self.target.generate(message)
            response = response[0] if isinstance(response, list) else response
            
        prompt_node.response = response
        prompt_node.results = self.predictor.predict([response])  

    def evaluate(self, prompt_nodes: 'list[PromptNode]',flag=False):
        for prompt_node in prompt_nodes:
            if prompt_node.fuzzer is None:
                prompt_node.fuzzer = self
                
            responses = []
            messages = []
            for question in self.questions:
                message = synthesis_message(question, prompt_node.prompt)
                print(message)
                if message is None: 
                    prompt_node.response = []
                    prompt_node.results = []
                    break
                if not self.generate_in_batch:
                    response = self.target.generate(message)
                    responses.append(response[0] if isinstance(
                        response, list) else response)
                else:
                    messages.append(message)
            else:
                if self.generate_in_batch:
                    responses = self.target.generate_batch(messages)
                print(f"Model responses: {responses}")
                prompt_node.response = responses
                prompt_node.results = self.predictor.predict(responses)
                
            result = prompt_node.results[0] if prompt_node.results else -1
            mutate_policy = self.choose_mutation_strategy(result)

            if mutate_policy is not None and not flag:
                logging.info(f"Initiating extra mutations for result {result}.")
                for attempt in range(2):  
                    mutated_results = mutate_policy.mutate_single(prompt_node)
                    logging.info(f"Extra mutation attempt {attempt+1} completed.")
                    self.evaluate(mutated_results,flag=True)
                    self.update(mutated_results)

    def update(self, prompt_nodes: 'list[PromptNode]'):
        self.current_iteration += 1

        for prompt_node in prompt_nodes:
            if prompt_node.num_jailbreak > 0:
                prompt_node.index = len(self.prompt_nodes)
                self.prompt_nodes.append(prompt_node)
                question_text = self.questions[0] if len(self.questions) == 1 else str(self.questions)
                self.writter.writerow([question_text, prompt_node.prompt,
                                       prompt_node.response, prompt_node.parent.index, prompt_node.results])

            self.current_jailbreak += prompt_node.num_jailbreak
            self.current_query += prompt_node.num_query
            self.current_reject += prompt_node.num_reject

        self.select_policy.update(prompt_nodes)

    def choose_mutation_strategy(self, result: int):
        from gptfuzzer.fuzzer.mutator import (
            MutateRandomSinglePolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand,OpenAIMutatorChangeStyle,
            OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten)
            
        if result == 0:
            mutators = [
                OpenAIMutatorCrossOver(self.openai_model, temperature=0.7),
                OpenAIMutatorGenerateSimilar(self.openai_model, temperature=0.7),
                OpenAIMutatorChangeStyle(self.openai_model, temperature=0.7)
            ]
            for mutator in mutators:
                mutator.fuzzer = self
                
            policy = MutateRandomSinglePolicy(mutators, concatentate=True)
            policy.fuzzer = self
            return policy
        elif result == 1:
            mutators = [
                OpenAIMutatorExpand(self.openai_model, temperature=0.7),
                OpenAIMutatorShorten(self.openai_model, temperature=0.7),
                OpenAIMutatorRephrase(self.openai_model, temperature=0.7),
            ]
            for mutator in mutators:
                mutator.fuzzer = self
            policy = MutateRandomSinglePolicy(mutators, concatentate=True)
            policy.fuzzer = self
            return policy
        else:
            return None

    def log(self):
        logging.info(
            f"Iteration {self.current_iteration}: {self.current_jailbreak} jailbreaks, {self.current_reject} rejects, {self.current_query} queries")

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
                'child_count',
                'level'
            ])
            
            sorted_nodes = sorted(self.prompt_nodes, key=lambda x: x.index if x.index is not None else -1)
            
            for node in sorted_nodes:
                parent_index = node.parent.index if node.parent is not None and node.parent.index is not None else None
                mutator_type = type(node.mutator).__name__ if node.mutator is not None else None
                
                writer.writerow([
                    node.index,
                    node.prompt,
                    node.response,
                    node.results,
                    node.visited_num,
                    node.reward,
                    parent_index,
                    mutator_type,
                    len(node.child),
                    node.level
                ])
        
        logging.info(f"Prompt nodes details written to {nodes_file}")