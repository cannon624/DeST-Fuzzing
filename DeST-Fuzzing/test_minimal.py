"""DeST-Fuzzing Test - Using real malicious questions from dataset"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='[%H:%M:%S]')

import pandas as pd
from destfuzzing.fuzzer.selection import EnhancedMCTSSelectPolicy
from destfuzzing.fuzzer.mutator import (
    OpenAIMutatorCrossOver, OpenAIMutatorExpand, OpenAIMutatorChangeStyle,
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten,
    OpenAIMutatorPolish, RewardBasedMutatePolicy)
from destfuzzing.fuzzer import destfuzzing
from destfuzzing.llm import OpenAILLM, APILLM
from destfuzzing.utils.predict import GPTJudgePredictor, RoBERTaPredictor

# ---- Read API config ----
with open('Info-for-test.txt', 'r', encoding='utf-8') as f:
    config = {}
    for line in f:
        line = line.strip()
        for sep in [':', '\uff1a']:
            if sep in line:
                k, v = line.split(sep, 1)
                config[k.strip()] = v.strip()
                break

API_KEY = config.get('API', '')
MODEL = config.get('model-name', 'deepseek-v4-flash')
JUDGE_MODEL_PATH = config.get('judge-model-path', '')

# ---- Load questions from dataset (select 3) ----
all_questions = pd.read_csv('datasets/questions/question_list.csv')['text'].tolist()
QUESTIONS = all_questions[:3]

# ---- Load seed templates (select 3) ----
all_seeds = pd.read_csv('datasets/prompts/destfuzzing.csv')['text'].tolist()
SEEDS = all_seeds[:3]

print('=' * 60)
print('DeST-Fuzzing Test - Malicious Questions from Dataset')
print(f'  Target model:  {MODEL}')
judge_type = 'RoBERTa' if JUDGE_MODEL_PATH else 'GPT'
print(f'  Judge:         {judge_type}')
print(f'  Questions:     {len(QUESTIONS)}')
for i, q in enumerate(QUESTIONS):
    print(f'    Q{i}: {q[:80]}...')
print(f'  Seeds:         {len(SEEDS)}')
print(f'  K (samples):   1')
print(f'  Query budget:  15 (per question)')
print('=' * 60)

# ---- Initialize models ----
mutator_llm = OpenAILLM(MODEL, API_KEY)
target_llm  = APILLM(MODEL, API_KEY)
if JUDGE_MODEL_PATH:
    judge = RoBERTaPredictor(JUDGE_MODEL_PATH)
else:
    judge = GPTJudgePredictor(api_key=API_KEY, model_path=MODEL)

conservative = [
    OpenAIMutatorRephrase(mutator_llm, temperature=0.5),
    OpenAIMutatorShorten(mutator_llm, temperature=0.5),
    OpenAIMutatorPolish(mutator_llm, temperature=0.5),
]
aggressive = [
    OpenAIMutatorCrossOver(mutator_llm, temperature=0.8),
    OpenAIMutatorExpand(mutator_llm, temperature=0.8),
    OpenAIMutatorChangeStyle(mutator_llm, temperature=0.8),
    OpenAIMutatorGenerateSimilar(mutator_llm, temperature=0.8),
]

fuzzer = destfuzzing(
    questions=QUESTIONS,
    target=target_llm,
    predictor=judge,
    initial_seed=SEEDS,
    mutate_policy=RewardBasedMutatePolicy(
        conservative_mutators=conservative,
        aggressive_mutators=aggressive,
        reward_threshold=1.0,
    ),
    select_policy=EnhancedMCTSSelectPolicy(),
    max_query=15,
    energy=1,
    openai_model=mutator_llm,
    K=1,
    H=5,
    lambda_uncertainty=0.3,
    alpha_gate=2.0,
    c_v=1.0,
    c_a=1.0,
    gamma_uncertainty=0.3,
)

print('\nRunning DeST-Fuzzing...\n')
fuzzer.run()
print('\nTest completed! Check results-*.csv and prompt_nodes_details-*.csv')
