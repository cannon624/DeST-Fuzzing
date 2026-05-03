"""
DeST-Fuzzing Full Experiment Suite
====================================
100 questions x 80 seed templates, all 5 experiments from the paper.
Results saved to ./full_experiment_results/
"""
import sys, os, json, time, random
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(message)s', datefmt='[%H:%M:%S]')

import pandas as pd
import numpy as np

from destfuzzing.fuzzer.selection import EnhancedMCTSSelectPolicy
from destfuzzing.fuzzer.mutator import (
    OpenAIMutatorCrossOver, OpenAIMutatorExpand, OpenAIMutatorChangeStyle,
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten,
    OpenAIMutatorPolish, RewardBasedMutatePolicy, ALL_OPERATOR_NAMES,
    RADICAL_EXPLORATION_OPERATORS, CONSERVATIVE_INFILTRATION_OPERATORS)
from destfuzzing.fuzzer import destfuzzing
from destfuzzing.llm import OpenAILLM, APILLM
from destfuzzing.utils.predict import GPTJudgePredictor, RoBERTaPredictor
from destfuzzing.utils.template import synthesis_message

# ─── Config ───────────────────────────────────────────────────
with open('Info-for-test.txt', 'r', encoding='utf-8') as f:
    cfg = {}
    for line in f:
        line = line.strip()
        for sep in [':', '\uff1a']:
            if sep in line:
                k, v = line.split(sep, 1)
                cfg[k.strip()] = v.strip()
                break

API_KEY = cfg.get('API', '')
MODEL = cfg.get('model-name', 'deepseek-v4-flash')
JUDGE_MODEL_PATH = cfg.get('judge-model-path', '')
OUT = f'full_experiment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(OUT, exist_ok=True)

ALL_Q = pd.read_csv('datasets/questions/question_list.csv')['text'].tolist()
ALL_S = pd.read_csv('datasets/prompts/destfuzzing.csv')['text'].tolist()

NQ = len(ALL_Q)  # 100
NS = len(ALL_S)  # 80

# Choose defense-state estimator: RoBERTa if judge-model-path is set, otherwise GPT-based
if JUDGE_MODEL_PATH:
    judge = RoBERTaPredictor(JUDGE_MODEL_PATH)
    judge_type = 'RoBERTa'
else:
    judge = GPTJudgePredictor(api_key=API_KEY, model_path=MODEL)
    judge_type = 'GPT'

print(f'{"="*70}')
print(f'DeST-Fuzzing FULL Experiment Suite')
print(f'  Model:     {MODEL}')
print(f'  Judge:     {judge_type}')
print(f'  Questions: {NQ}')
print(f'  Seeds:     {NS}')
print(f'  Output:    {OUT}/')
print(f'{"="*70}')

mutator = OpenAILLM(MODEL, API_KEY)
target  = APILLM(MODEL, API_KEY)

cons = [OpenAIMutatorRephrase(mutator,temperature=0.5),
        OpenAIMutatorShorten(mutator,temperature=0.5),
        OpenAIMutatorPolish(mutator,temperature=0.5)]
aggr = [OpenAIMutatorCrossOver(mutator,temperature=0.8),
        OpenAIMutatorExpand(mutator,temperature=0.8),
        OpenAIMutatorChangeStyle(mutator,temperature=0.8),
        OpenAIMutatorGenerateSimilar(mutator,temperature=0.8)]

def mkfuzzer(qs, ss, budget, K=1, **kw):
    return destfuzzing(
        questions=qs, target=target, predictor=judge, initial_seed=ss,
        mutate_policy=RewardBasedMutatePolicy(conservative_mutators=cons,
            aggressive_mutators=aggr, reward_threshold=1.0),
        select_policy=EnhancedMCTSSelectPolicy(),
        max_query=budget, energy=1, openai_model=mutator,
        K=K, H=kw.get('H',10),
        lambda_uncertainty=kw.get('lam',0.3),
        alpha_gate=kw.get('alpha',2.0), c_v=1.0, c_a=1.0,
        gamma_uncertainty=kw.get('gamma',0.3),
        beta_smooth=kw.get('beta',0.1),
        n_min=kw.get('n_min',3), delta_die=kw.get('delta',0.05))


# ═══════════════════════════════════════════════════════════════
# Experiment 1: Static Seed Effectiveness (Table 2)
# ═══════════════════════════════════════════════════════════════
def exp1_static():
    print(f'\n{"─"*70}\nExp 1: Static Seed Effectiveness (Table 2)\n{"─"*70}')
    
    records = []
    for qi, q in enumerate(ALL_Q):
        q_succ = 0
        for si, s in enumerate(ALL_S):
            msg = synthesis_message(q, s)
            if not msg: continue
            resp = target.generate(msg, temperature=0, max_tokens=256, n=1)
            txt = resp[0] if isinstance(resp,list) else resp
            lb = judge.predict([txt])[0]
            records.append({'qi':qi, 'si':si, 'label':lb,
                          'full_accept':1 if lb==3 else 0,
                          'partial_accept':1 if lb>=2 else 0})
            if lb==3: q_succ+=1
        if (qi+1) % 20 == 0:
            print(f'  Progress: {qi+1}/{NQ} questions done')
    
    df = pd.DataFrame(records)
    df.to_csv(f'{OUT}/exp1_static_full.csv', index=False)
    
    # JQN (Jailbroken Query Number) - Eq 23
    jqn = len(set(r['qi'] for r in records if r['full_accept']))
    
    # Top-1 ASR: best single template's success rate
    template_asr = {}
    for si in range(NS):
        succ = df[(df.si==si)]['full_accept'].sum()
        template_asr[si] = succ / NQ
    top1_asr = max(template_asr.values()) if template_asr else 0
    
    # Top-5 ASR
    top5_templates = sorted(template_asr, key=template_asr.get, reverse=True)[:5]
    top5_jqn = len(set(r['qi'] for r in records if r['full_accept'] and r['si'] in top5_templates))
    top5_asr = top5_jqn / NQ
    
    # Avg successful templates per query
    per_q = df.groupby('qi')['full_accept'].sum()
    avg_succ = per_q.mean()
    
    # Invalid seeds (zero success over all queries)
    invalid = sum(1 for si in range(NS) 
                  if df[df.si==si]['full_accept'].sum() == 0)
    
    # Per-seed success counts
    seed_success = {si: int(df[df.si==si]['full_accept'].sum()) for si in range(NS)}
    
    metrics = {
        'JQN': f'{jqn}/{NQ}',
        'Top1_ASR': f'{top1_asr:.1%}',
        'Top5_ASR': f'{top5_asr:.1%}',
        'Avg_Succ_per_Q': round(avg_succ, 2),
        'Invalid_Seeds': invalid,
        'Total_Seeds': NS,
        'Total_Questions': NQ,
        'Per_Seed_Success': seed_success,
        'Per_Question_Success': {int(k): int(v) for k, v in per_q.to_dict().items()},
    }
    with open(f'{OUT}/exp1_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f'  JQN={jqn}/{NQ}  Top1={top1_asr:.1%}  Top5={top5_asr:.1%}  '
          f'Avg={avg_succ:.2f}  Invalid={invalid}')
    return metrics


# ═══════════════════════════════════════════════════════════════
# Experiment 2: Full DeST-Fuzzing Search (Table 3)
# ═══════════════════════════════════════════════════════════════
def exp2_full_search():
    print(f'\n{"─"*70}\nExp 2: Full DeST-Fuzzing Search (Table 3)\n{"─"*70}')
    
    # Identify effective vs ineffective seeds from Exp 1
    df = pd.read_csv(f'{OUT}/exp1_static_full.csv')
    seed_succ = df.groupby('si')['full_accept'].sum()
    valid_seeds = [si for si in range(NS) if seed_succ[si] > 0]
    invalid_seeds = [si for si in range(NS) if seed_succ[si] == 0]
    
    # Hard subset: questions unsolved by static seeds
    per_q_succ = df.groupby('qi')['full_accept'].sum()
    hard_qs = [qi for qi in range(NQ) if per_q_succ[qi] == 0]
    
    print(f'  Valid seeds: {len(valid_seeds)}, Invalid seeds: {len(invalid_seeds)}')
    print(f'  Hard questions (unsolved by static): {len(hard_qs)}')
    
    # Run on hard subset with all seeds
    hard_questions = [ALL_Q[qi] for qi in hard_qs[:20]]  # limit for time
    all_seed_texts = [ALL_S[si] for si in range(min(NS, 10))]  # limit seeds
    
    if hard_questions and all_seed_texts:
        print(f'  Running DeST-Fuzzing on {len(hard_questions)} hard questions...')
        fz = mkfuzzer(hard_questions, all_seed_texts, budget=min(200, len(hard_questions)*10), K=1)
        fz.run()
        
        jb = sum(1 for n in fz.prompt_nodes if n.defense_state==3)
        metrics = {
            'hard_questions': len(hard_questions),
            'jailbreaks': jb,
            'queries': fz.current_query,
            'nodes': len(fz.prompt_nodes),
            'state_dist': {i: sum(1 for n in fz.prompt_nodes if n.defense_state==i) for i in range(4)},
        }
    else:
        metrics = {'note': 'No hard questions or seeds available'}
    
    with open(f'{OUT}/exp2_search.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print(f'  Search metrics: {json.dumps(metrics, default=str)[:200]}')
    return metrics


# ═══════════════════════════════════════════════════════════════
# Experiment 3: Component Ablation (Table 4/5)
# ═══════════════════════════════════════════════════════════════
class AbFuzzer(destfuzzing):
    def __init__(self, mode='full', *a, **kw):
        self.mode = mode; super().__init__(*a, **kw)
    def _select_mutation_operator(self, pn):
        if self.mode=='no_ta': import random; return random.choice(ALL_OPERATOR_NAMES)
        return super()._select_mutation_operator(pn)
    def _backup_value(self, cn, rt):
        if self.mode=='no_ug':
            dp = cn.boundary_potential - cn.parent.boundary_potential if cn.parent else 0
            r = dp - self.lambda_uncertainty*cn.uncertainty
            cur=cn
            while cur: cur.W+=r; cur.N+=1; cur=cur.parent
            return
        if self.mode=='no_bp':
            r = (1.0 if cn.defense_state==3 else 0.0) - self.lambda_uncertainty*cn.uncertainty
            cur=cn
            while cur: cur.W+=r; cur.N+=1; cur=cur.parent
            return
        super()._backup_value(cn, rt)

def exp3_ablation():
    print(f'\n{"─"*70}\nExp 3: Component Ablation (Table 4/5)\n{"─"*70}')
    
    # Use a subset of hard questions
    df = pd.read_csv(f'{OUT}/exp1_static_full.csv')
    per_q = df.groupby('qi')['full_accept'].sum()
    hard_qs = [qi for qi in range(NQ) if per_q[qi] == 0][:3]
    qs = [ALL_Q[qi] for qi in hard_qs] if hard_qs else ALL_Q[:3]
    ss = [ALL_S[si] for si in range(min(NS, 5))]
    
    variants = [
        ('full', 'Full DeST-Fuzzing'),
        ('no_bp', 'w/o Boundary Potential'),
        ('no_ta', 'w/o Transition-Aware'),
        ('no_ug', 'w/o Uncertainty Gate'),
    ]
    
    results = []
    for mode, label in variants:
        print(f'  [{mode}] {label}')
        if mode=='full':
            fz = mkfuzzer(qs, ss, budget=30, K=1)
        else:
            fz = AbFuzzer(mode=mode, questions=qs, target=target, predictor=judge,
                    initial_seed=ss, mutate_policy=RewardBasedMutatePolicy(
                        conservative_mutators=cons, aggressive_mutators=aggr, reward_threshold=1.0),
                    select_policy=EnhancedMCTSSelectPolicy(),
                    max_query=30, energy=1, openai_model=mutator, K=1, H=5,
                    lam=0.3, alpha=2.0)
        fz.run()
        jb = sum(1 for n in fz.prompt_nodes if n.defense_state==3)
        pot = float(np.mean([n.boundary_potential for n in fz.prompt_nodes if n.results]))
        unc = float(np.mean([n.uncertainty for n in fz.prompt_nodes if n.results]))
        results.append({'mode':mode, 'label':label, 'jailbreaks':jb,
                       'queries':fz.current_query, 'avg_phi':pot, 'avg_U':unc})
        print(f'    JB:{jb} Phi:{pot:.3f} U:{unc:.3f}')
    
    with open(f'{OUT}/exp3_ablation.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ═══════════════════════════════════════════════════════════════
# Experiment 4: Operator Transition Diagnostics (Figure 6)
# ═══════════════════════════════════════════════════════════════
def exp4_operator_diag():
    print(f'\n{"─"*70}\nExp 4: Operator Transition Diagnostics (Figure 6)\n{"─"*70}')
    
    qs = ALL_Q[:5]; ss = [ALL_S[si] for si in range(min(NS, 5))]
    fz = mkfuzzer(qs, ss, budget=40, K=1)
    fz.run()
    
    # Per-operator stats
    ops = {}
    for s in range(4):
        for a in ALL_OPERATOR_NAMES:
            tc = fz.transition_counts.get(s,{}).get(a,{})
            tot = sum(tc.values())
            if tot==0: continue
            pos = sum(tc.get(sp,0) for sp in range(s+1,4))
            reg = sum(tc.get(sp,0) for sp in range(0,s))
            s3  = tc.get(3,0)
            if a not in ops: ops[a] = {'tot':0,'pos':0,'reg':0,'s3':0}
            ops[a]['tot']+=tot; ops[a]['pos']+=pos; ops[a]['reg']+=reg; ops[a]['s3']+=s3
    
    rpt = {}
    for a, st in ops.items():
        t = st['tot']
        rpt[a] = {
            'n': t,
            'positive_rate': round(st['pos']/t, 3),
            'regression_rate': round(st['reg']/t, 3),
            'state3_rate': round(st['s3']/t, 3),
            'category': 'Radical' if a in RADICAL_EXPLORATION_OPERATORS else 'Conservative',
        }
    
    # By state and operator group (Table 7)
    state_group = {}
    for s in range(4):
        for gname, gops in [('Radical', RADICAL_EXPLORATION_OPERATORS),
                             ('Conservative', CONSERVATIVE_INFILTRATION_OPERATORS)]:
            ttot, tpos, treg = 0, 0, 0
            for a in gops:
                tc = fz.transition_counts.get(s,{}).get(a,{})
                ttot += sum(tc.values())
                tpos += sum(tc.get(sp,0) for sp in range(s+1,4))
                treg += sum(tc.get(sp,0) for sp in range(0,s))
            if ttot > 0:
                state_group[f'state_{s}_{gname}'] = {
                    'total': ttot,
                    'positive_rate': round(tpos/ttot, 3),
                    'regression_rate': round(treg/ttot, 3),
                }
    
    with open(f'{OUT}/exp4_operators.json', 'w') as f:
        json.dump({'per_operator': rpt, 'by_state_group': state_group}, f, indent=2)
    
    print('\n  Per-Operator Transition Rates:')
    for a in sorted(rpt.keys()):
        r = rpt[a]
        print(f'    {a:20s} [{r["category"]:12s}] +{r["positive_rate"]:.1%} '
              f'-{r["regression_rate"]:.1%} S3:{r["state3_rate"]:.1%} (n={r["n"]})')
    
    return rpt


# ═══════════════════════════════════════════════════════════════
# Experiment 5: Hyperparameter Sensitivity (Table 6)
# ═══════════════════════════════════════════════════════════════
def exp5_hyperparams():
    print(f'\n{"─"*70}\nExp 5: Hyperparameter Sensitivity (Table 6)\n{"─"*70}')
    
    qs = ALL_Q[:3]; ss = [ALL_S[si] for si in range(min(NS, 3))]
    results = {'K': [], 'lambda': [], 'alpha': [], 'budget': []}
    
    for K in [1, 2, 3]:
        print(f'  K={K}...')
        fz = mkfuzzer(qs, ss, budget=20, K=K)
        fz.run()
        jb = sum(1 for n in fz.prompt_nodes if n.defense_state==3)
        results['K'].append({'K':K, 'jailbreaks':jb, 'queries':fz.current_query})
    
    for lam in [0.1, 0.3, 0.5, 0.8]:
        print(f'  lambda={lam}...')
        fz = mkfuzzer(qs, ss, budget=20, K=1, lam=lam)
        fz.run()
        jb = sum(1 for n in fz.prompt_nodes if n.defense_state==3)
        avgU = float(np.mean([n.uncertainty for n in fz.prompt_nodes if n.results]))
        results['lambda'].append({'lambda':lam, 'jailbreaks':jb, 'avg_uncertainty':avgU})
    
    for alpha in [0.5, 1.0, 2.0, 3.0, 4.0]:
        print(f'  alpha={alpha}...')
        fz = mkfuzzer(qs, ss, budget=20, K=1, alpha=alpha)
        fz.run()
        jb = sum(1 for n in fz.prompt_nodes if n.defense_state==3)
        avgU = float(np.mean([n.uncertainty for n in fz.prompt_nodes if n.results]))
        results['alpha'].append({'alpha':alpha, 'jailbreaks':jb, 'avg_uncertainty':avgU})
    
    for B in [100, 200, 300]:
        print(f'  Budget={B}...')
        fz = mkfuzzer(qs, ss, budget=B, K=1)
        fz.run()
        jb = sum(1 for n in fz.prompt_nodes if n.defense_state==3)
        results['budget'].append({'budget':B, 'jailbreaks':jb, 'queries':fz.current_query})
    
    with open(f'{OUT}/exp5_hyperparams.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    all_r = {}
    start = time.time()
    
    for name, fn in [
        ('exp1_static', exp1_static),
        ('exp2_search', exp2_full_search),
        ('exp3_ablation', exp3_ablation),
        ('exp4_operators', exp4_operator_diag),
        ('exp5_hyperparams', exp5_hyperparams),
    ]:
        try:
            all_r[name] = fn()
        except Exception as e:
            print(f'\n  *** {name} FAILED: {e}')
            import traceback; traceback.print_exc()
            all_r[name] = {'error': str(e)}
    
    elapsed = time.time() - start
    all_r['_meta'] = {'model': MODEL, 'NQ': NQ, 'NS': NS,
                      'elapsed_seconds': round(elapsed, 0),
                      'output_dir': OUT}
    
    with open(f'{OUT}/all_results.json', 'w') as f:
        json.dump(all_r, f, indent=2, default=str)
    
    print(f'\n{"="*70}')
    print(f'All experiments completed!')
    print(f'  Time: {elapsed/60:.0f} min')
    print(f'  Results: {OUT}/')
    print(f'{"="*70}')
