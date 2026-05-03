"""
DeST-Fuzzing Experiment Suite (scaled for API testing)
========================================================
Results saved to ./experiment_results/
"""
import sys, os, json, time, math
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='[%H:%M:%S]')

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

# Config
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
OUT = f'experiment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
os.makedirs(OUT, exist_ok=True)

ALL_Q = pd.read_csv('datasets/questions/question_list.csv')['text'].tolist()
ALL_S = pd.read_csv('datasets/prompts/destfuzzing.csv')['text'].tolist()

NQ, NS = 9, 5  # questions, seeds for static test

# Choose defense-state estimator
if JUDGE_MODEL_PATH:
    judge = RoBERTaPredictor(JUDGE_MODEL_PATH)
    judge_type = 'RoBERTa'
else:
    judge = GPTJudgePredictor(api_key=API_KEY, model_path=MODEL)
    judge_type = 'GPT'

print(f'DeST-Fuzzing Experiment Suite\n  Model: {MODEL}\n  Judge: {judge_type}\n  Output: {OUT}/\n')

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
        K=K, H=kw.get('H',5), lambda_uncertainty=kw.get('lam',0.3),
        alpha_gate=kw.get('alpha',2.0), c_v=1.0, c_a=1.0,
        gamma_uncertainty=kw.get('gamma',0.3))


# ═══ Exp 1: Static Seed Effectiveness (Table 2) ═══
def exp1():
    print('─'*60 + '\nExp 1: Static Seed Effectiveness\n' + '─'*60)
    rows = []
    for qi, q in enumerate(ALL_Q[:NQ]):
        succ = 0
        for si, s in enumerate(ALL_S[:NS]):
            msg = synthesis_message(q, s)
            if not msg: continue
            r = target.generate(msg, temperature=0, max_tokens=256, n=1)
            txt = r[0] if isinstance(r,list) else r
            lb = judge.predict([txt])[0]
            rows.append({'qi':qi,'q':q[:50],'si':si,'label':lb,'s3':1 if lb==3 else 0})
            if lb==3: succ+=1
        print(f'  Q{qi}: {succ}/{NS} full-accept  [{q[:50]}...]')
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/exp1_static.csv', index=False)
    jqn = len(set(r['qi'] for r in rows if r['s3']))
    m = {'JQN':f'{jqn}/{NQ}','N_Q':NQ,'N_S':NS,
         'Avg_S3_per_Q':round(df.groupby('qi')['s3'].sum().mean(),2)}
    json.dump(m, open(f'{OUT}/exp1_metrics.json','w'), indent=2)
    print(f'  => JQN={jqn}/{NQ}')
    return m


# ═══ Exp 2: Full DeST-Fuzzing Search (Table 3/4) ═══
def exp2():
    print('─'*60 + '\nExp 2: Full DeST-Fuzzing Search\n' + '─'*60)
    qs = ALL_Q[:3]; ss = ALL_S[:3]
    fz = mkfuzzer(qs, ss, budget=30, K=1)
    fz.run()
    jb = sum(1 for n in fz.prompt_nodes if n.defense_state==3)
    m = {'jailbreaks':jb,'queries':fz.current_query,'nodes':len(fz.prompt_nodes),
         'state_dist':{i:sum(1 for n in fz.prompt_nodes if n.defense_state==i) for i in range(4)}}
    json.dump(m, open(f'{OUT}/exp2_search.json','w'), indent=2, default=str)
    print(f'  => {jb} jailbreaks, {fz.current_query} queries')
    return m


# ═══ Exp 3: Component Ablation (Table 4/5) ═══
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

def exp3():
    print('─'*60 + '\nExp 3: Component Ablation\n' + '─'*60)
    qs=ALL_Q[:2]; ss=ALL_S[:2]
    res=[]
    for mode,label in [('full','Full'),('no_bp','w/o Boundary Potential'),
                        ('no_ta','w/o Transition-Aware'),('no_ug','w/o Uncertainty Gate')]:
        print(f'  [{mode}] {label}')
        if mode=='full': fz=mkfuzzer(qs,ss,budget=15,K=1)
        else: fz=AbFuzzer(mode=mode,questions=qs,target=target,predictor=judge,
                initial_seed=ss,mutate_policy=RewardBasedMutatePolicy(
                    conservative_mutators=cons,aggressive_mutators=aggr,reward_threshold=1.0),
                select_policy=EnhancedMCTSSelectPolicy(),
                max_query=15,energy=1,openai_model=mutator,K=1,H=5,lam=0.3,alpha=2.0)
        fz.run()
        jb=sum(1 for n in fz.prompt_nodes if n.defense_state==3)
        pot=float(np.mean([n.boundary_potential for n in fz.prompt_nodes if n.results]))
        unc=float(np.mean([n.uncertainty for n in fz.prompt_nodes if n.results]))
        res.append({'mode':mode,'label':label,'jailbreaks':jb,'queries':fz.current_query,
                    'avg_phi':pot,'avg_U':unc})
        print(f'    JB:{jb} Phi:{pot:.3f} U:{unc:.3f}')
    json.dump(res, open(f'{OUT}/exp3_ablation.json','w'), indent=2)
    return res


# ═══ Exp 4: Operator Diagnostics (Figure 6) ═══
def exp4():
    print('─'*60 + '\nExp 4: Operator Transition Diagnostics\n' + '─'*60)
    qs=ALL_Q[:3]; ss=ALL_S[:3]
    fz=mkfuzzer(qs,ss,budget=25,K=1)
    fz.run()
    ops={}
    for s in range(4):
        for a in ALL_OPERATOR_NAMES:
            tc=fz.transition_counts.get(s,{}).get(a,{})
            tot=sum(tc.values())
            if tot==0: continue
            pos=sum(tc.get(sp,0) for sp in range(s+1,4))
            reg=sum(tc.get(sp,0) for sp in range(0,s))
            s3=tc.get(3,0)
            if a not in ops: ops[a]={'tot':0,'pos':0,'reg':0,'s3':0}
            ops[a]['tot']+=tot; ops[a]['pos']+=pos; ops[a]['reg']+=reg; ops[a]['s3']+=s3
    rpt={}
    for a,st in ops.items():
        t=st['tot']
        rpt[a]={'n':t,'pos_r':round(st['pos']/t,3),'reg_r':round(st['reg']/t,3),
                's3_r':round(st['s3']/t,3),
                'cat':'Radical' if a in RADICAL_EXPLORATION_OPERATORS else 'Conservative'}
    json.dump(rpt, open(f'{OUT}/exp4_operators.json','w'), indent=2)
    for a in sorted(rpt.keys()):
        r=rpt[a]; print(f'  {a:20s} [{r["cat"]:12s}] +{r["pos_r"]:.1%} -{r["reg_r"]:.1%} S3:{r["s3_r"]:.1%} (n={r["n"]})')
    return rpt


# ═══ Exp 5: Hyperparameter Sensitivity (Table 6) ═══
def exp5():
    print('─'*60 + '\nExp 5: Hyperparameter Sensitivity\n' + '─'*60)
    qs=ALL_Q[:2]; ss=ALL_S[:2]
    r={'K':[],'lambda':[],'alpha':[]}
    for K in [1,2]:
        print(f'  K={K}...')
        fz=mkfuzzer(qs,ss,budget=12,K=K); fz.run()
        r['K'].append({'K':K,'jb':sum(1 for n in fz.prompt_nodes if n.defense_state==3)})
    for lam in [0.1,0.3,0.5]:
        print(f'  lambda={lam}...')
        fz=mkfuzzer(qs,ss,budget=12,K=1,lam=lam); fz.run()
        r['lambda'].append({'lambda':lam,'jb':sum(1 for n in fz.prompt_nodes if n.defense_state==3),
                           'avgU':float(np.mean([n.uncertainty for n in fz.prompt_nodes if n.results]))})
    for alpha in [1.0,2.0,3.0]:
        print(f'  alpha={alpha}...')
        fz=mkfuzzer(qs,ss,budget=12,K=1,alpha=alpha); fz.run()
        r['alpha'].append({'alpha':alpha,'jb':sum(1 for n in fz.prompt_nodes if n.defense_state==3),
                          'avgU':float(np.mean([n.uncertainty for n in fz.prompt_nodes if n.results]))})
    json.dump(r, open(f'{OUT}/exp5_hyperparams.json','w'), indent=2)
    return r


if __name__=='__main__':
    all_r={}
    for name,fn in [('exp1_static',exp1),('exp2_search',exp2),
                     ('exp3_ablation',exp3),('exp4_operators',exp4),('exp5_hyperparams',exp5)]:
        try:
            all_r[name]=fn()
        except Exception as e:
            print(f'  *** {name} FAILED: {e}')
            import traceback; traceback.print_exc()
            all_r[name]={'error':str(e)}
    json.dump(all_r, open(f'{OUT}/all_results.json','w'), indent=2, default=str)
    print(f'\n{"="*60}\nDone! Results: {OUT}/\n{"="*60}')
