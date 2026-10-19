"""Diagnostic-only confirmation controller; explicitly not an agent benchmark."""
import copy,json
from pathlib import Path
import torch
from sts2_env.entity.ppo import EntityAgent
from sts2_env.training.environment import make_env
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/training-audit-20261019'
torch.set_num_threads(2)
analysis=json.loads((OUT/'analysis.json').read_text());results=[]
for profile in ('shaped','sparse'):
    run=analysis['runs'][profile]
    agent,_=EntityAgent.load(OUT/'inputs'/run['directory']/'best_model/best_model.pt','cpu')
    for seed in (20019,20026):
        cfg=copy.deepcopy(run['config']);cfg['evaluation']['seeds']=[seed]
        env=make_env(cfg,evaluation=True);obs,_=env.reset();choice_steps=0;forced=[]
        try:
            for step in range(2000):
                mgr=env.unwrapped._mgr;mask=env.env.action_masks();a,_=agent.predict(obs,mask,True)
                combat=mgr.get_combat_state() if mgr.phase=='COMBAT' else None
                pending=combat.pending_choice if combat else mgr.run_state.pending_choice
                choice_steps=choice_steps+1 if pending else 0
                if pending and mask[0] and pending.can_confirm() and (len(pending.selected_indices)>=pending.max_choices or choice_steps>=8):
                    if a!=0:forced.append(dict(step=step+1,floor=mgr.run_state.total_floor,prompt=str(pending.prompt),original_action=a))
                    a=0;choice_steps=0
                obs,_,term,trunc,info=env.step(a)
                if term or trunc:
                    results.append(dict(profile=profile,seed=seed,steps=step+1,win=bool(info['is_success']),timeout=bool(trunc),floor=info['floor'],confirm_interventions=forced))
                    print(json.dumps(results[-1]),flush=True);break
        finally:env.close()
(OUT/'confirmation-interventions.json').write_text(json.dumps(results,indent=2)+'\n')
