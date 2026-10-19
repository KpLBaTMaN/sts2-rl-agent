"""CPU-only saved-policy probes; no parameter updates or live-game controls."""
from __future__ import annotations
import copy, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sts2_env.entity.ppo import EntityAgent,stack_observations,to_tensors
from sts2_env.entity.model import masked_logits
from sts2_env.training.environment import make_env
from sts2_env.watch.player import model_decision
from sts2_env.watch.entity_adapter import EntityWatchAdapter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/training-audit-20261019'
torch.set_num_threads(2)

def main():
    analysis=json.loads((OUT/'analysis.json').read_text());results={};allcamp=[]
    watch=OUT/'inputs/watch/20261019-201323-e501bc/decisions.jsonl'
    rest=next(json.loads(l)['state'] for l in watch.read_text().splitlines() if json.loads(l)['state']['type']=='rest_site')
    for profile,run in analysis['runs'].items():
        for ck in (['best_model/best_model.pt','final_model.pt'] if run['completed'] else ['best_model/best_model.pt']):
            directory=OUT/'inputs'/run['directory'];agent,saved=EntityAgent.load(directory/ck,'cpu')
            label=f"{profile}/{'best' if 'best' in ck else 'final'}"
            for fraction in (.25,.5,.75,1.0):
                state=copy.deepcopy(rest);hp=round(fraction*state['run_state']['max_hp']);state['run_state']['hp']=hp
                if 'combat_snapshot' in state:state['combat_snapshot']['player']['hp']=hp
                _,proposal=model_decision(agent,EntityWatchAdapter(),state)
                probs={p['label']:p['probability'] for p in proposal['probabilities']}
                allcamp.append(dict(model=label,steps=saved['num_timesteps'],hp_fraction=fraction,probabilities=probs,chosen=proposal['label']))
            if 'final' in label or profile=='survival':continue
            cfg=run['config'];episodes=[]
            for seed in (20019,20026):
                config=copy.deepcopy(cfg);config['evaluation']['seeds']=[seed]
                env=make_env(config,evaluation=True);obs,_=env.reset();seen=Counter();tail=[];phases=Counter();reason='limit'
                try:
                    for step in range(2000):
                        mgr=env.unwrapped._mgr;mask=env.env.action_masks();action,_=agent.predict(obs,mask,True)
                        phase=mgr.phase;phases[phase]+=1
                        key=hashlib.sha256(b''.join(v.tobytes() for v in obs.values())+bytes([action])).hexdigest()
                        seen[key]+=1
                        action_list=mgr.get_available_actions()
                        record=dict(step=step+1,floor=mgr.run_state.total_floor,phase=phase,action=action,
                            hp=mgr.run_state.player.current_hp,available=action_list[:10],
                            pending=str(type(mgr.run_state.pending_choice).__name__),
                            relics=list(mgr.run_state.player.relics))
                        tail=(tail+[record])[-8:]
                        if phase!='COMBAT' and seen[key]>=4:
                            logits,_=agent.model(to_tensors(stack_observations([obs]),agent.device))
                            probs=torch.softmax(masked_logits(logits,torch.as_tensor(mask[None],dtype=torch.bool)),-1)[0]
                            record['policy_top']=[dict(action=int(i),prob=float(probs[i])) for i in torch.topk(probs,min(5,int(mask.sum()))).indices]
                            reason='repeated_noncombat_observation_and_action';break
                        obs,_,term,trunc,info=env.step(action)
                        if term or trunc:reason='win' if info['is_success'] else ('timeout' if trunc else 'loss');break
                    episodes.append(dict(seed=seed,reason=reason,steps=step+1,phases=dict(phases),tail=tail))
                    print(label,seed,reason,step+1,tail[-1]['phase'],flush=True)
                finally:env.close()
            results[label]=dict(checkpoint_steps=saved['num_timesteps'],episodes=episodes)
    (OUT/'policy-probes.json').write_text(json.dumps(dict(campfires=allcamp,replays=results),indent=2)+'\n')
    print('Campfires',json.dumps(allcamp),flush=True)

if __name__=='__main__':main()
