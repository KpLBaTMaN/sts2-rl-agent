"""Reproducible descriptive audit; freezes logged inputs without touching training."""
from __future__ import annotations
import csv, hashlib, json, math, shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/training-audit-20261019'
SNAP = OUT / 'inputs'
BATCH = ROOT / 'output/reward-comparison-20261019'

def write_json(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def rows(p):
    data = list(csv.DictReader(p.open(encoding='utf-8', newline='')))
    for r in data:
        for k, v in r.items():
            if v in ('True', 'False'): r[k] = v == 'True'
            else:
                try: r[k] = float(v) if '.' in v or 'e' in v.lower() else int(v)
                except (ValueError, TypeError): pass
    return data

def wilson(k, n):
    if not n: return [None, None]
    z=1.95996398454; p=k/n; d=1+z*z/n
    m=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [m-h,m+h]

def mean(data, key): return float(np.mean([r[key] for r in data])) if data else None

def snapshot():
    if (OUT/'manifest.json').exists(): return
    OUT.mkdir(parents=True,exist_ok=True)
    files=[]
    for run in sorted((BATCH/'runs').iterdir()):
        dest=SNAP/run.name;dest.mkdir(parents=True,exist_ok=True)
        for p in run.iterdir():
            if p.is_file() and p.suffix in ('.json','.jsonl','.csv'):
                raw=p.read_bytes()
                if p.suffix=='.jsonl': raw=raw[:raw.rfind(b'\n')+1]
                (dest/p.name).write_bytes(raw)
                files.append(dict(source=str(p),snapshot=str(dest/p.name),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        for name in ('best_model/best_model.pt','final_model.pt'):
            p=run/name
            if p.exists():
                target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(p,target)
                files.append(dict(source=str(p),snapshot=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    for watch in sorted((ROOT/'output/watch').glob('*/decisions.jsonl')):
        dest=SNAP/'watch'/watch.parent.name;dest.mkdir(parents=True,exist_ok=True)
        raw=watch.read_bytes();raw=raw[:raw.rfind(b'\n')+1];(dest/watch.name).write_bytes(raw)
        for name in ('session.json','summary.json'):
            p=watch.parent/name
            if p.exists():shutil.copy2(p,dest/name)
        files.append(dict(source=str(watch),snapshot=str(dest/watch.name),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    write_json(OUT/'manifest.json',dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),files=files))

def analyze():
    snapshot();report=dict(manifest=json.loads((OUT/'manifest.json').read_text()),runs={},comparisons={},watch={})
    tables={};evaluation_rows=[]
    for run in sorted(SNAP.glob('long-*')):
        cfg=json.loads((run/'config.json').read_text());profile=cfg['reward']['profile']
        metrics=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
        rollouts=[r for r in metrics if r['kind']=='rollout']
        ev=[r for r in metrics if r['kind']=='evaluation']
        periodic=[r for r in ev if r.get('label')=='periodic']
        data=dict(directory=run.name,config=cfg,rollouts=len(rollouts),steps=max(r.get('steps',0) for r in metrics),
                  periodic=periodic,completed=(run/'final-results.json').exists(),results={})
        for label in ('best-results','final-results','latest-evaluation','random-baseline'):
            file=run/f'{label}-episodes.csv'
            if not file.exists():continue
            rr=rows(file);summary=json.loads((run/f'{label}.json').read_text())
            assert len(rr)==summary['episodes'] and len({r['seed'] for r in rr})==len(rr)
            assert sum(r['win'] for r in rr)==summary['wins']
            assert sum(r['timed_out'] for r in rr)==summary['timeouts']
            assert all(not(r['win'] and r['timed_out']) for r in rr)
            timeout=[r for r in rr if r['timed_out']]
            info=dict(summary=summary,win_ci=wilson(summary['wins'],len(rr)),
                boss_ci=wilson(sum(r['boss_reached'] for r in rr),len(rr)),
                timeout_decision_share=sum(r['decisions'] for r in timeout)/sum(r['decisions'] for r in rr),
                timeout_floors=dict(Counter(r['floor'] for r in timeout)),
                timeout_relics=dict(Counter(r['relics'] for r in timeout)),
                timeouts_with_hp=sum(r['hp']>0 for r in timeout),
                non_timeout_mean_decisions=mean([r for r in rr if not r['timed_out']],'decisions'),
                at_least_one_upgrade=sum(r['upgraded_cards']>0 for r in rr)/len(rr),
                at_least_one_extra_relic=sum(r['relic_count']>1 for r in rr)/len(rr),
                boss_rows=[dict(seed=r['seed'],win=r['win'],hp=r['hp'],timed_out=r['timed_out']) for r in rr if r['boss_reached']])
            data['results'][label]=info;tables[(profile,label)]=rr
            evaluation_rows += [dict(profile=profile,checkpoint=label,**r) for r in rr]
        if periodic:
            best=max(periodic,key=lambda e:(e['summary']['win_rate'],e['summary']['average_floor']))
            data['selected_validation_steps']=best['steps']
            data['evaluation_seconds']=sum(max(0,r.get('elapsed_seconds',0)-next((m.get('elapsed_seconds',0) for m in reversed(metrics[:i]) if m['kind']=='evaluating' and m.get('completed')==0),r.get('elapsed_seconds',0))) for i,r in enumerate(metrics) if r['kind']=='evaluation')
        data['training_first100']={k:float(np.mean([r[k] for r in rollouts[:100]])) for k in ('train_win_rate','entropy','clip_fraction','value_loss','floor','episode_length','fps')}
        data['training_last100']={k:float(np.mean([r[k] for r in rollouts[-100:]])) for k in ('train_win_rate','entropy','clip_fraction','value_loss','floor','episode_length','fps')}
        data['last_rollout']=rollouts[-1]
        write_json(OUT/f'{profile}-rollouts.json',rollouts)
        report['runs'][profile]=data
    rng=np.random.default_rng(1701)
    for label, left, right in [('best-results','shaped','sparse'),('final-results','shaped','sparse')]:
        a={r['seed']:r for r in tables[left,label]};b={r['seed']:r for r in tables[right,label]};seeds=sorted(a.keys()&b.keys())
        delta=np.array([int(a[s]['win'])-int(b[s]['win']) for s in seeds]);idx=rng.integers(0,len(seeds),(20000,len(seeds)))
        boot=delta[idx].mean(1)
        winleft=sum(a[s]['win'] and not b[s]['win'] for s in seeds);winright=sum(b[s]['win'] and not a[s]['win'] for s in seeds);n=winleft+winright
        p=min(1,2*sum(math.comb(n,i) for i in range(min(winleft,winright)+1))/2**n) if n else 1
        floor=np.array([a[s]['floor']-b[s]['floor'] for s in seeds])
        report['comparisons'][label]=dict(n=len(seeds),left=left,right=right,win_difference=float(delta.mean()),
            paired_bootstrap_ci=np.quantile(boot,[.025,.975]).tolist(),left_only=winleft,right_only=winright,
            mcnemar_exact_p=p,mean_floor_difference=float(floor.mean()),floor_bootstrap_ci=np.quantile(floor[idx].mean(1),[.025,.975]).tolist())
    for profile in ('sparse','shaped'):
        a={r['seed']:r for r in tables[profile,'best-results']};b={r['seed']:r for r in tables[profile,'final-results']}
        delta=np.array([int(a[s]['win'])-int(b[s]['win']) for s in sorted(a)])
        idx=rng.integers(0,len(delta),(20000,len(delta)))
        report['comparisons'][profile+'_best_vs_final']=dict(win_difference=float(delta.mean()),ci=np.quantile(delta[idx].mean(1),[.025,.975]).tolist())
    # Trace repetitions are diagnostics, not independent episodes. Retain the latest
    # session for each (checkpoint, seed) when describing campfire behavior.
    latest={}
    for p in sorted((SNAP/'watch').glob('*/decisions.jsonl')):
        ss=p.parent/'session.json'
        if not ss.exists():continue
        session=json.loads(ss.read_text());key=(session['model'],session['live']['seed'])
        trace=[json.loads(l) for l in p.read_text().splitlines()]
        if trace:latest[key]=(p,session,trace)
    watch=[]
    for p,session,trace in latest.values():
        camp=[];illegal=[]
        for d in trace:
            s,w=d['state'],d['sent']
            if s['type']=='rest_site':
                options={o.get('index',i):o for i,o in enumerate(s['options'])}
                camp.append(dict(hp=s['run_state']['hp'],max_hp=s['run_state']['max_hp'],floor=s['run_state']['floor'],
                    chosen=options.get(w.get('index'),{}).get('id'),probabilities=d['proposal']['probabilities']))
            if s['type']=='combat_action' and w['action']=='play' and w.get('card_index',100)<len(s['hand']):
                card=s['hand'][w['card_index']]
                if not card.get('playable',True):illegal.append(dict(request=s['request_id'],card=card['id']))
        watch.append(dict(path=str(p),model=session['model'],seed=session['live']['seed'],decisions=len(trace),campfires=camp,illegal=illegal))
    report['watch']=watch
    # Saving selected tables keeps all computed claims inspectable.
    with (OUT/'evaluation-episodes.csv').open('w',newline='',encoding='utf-8') as f:
        fields=list(dict.fromkeys(k for row in evaluation_rows for k in row));writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(evaluation_rows)
    write_json(OUT/'analysis.json',report)
    concise={p:dict(steps=d['steps'],selected_validation_steps=d.get('selected_validation_steps'),results={k:dict(wins=v['summary']['wins'],n=v['summary']['episodes'],timeout_share=round(v['timeout_decision_share'],3),timeouts=v['summary']['timeouts'],timeout_relics=v['timeout_relics']) for k,v in d['results'].items()}) for p,d in report['runs'].items()}
    print(json.dumps(concise,indent=2));print('Comparisons',json.dumps(report['comparisons'],indent=2))

if __name__=='__main__':analyze()
