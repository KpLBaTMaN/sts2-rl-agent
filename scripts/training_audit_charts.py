"""Publication-style audit figures from frozen JSON/CSV inputs."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/training-audit-20261019';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
r=json.loads((OUT/'analysis.json').read_text());probes=json.loads((OUT/'policy-probes.json').read_text())
COL={'sparse':'#3166a8','shaped':'#008575','survival':'#d48a18'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','axes.grid':True,'grid.alpha':.18,'savefig.dpi':180})
def save(name):plt.savefig(FIG/f'{name}.png',bbox_inches='tight');plt.close()
fig,ax=plt.subplots(1,2,figsize=(10.5,3.5))
for p,d in r['runs'].items():
 e=d['periodic'];x=[v['steps']/1e6 for v in e]
 ax[0].plot(x,[v['summary']['win_rate']*100 for v in e],'-o',ms=3,color=COL[p],label=p.title())
 ax[1].plot(x,[v['summary']['average_floor'] for v in e],'-o',ms=3,color=COL[p])
for a in ax:a.set_xlabel('Training decisions (millions)');a.axvline(1.253376,color='#999999',ls=':',lw=1)
ax[0].set(title='Validation win rate: progress is unstable',ylabel='Wins (%)',ylim=(-.5,10));ax[0].legend(frameon=False)
ax[1].set(title='Shaped improves progress more than wins',ylabel='Mean floor reached',ylim=(0,17));fig.tight_layout();save('learning-curves')
fig,ax=plt.subplots(1,2,figsize=(10.5,3.5));labels=[];wins=[];lo=[];hi=[];colors=[];deaths=[];tos=[]
for p in ('sparse','shaped'):
 for ck in ('best-results','final-results'):
  d=r['runs'][p]['results'][ck];s=d['summary'];labels.append(p.title()+'\n'+('Selected' if ck=='best-results' else 'Final'));wins.append(s['win_rate']*100);lo.append((s['win_rate']-d['win_ci'][0])*100);hi.append((d['win_ci'][1]-s['win_rate'])*100);colors.append(COL[p]);deaths.append(s['losses']/2);tos.append(s['timeouts']/2)
x=np.arange(4);ax[0].bar(x,wins,color=colors,alpha=.85);ax[0].errorbar(x,wins,yerr=[lo,hi],fmt='none',ecolor='#263344',capsize=4)
for i,w in enumerate(wins):ax[0].text(i,w+.3,f'{int(w*2)}/200',ha='center',fontsize=9)
ax[0].set(title='Holdout wins and 95% Wilson intervals',ylabel='Wins (%)',ylim=(0,12),xticks=x,xticklabels=labels)
ax[1].bar(x,wins,label='Win',color='#008575');ax[1].bar(x,deaths,bottom=wins,label='Loss',color='#a8b4c1');ax[1].bar(x,tos,bottom=np.array(wins)+deaths,label='Timeout',color='#d48a18');ax[1].set(title='Failure mix on the same 200 seeds',ylabel='Episodes (%)',ylim=(0,100),xticks=x,xticklabels=labels);ax[1].legend(frameon=False,ncol=3,fontsize=10,loc='upper center',bbox_to_anchor=(.5,-.22));fig.tight_layout();save('holdout-results')
fig,ax=plt.subplots(1,2,figsize=(10.5,3.5));lab=['Sparse selected','Shaped selected'];shares=[r['runs'][p]['results']['best-results']['timeout_decision_share']*100 for p in ('sparse','shaped')]
ax[0].barh(lab,shares,color='#d48a18');ax[0].barh(lab,[100-v for v in shares],left=shares,color='#dbe2e9');ax[0].set(xlim=(0,100),xlabel='Share of all evaluation decisions (%)',title='Timeouts consume most decision budget')
for i,s in enumerate(shares):ax[0].text(s/2,i,f'{s:.1f}%',ha='center',va='center',weight='bold')
inter=json.loads((OUT/'confirmation-interventions.json').read_text());xx=np.arange(4)
ax[1].bar(xx-.17,[2000]*4,width=.34,label='Logged timeout limit',color='#d48a18');ax[1].bar(xx+.17,[v['steps'] for v in inter],width=.34,label='Confirmation intervention',color='#008575');ax[1].set(xticks=xx,xticklabels=['Shaped\n20019','Shaped\n20026','Sparse\n20019','Sparse\n20026'],ylabel='Decisions',title='Four diagnostic failures resolve');ax[1].legend(frameon=False,fontsize=10,loc='upper center',bbox_to_anchor=(.5,-.24));fig.tight_layout();save('choice-bottleneck')
fig,ax=plt.subplots(figsize=(10.5,3.5))
for m in dict.fromkeys(d['model'] for d in probes['campfires']):
 data=[d for d in probes['campfires'] if d['model']==m];p=m.split('/')[0]
 ax.semilogy([d['hp_fraction']*100 for d in data],[max(d['probabilities'].get('rest_site: Smith',0)*100,1e-9) for d in data],'-o' if '/best' in m else '--o',color=COL[p],label=m)
ax.set(title='Campfire probe: upgrading has almost no probability',xlabel='HP (% of maximum)',ylabel='Smith probability (%) - log scale',xticks=[25,50,75,100],ylim=(1e-8,100));ax.legend(frameon=False,fontsize=11,ncol=3);fig.tight_layout();save('campfire-probe')
fig,ax=plt.subplots(1,2,figsize=(10.5,3.5));terms=['floor','hp','elite','enemy_damage','max_hp','relic'];width=.36
for j,ck in enumerate(('best-results','final-results')):
 data=r['runs']['shaped']['results'][ck]['summary']['average_reward_terms'];ax[0].bar(np.arange(len(terms))+(j-.5)*width,[data[t] for t in terms],width,label='Selected' if j==0 else 'Final',color='#008575' if j==0 else '#97cfc4')
ax[0].axhline(0,color='#333',lw=.8);ax[0].set(xticks=np.arange(len(terms)),xticklabels=['Floor','HP','Elite','Enemy\ndamage','Max\nHP','Relic'],ylabel='Mean shaping reward per episode',title='Shaping signal: progress and HP dominate');ax[0].legend(frameon=False)
for i,p in enumerate(('sparse','shaped')):
 s=r['runs'][p]['results']['best-results']['summary'];ax[1].bar(i-.18,s['boss_reached_rate']*100,.36,color=COL[p],label='Reach boss' if i==0 else None);ax[1].bar(i+.18,s['boss_conversion_rate']*100,.36,color=COL[p],alpha=.35,label='Win given boss reached' if i==0 else None)
ax[1].set(xticks=[0,1],xticklabels=['Sparse selected','Shaped selected'],ylabel='Rate (%)',title='Reaching the boss is not beating the boss');ax[1].legend(frameon=False,fontsize=10,loc='upper center',bbox_to_anchor=(.5,-.15));fig.tight_layout();save('reward-and-boss')
fig,ax=plt.subplots(1,2,figsize=(10.5,3.5))
for p,d in r['runs'].items():
 data=json.loads((OUT/f'{p}-rollouts.json').read_text());x=np.array([v['steps']/1e6 for v in data]);
 for a,key in zip(ax,('entropy','clip_fraction')):
  y=np.array([v[key] for v in data]);w=30
  if len(y)>=w:a.plot(x[w-1:],np.convolve(y,np.ones(w)/w,'valid')*(100 if key=='clip_fraction' else 1),color=COL[p],label=p.title())
ax[0].set(title='Global entropy does not reveal campfire collapse',xlabel='Training decisions (millions)',ylabel='Entropy (nats; 30-rollout average)');ax[0].legend(frameon=False)
ax[1].set(title='PPO clipping: no obvious runaway spike',xlabel='Training decisions (millions)',ylabel='Clipped samples (%; 30-rollout average)');fig.tight_layout();save('optimizer-diagnostics')
print('Wrote',len(list(FIG.glob('*.png'))),'audit charts')
