'use strict';
const $ = id => document.getElementById(id);
const colors = {sparse:'#66dfba',shaped:'#f2b16e',survival:'#b4a0ff'};
const titles = {sparse:'Sparse',shaped:'Shaped',survival:'Survival'};
const descriptions = {sparse:'Learn from the objective result alone.',shaped:'Reward progress, health and combat success.',survival:'Give health preservation more weight.'};
const nf = new Intl.NumberFormat('en-GB');
const esc = s => String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = v => Number.isFinite(v) ? `${(v*100).toFixed(1)}%` : '—';
const num = (v,d=1) => Number.isFinite(v) ? v.toFixed(d) : '—';
const compact = v => v>=1e6 ? `${(v/1e6).toFixed(v%1e6===0?0:2)}M` : v>=1000 ? `${(v/1000).toFixed(0)}K` : String(Math.round(v));
const get = (o,path) => path.split('.').reduce((a,k)=>a?.[k],o);
const last = a => a?.[a.length-1];
const flatten = (o,p='') => Object.entries(o).flatMap(([k,v])=>v&&typeof v==='object'&&!Array.isArray(v)?flatten(v,p+k+'.'):[[p+k,v]]);
let state, frozen=false, busy=false, activeTab='overview', lastLogProfile='', lastConfigKey='', charts=[];
const visible = new Set(Object.keys(colors));
const metrics = {
  train_win_rate:['Training win rate','Rolling mean of the last 200 completed training episodes; exploratory actions.','percent'],
  floor:['Training floor reached','Rolling mean floor over the last 200 completed training episodes.','number'],
  reward:['Episode reward','Profile-specific reward over the last 200 episodes. Scales differ between profiles.','number'],
  episode_length:['Decisions per game run','Average decisions per completed training episode, rolling window of 200.','number'],
  loop_rate:['Decision loop rate','Share of the last 200 training episodes that repeated an exact decision cycle three times.','percent'],
  entropy:['Policy entropy','Action uncertainty during PPO updates. Lower values mean more concentrated choices.','number'],
  policy_loss:['Policy loss','Mean clipped PPO policy loss across minibatch updates.','number'],
  value_loss:['Value loss','Mean value prediction loss across minibatch updates.','number'],
  clip_fraction:['PPO clip fraction','Fraction of sampled probability ratios outside the PPO clipping interval.','percent'],
  fps:['Training speed','Cumulative decisions per second, including evaluation overhead so far.','number'],
  episodes:['Completed game runs','Cumulative finished training attempts, including wins, losses and timeouts. Evaluation attempts are excluded.','number'],
  runs_per_hour:['Game runs per hour','Completed training attempts divided by elapsed hours at each rollout, including evaluation and update overhead.','number']
};
for(const [key,title] of Object.entries({gold:'Gold',max_hp:'Maximum HP',deck_size:'Deck size',upgraded_cards:'Upgraded cards',relic_count:'Relics',potions_held:'Potions held',card_removals:'Shop card removals',elites:'Elite rooms visited',monsters:'Monster rooms visited',shops:'Shops visited',rest_sites:'Rest sites visited',events:'Events visited',treasures:'Treasures visited',attacks:'Attack cards',skills:'Skill cards',powers:'Power cards',curses_statuses:'Curses / statuses'})) metrics['run_averages.'+key]=[title,'End-of-episode average over the last 200 completed training runs.','number'];
for(const key of ['win','death','timeout','floor','hp','elite','enemy_damage','max_hp','relic','gold','upgrade','combat_turn']) metrics['reward_terms.'+key]=['Reward contribution · '+key,'Average per-episode contribution from this reward term, last 200 episodes. Profile scales differ.','number'];
$('metric').innerHTML=Object.entries(metrics).map(([key,[title]])=>`<option value="${key}">${title}</option>`).join('');

function duration(seconds){
  if(!Number.isFinite(seconds)) return '—';
  const hours=Math.floor(seconds/3600),minutes=Math.floor(seconds%3600/60);
  return hours ? `${hours}h ${minutes}m` : `${minutes}m ${Math.round(seconds%60)}s`;
}
function currentSteps(run){return Math.max(run.live?.steps||0,last(run.rollouts)?.steps||0);}
function validation(run){return last(run.evaluations.filter(e=>e.label==='periodic'))?.summary;}
function metricValue(row,key){
  if(key==='runs_per_hour')return Number.isFinite(row.episodes)&&row.elapsed_seconds>0?row.episodes*3600/row.elapsed_seconds:null;
  if(key==='episode_length'&&!row.episodes)return null;
  return get(row,key);
}
function trainingActivity(run){
  // live.episodes is the evaluation batch size while evaluating; use training rollouts only.
  const row=last(run.rollouts),completed=Number.isFinite(row?.episodes)?row.episodes:row?null:0;
  return {completed,recentDecisions:row?metricValue(row,'episode_length'):null,
    runsPerHour:row?metricValue(row,'runs_per_hour'):null,
    floor:completed>0&&Number.isFinite(row?.floor)?row.floor:null};
}
function render(){
  if(!state) return;
  const runs=state.runs,active=runs.find(r=>r.status==='running'),selected=active||[...runs].reverse().find(r=>r.status==='completed');
  const total=runs.reduce((n,r)=>n+r.config.training.total_timesteps,0),steps=runs.reduce((n,r)=>n+Math.min(currentSteps(r),r.config.training.total_timesteps),0);
  $('total-progress').textContent=pct(steps/total);$('overall-bar').value=steps/total*100;
  $('progress-caption').textContent=`${nf.format(steps)} / ${nf.format(total)} decisions`;
  $('batch-state').textContent=state.batch.status;$('batch-state').className='pill '+state.batch.status;
  $('active-run').textContent=active?titles[active.profile]:state.batch.status==='completed'?'Complete':state.batch.status==='paused'?'Queue paused':state.batch.status==='failed'?'Stopped':'Waiting';
  const live=active?.live;
  $('active-caption').textContent=active?(live.kind==='evaluating'?`${live.label} evaluation · ${live.completed} / ${live.episodes}`:live.kind==='starting'?'Building rollout buffers':live.steps?`${nf.format(currentSteps(active))} decisions · ${live.kind}`:'Initialising / random baseline'):state.batch.status==='completed'?'All three profiles evaluated':'One model trains at a time';
  const summary=selected&&(selected.best||validation(selected));
  $('validation-win').textContent=pct(summary?.win_rate);
  $('validation-caption').textContent=summary?`${titles[selected.profile]} · ${summary.wins} / ${summary.episodes} ${selected.best?'final test':'validation'} wins`:'First validation after 250K decisions';
  const activities=runs.map(trainingActivity),knownCounts=activities.every(a=>Number.isFinite(a.completed));
  $('completed-runs').textContent=knownCounts?nf.format(activities.reduce((n,a)=>n+a.completed,0)):'—';
  $('completed-caption').textContent='Training attempts · wins, losses & timeouts';
  const activity=selected&&trainingActivity(selected);
  $('decisions-per-run').textContent=num(activity?.recentDecisions);
  $('decisions-caption').textContent=activity?.completed?`${titles[selected.profile]} · last ${Math.min(200,activity.completed)} completed runs`:'Waiting for a completed training run';
  const row=last(active?.rollouts),speed=row?.fps;
  $('throughput').textContent=speed?`${nf.format(Math.round(speed))} /s`:'—';
  $('eta').textContent=speed?`~${duration((total-steps)/speed)} left in queue · estimate`:'Estimating after the first rollout';
  $('pause-queue').textContent=state.control.pause_after_current?'Resume queue':'Pause after current experiment';
  $('pause-queue').disabled=['completed','failed'].includes(state.batch.status);
  const error=$('error-banner');error.hidden=!state.batch.error;error.textContent=state.batch.error||'';
  $('sequence').innerHTML=runs.map((r,i)=>{
    const n=currentSteps(r),goal=r.config.training.total_timesteps,percent=Math.min(n/goal,1)*100;
    const activity=trainingActivity(r);
    const label=r.status==='running'?(r.live.kind==='evaluating'?'Evaluating':'Training'):r.status;
    return `<button class="run-card ${r.status==='running'?'active':''}" data-inspect="${r.profile}" title="Inspect ${titles[r.profile]} variables"><div class="run-card-top"><span class="run-name"><span class="run-number">0${i+1}</span><span class="${r.profile}-text">${titles[r.profile]}</span></span><span class="run-state">${esc(label)}</span></div><div class="run-description">${descriptions[r.profile]}</div><div class="run-bottom"><span>${compact(n)} / ${compact(goal)} decisions</span><strong>${percent.toFixed(1)}%</strong></div><div class="bar"><span class="${r.profile}" data-progress="${percent}"></span></div><div class="run-activity">${Number.isFinite(activity.completed)?nf.format(activity.completed):'—'} game runs · ${num(activity.recentDecisions)} decisions/run (recent)</div></button>`;
  }).join('');
  document.querySelectorAll('[data-inspect]').forEach(b=>b.onclick=()=>{$('config-run').value=b.dataset.inspect;switchTab('variables');renderVariables();});
  document.querySelectorAll('[data-progress]').forEach(b=>{b.style.width=b.dataset.progress+'%';});
  $('activity-body').innerHTML=runs.map(r=>{
    const a=trainingActivity(r);
    return `<tr><td><span class="row-label"><i class="swatch ${r.profile}"></i>${titles[r.profile]}</span></td><td>${nf.format(currentSteps(r))}</td><td>${Number.isFinite(a.completed)?nf.format(a.completed):'—'}</td><td>${num(a.recentDecisions)}</td><td>${Number.isFinite(a.runsPerHour)?nf.format(Math.round(a.runsPerHour)):'—'}</td><td>${num(a.floor)}</td></tr>`;
  }).join('');
  $('comparison-body').innerHTML=runs.map(r=>{
    const s=r.best||validation(r),source=r.best?'Best checkpoint on separate final holdout seeds':'Latest periodic validation (checkpoint selection seeds)';
    return `<tr title="${source}"><td><span class="row-label"><i class="swatch ${r.profile}"></i>${titles[r.profile]}</span></td><td><span class="status-label">${esc(r.status)}</span></td><td>${nf.format(currentSteps(r))}</td><td>${pct(s?.win_rate)}</td><td>${num(s?.average_floor)}</td><td>${pct(s?.boss_reached_rate)}</td><td>${num(s?.average_winning_hp)}</td><td>${s?.timeouts??'—'} / ${s?.stalls??'—'}</td><td>${r.checkpoints.length}</td></tr>`;
  }).join('');
  const gpu=state.gpu;
  $('hardware').textContent=gpu?.name?`${gpu.name} · ${gpu.utilization}% GPU · ${num(gpu.memory_used/1024,1)} / ${num(gpu.memory_total/1024,1)} GB · ${gpu.temperature}°C`:'GPU telemetry unavailable';
  $('output-location').title=state.root;
  $('last-update').textContent=`Updated ${new Date(state.server_utc).toLocaleTimeString('en-GB')} · every 5s`;
  if(activeTab==='variables')renderVariables();
  if(activeTab==='overview'&&!frozen)drawCharts();
}

function renderVariables(){
  if(!state)return;
  const runs=state.runs,terms={win:'Once the Act 1 boss is defeated',death:'Once on death',timeout:'Once on decision limit',floor:'Per new floor reached',hp:'Change in curved HP fraction',elite:'Per elite victory',enemy_damage:'Fraction of enemy HP removed per combat',max_hp:'Per maximum HP point changed',relic:'Per relic gained',gold:'Per gold gained',upgrade:'Per completed permanent upgrade, once per card',combat_turn:'Per combat turn after the first',hp_curve:'HP reward function'};
  $('weights-body').innerHTML=Object.entries(terms).map(([key,note])=>`<tr><td>${esc(key)}</td>${runs.map(r=>`<td class="${r.profile}-text">${esc(r.config.reward[key] ?? 0)}</td>`).join('')}<td>${note}</td></tr>`).join('');
  const run=runs.find(r=>r.profile===$('config-run').value),query=$('search-config').value.toLowerCase();
  const key=run.profile+'|'+query;if(key===lastConfigKey)return;lastConfigKey=key;
  $('config-body').innerHTML=flatten(run.config).filter(([k,v])=>(k+' '+JSON.stringify(v)).toLowerCase().includes(query)).map(([k,v])=>`<tr><td>${esc(k)}</td><td>${Array.isArray(v)?`<details><summary>${v.length} values · ${esc(v.slice(0,5).join(', '))}${v.length>5?'…':''}</summary><div>${esc(v.join(', '))}</div></details>`:esc(v===null?'null':v)}</td></tr>`).join('');
}

function series(key,evaluation=false){
  const smoothing=Number($('smoothing').value),range=Number($('range').value);
  return state.runs.filter(r=>visible.has(r.profile)).map(r=>{
    let data=(evaluation?r.evaluations.filter(e=>e.label==='periodic'):r.rollouts).map(row=>({x:row.steps,y:evaluation?get(row.summary,key):metricValue(row,key)})).filter(p=>Number.isFinite(p.y));
    if(!evaluation&&smoothing>1)data=data.map((p,i)=>({...p,y:data.slice(Math.max(0,i-smoothing+1),i+1).reduce((n,v)=>n+v.y,0)/Math.min(i+1,smoothing)}));
    if(range&&data.length){const end=last(data).x;data=data.filter(p=>p.x>=end-range);}
    return {profile:r.profile,label:titles[r.profile],color:colors[r.profile],points:data};
  });
}

function draw(canvasId,datasets,{percent=false,zero=true}={}){
  const canvas=$(canvasId),parent=canvas.parentElement,rect=parent.getBoundingClientRect();
  if(!rect.width||!rect.height)return;
  const dpr=window.devicePixelRatio||1,w=rect.width,h=rect.height;
  canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
  const pad={l:47,r:15,t:14,b:31},pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;
  const points=datasets.flatMap(s=>s.points),xs=points.map(p=>p.x),ys=points.map(p=>p.y);
  let minX=Number($('range').value)&&xs.length?Math.min(...xs):0,maxX=Math.max(100000,...xs);
  let minY=zero?Math.min(0,...ys):Math.min(0,...ys),maxY=percent?1:Math.max(1,...ys);
  if(maxY===minY)maxY=minY+1;
  if(!percent)maxY+=(maxY-minY)*.08;
  const x=v=>pad.l+(v-minX)/(maxX-minX)*pw,y=v=>pad.t+ph-(v-minY)/(maxY-minY)*ph;
  ctx.font='10px Segoe UI';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){
    const value=minY+(maxY-minY)*i/4,py=y(value);ctx.strokeStyle='#24303b';ctx.setLineDash([2,4]);ctx.beginPath();ctx.moveTo(pad.l,py);ctx.lineTo(w-pad.r,py);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle='#7c8fa1';ctx.textAlign='right';ctx.fillText(percent?`${Math.round(value*100)}%`:Math.abs(value)>1000?compact(value):num(value,Math.abs(maxY)<10?2:1),pad.l-10,py+3);
  }
  ctx.textAlign='center';for(let i=0;i<=4;i++){const value=minX+(maxX-minX)*i/4;ctx.fillText(compact(value),x(value),h-10);}
  for(const s of datasets){if(!s.points.length)continue;ctx.strokeStyle=s.color;ctx.lineWidth=2;ctx.lineJoin='round';ctx.beginPath();s.points.forEach((p,i)=>i?ctx.lineTo(x(p.x),y(p.y)):ctx.moveTo(x(p.x),y(p.y)));ctx.stroke();if(s.points.length<60)s.points.forEach(p=>{ctx.fillStyle=s.color;ctx.beginPath();ctx.arc(x(p.x),y(p.y),3,0,2*Math.PI);ctx.fill();});}
  if(!points.length){ctx.fillStyle='#a4b4c2';ctx.font='12px Segoe UI';ctx.textAlign='center';ctx.fillText(visible.size?'Waiting for the first measurement':'Select a reward profile to plot',w/2,h/2);ctx.fillStyle='#65798c';ctx.font='10px Segoe UI';ctx.fillText('Live data appears here as training progresses',w/2,h/2+22);}
  const tooltip=parent.querySelector('.tooltip');
  canvas.onmousemove=e=>{
    if(!points.length)return;const px=e.offsetX,target=minX+(px-pad.l)/pw*(maxX-minX);
    const values=datasets.map(s=>({s,p:s.points.reduce((best,p)=>!best||Math.abs(p.x-target)<Math.abs(best.x-target)?p:best,null)})).filter(v=>v.p);
    tooltip.textContent=values.map(({s,p})=>`${s.label} · ${nf.format(p.x)} decisions\n${percent?pct(p.y):num(p.y,3)}`).join('\n');
    tooltip.hidden=false;tooltip.style.left=Math.max(0,Math.min(px+12,w-190))+'px';tooltip.style.top='20px';
  };
  canvas.onmouseleave=()=>{tooltip.hidden=true;};
}
function drawCharts(){
  const evaluation=$('curve-source').value==='validation';
  $('win-eyebrow').textContent=evaluation?'GENERALISATION':'LEARNING';
  $('win-source-tag').textContent=evaluation?'VALIDATION · 50 SEEDS':'TRAINING · LAST 200 EPISODES';
  $('floor-source-tag').textContent=evaluation?'VALIDATION · AVG FLOOR':'TRAINING · AVG FLOOR';
  $('win-description').textContent=evaluation?'Deterministic evaluation on the same reserved seeds for every profile.':'Rolling win rate over the last 200 completed exploratory training episodes.';
  $('floor-description').textContent=evaluation?'Floor reached during evaluation, including unsuccessful attempts.':'Rolling average floor over the last 200 completed training episodes.';
  draw('win-chart',series(evaluation?'win_rate':'train_win_rate',evaluation),{percent:true});draw('floor-chart',series(evaluation?'average_floor':'floor',evaluation));
  const key=$('metric').value;draw('metric-chart',series(key),{percent:metrics[key][2]==='percent'});$('metric-description').textContent=metrics[key][1];
}
function switchTab(tab){activeTab=tab;document.querySelectorAll('.view').forEach(v=>v.hidden=v.id!==tab);document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('selected',b.dataset.tab===tab));if(tab==='variables')renderVariables();if(tab==='overview')drawCharts();if(tab==='logs')loadLog();}
async function loadLog(){
  const profile=$('log-run').value;
  try{const response=await fetch('/api/log?profile='+profile);if(!response.ok)throw Error(response.status);const log=await response.json();$('log-text').textContent=log.text;if($('follow-log').checked||lastLogProfile!==profile)$('log-text').scrollTop=$('log-text').scrollHeight;lastLogProfile=profile;}catch(e){$('log-text').textContent='Could not load log: '+e.message;}
}
async function refresh(){
  if(busy)return;busy=true;
  try{const response=await fetch('/api/state');if(!response.ok)throw Error('HTTP '+response.status);state=await response.json();render();$('connection').textContent='Live connection';$('connection-dot').classList.remove('offline');if(activeTab==='logs')await loadLog();}
  catch(e){$('connection').textContent='Disconnected · retrying';$('connection-dot').classList.add('offline');$('last-update').textContent='Data may be stale · '+e.message;}
  finally{busy=false;}
}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>switchTab(b.dataset.tab));
document.querySelectorAll('[data-profile]').forEach(b=>b.onchange=()=>{b.checked?visible.add(b.dataset.profile):visible.delete(b.dataset.profile);drawCharts();});
for(const id of ['range','smoothing','metric','curve-source'])$(id).onchange=drawCharts;
$('freeze').onclick=()=>{frozen=!frozen;$('freeze').textContent=frozen?'Resume charts':'Freeze charts';if(!frozen)drawCharts();};
$('config-run').onchange=renderVariables;$('search-config').oninput=renderVariables;$('log-run').onchange=loadLog;
$('download-metrics').onclick=()=>{const profile=$('export-run').value;window.location.href='/api/export?kind=metrics&profile='+profile;};
$('download-config').onclick=()=>{window.location.href='/api/export?kind=config&profile='+$('config-run').value;};
$('pause-queue').onclick=async()=>{const path=state.control.pause_after_current?'/api/resume-queue':'/api/pause-queue';$('pause-queue').disabled=true;try{const response=await fetch(path,{method:'POST',headers:{'X-Dashboard-Request':'1'}});if(!response.ok)throw Error('HTTP '+response.status);await refresh();}catch(e){alert('Queue control failed: '+e.message);$('pause-queue').disabled=false;}};
window.addEventListener('resize',()=>{if(state&&activeTab==='overview'&&!frozen)drawCharts();});
refresh();setInterval(refresh,5000);
