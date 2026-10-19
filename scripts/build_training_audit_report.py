"""Build the reviewed PDF audit from frozen data and diagnostic results."""
from pathlib import Path
import json, math
from datetime import datetime
from zoneinfo import ZoneInfo
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/training-audit-20261019';PDF=ROOT/'output/pdf/training-audit-20261019.pdf';PDF.parent.mkdir(exist_ok=True)
r=json.loads((OUT/'analysis.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text());probes=json.loads((OUT/'policy-probes.json').read_text());inter=json.loads((OUT/'confirmation-interventions.json').read_text())
for name,file in [('Audit','arial.ttf'),('AuditBold','arialbd.ttf')]:
    pdfmetrics.registerFont(TTFont(name,Path('C:/Windows/Fonts')/file))
pdfmetrics.registerFontFamily('Audit',normal='Audit',bold='AuditBold',italic='Audit',boldItalic='AuditBold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle('BodyAudit',fontName='Audit',fontSize=9.6,leading=13.5,textColor=colors.HexColor('#263549'),spaceAfter=8))
styles.add(ParagraphStyle('SmallAudit',parent=styles['BodyAudit'],fontSize=8.1,leading=11.3,spaceAfter=6))
styles.add(ParagraphStyle('TitleAudit',fontName='AuditBold',fontSize=26,leading=31,textColor=colors.HexColor('#142d43'),spaceAfter=14))
styles.add(ParagraphStyle('HeadingAudit',fontName='AuditBold',fontSize=17,leading=22,textColor=colors.HexColor('#142d43'),spaceAfter=12))
styles.add(ParagraphStyle('SubAudit',fontName='AuditBold',fontSize=11,leading=15,textColor=colors.HexColor('#007f72'),spaceBefore=5,spaceAfter=7))
styles.add(ParagraphStyle('CellAudit',parent=styles['SmallAudit'],fontSize=8.2,leading=11,spaceAfter=0))
story=[]
def p(text,small=False):story.append(Paragraph(text,styles['SmallAudit' if small else 'BodyAudit']))
def title(text):story.append(Paragraph(text,styles['TitleAudit']))
def heading(text):story.append(Paragraph(text,styles['HeadingAudit']))
def sub(text):story.append(Paragraph(text,styles['SubAudit']))
def page():story.append(PageBreak())
def table(header,data,widths):
    cells=[[Paragraph(str(x),styles['CellAudit']) for x in row] for row in [header]+data]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eef4')),('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#b3c3cf')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f8fa')]),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#cad5de'))]))
    story.extend([t,Spacer(1,10)])
def fig(name,caption):
    from PIL import Image as PILImage
    path=OUT/'figures'/f'{name}.png';w,h=PILImage.open(path).size
    story.append(Image(str(path),width=509,height=509*h/w));p(caption,True)
def pct(v):return f'{100*v:.1f}%'
def source(text):p(text,True)

stamp=datetime.fromisoformat(manifest['snapshot_utc']).astimezone(ZoneInfo('Europe/London')).strftime('%d %B %Y, %H:%M %Z')
title('SpireFormer\ntraining audit')
p(f'<b>Evidence snapshot:</b> {stamp}. Two completed reward experiments and one ongoing experiment. CPU-only diagnostic probes; no new training was launched.')
sub('Decision: fix confirmation behavior before spending the next training budget')
p('The strongest bottleneck is a learned choice-completion failure. Selected policies repeatedly toggle or attempt choices instead of confirming them. Timeout episodes consume most evaluation decisions and contaminate apparent relic and deck-quality relationships. Reward tuning is still worthwhile, but it should follow a corrected, tested decision interface.')
data=[]
for profile in ('sparse','shaped'):
 for ck in ('best-results','final-results'):
    d=r['runs'][profile]['results'][ck];s=d['summary'];ci=d['win_ci']
    data.append([profile.title()+' '+('selected' if ck=='best-results' else 'final'),f"{s['wins']}/200 ({pct(s['win_rate'])})",f'{pct(ci[0])} - {pct(ci[1])}',pct(s['boss_reached_rate']),f"{s['timeouts']}/200"])
table(['Checkpoint','Holdout wins','95% win interval','Reach boss','Timeouts'],data,[115,107,112,85,90])
sub('What the evidence supports')
p('<b>1. Interface reliability is the first priority.</b> Four targeted confirmation interventions removed four known timeouts; one replay became a win. This is a mechanism test on selected failures, not a population win-rate estimate.')
p('<b>2. More steps did not consistently improve the policy.</b> Both selected checkpoints are from 3,002,368 steps. Shaped fell from 10/200 holdout wins at that checkpoint to 1/200 at the final 5,001,216-step checkpoint.')
p('<b>3. Reward shaping improved travel, not proven overall victory.</b> Selected shaped reached a mean floor of 10.50 versus 8.525 for selected sparse, but won 5% versus 6%. Their paired win-rate difference is inconclusive.')
p('<b>4. Campfire choice is nearly saturated.</b> Full-HP Rest appears in real traces and persists in saved-policy counterfactual probes. A phase-specific exploration and credit-assignment issue is more plausible than a lack of model capacity alone.')
p(f"<b>Survival is unfinished:</b> {r['runs']['survival']['steps']:,}/5,000,000 steps at the snapshot. Do not rank it against completed 5M runs yet.",True)
source('Evidence: frozen metrics.jsonl, per-seed evaluation CSVs, saved model metadata, live decision traces and CPU replays. Confidence intervals cover environment-seed variation for these checkpoints, not training-seed variation.')

page();heading('1 / Learning progress and uncertainty')
fig('learning-curves','Each validation point uses the same 50 seeds, 10000-10049. Repeated points are correlated and are not additional independent trials. The dotted line marks the last completed survival validation budget (~1.25M).')
fig('holdout-results','Holdout evaluations use the same 200 seeds, 20000-20199, for selected and final sparse/shaped checkpoints. Selected means chosen on validation win rate with mean floor as the tie-breaker.')
p('The selected shaped model reaches the boss in 64/200 episodes (32%), compared with 39/200 (19.5%) for selected sparse. But it converts only 10/64 boss visits (15.6%) to wins, versus 12/39 (30.8%). Better routing or survival to the boss is not sufficient combat/deck strength to beat it.')
p('Paired seed bootstrap, shaped minus sparse selected: win-rate difference <b>-1.0 percentage point, 95% interval [-4.0, +2.0]</b>; mean-floor difference <b>+1.975, interval [+1.27, +2.69]</b>. Exact paired discordance test for wins: p=0.754. These results do not establish a winning reward profile.')
p('Shaped selected versus shaped final improves wins by +4.5 points, paired 95% interval [+1.5, +8.0]. This supports preserving selected checkpoints. It does not prove overfitting is the cause; constant updates, evaluation noise and policy shifts are competing explanations.',True)

page();heading('2 / The largest actionable failure: choice loops')
fig('choice-bottleneck','Selected sparse: 34/200 timeouts consume 58.0% of evaluation decisions. Selected shaped: 29/200 consume 53.3%. These are evaluation decision shares; logged data do not identify the corresponding fraction of stochastic training decisions.')
table(['Diagnostic replay','Observed baseline','Confirmation intervention','Outcome'],[[f"{d['profile']} / seed {d['seed']}",'2000 decisions; timeout',f"{d['steps']} decisions; 1 confirmation",('Win' if d['win'] else f"Loss on floor {d['floor']}")] for d in inter],[119,119,163,108])
p('<b>Seed 20019:</b> both policies remain in a discard prompt on floor 4. They toggle selections instead of choosing action 0 to confirm. <b>Seed 20026:</b> both become stuck in Whetstone\'s two-card upgrade prompt on floor 9. The selected shaped model continues choosing an unselected card even after two cards are selected; the observation and action repeat without advancing.')
p('The diagnostic controller only confirms a legal pending choice after its quota is filled or after eight choice steps. It does not choose better combat cards or update weights. Across these four selected cases, decisions fall from 8000 to 1510 (81.1% fewer), all timeouts disappear, and sparse seed 20026 wins. This is strong evidence for this mechanism in these cases, with no estimate of overall uplift.')
p('<b>Important confound:</b> every extra-relic end state in the completed holdout tables occurs in a timeout. Whetstone and War Paint require upgrade choices. Their apparent 0% win association is therefore not evidence that they are bad relics. Do not use the existing item win-rate tables as causal card/relic rankings.')
p('<b>Recommended change:</b> represent a multi-choice decision as a valid set plus completion, or prevent no-op additions when the maximum is reached and offer a clear confirm action. Preserve meaningful optional selection; do not blanket auto-confirm every prompt at zero selections. Add trace-based regression cases and measure choice completion rate.',True)

page();heading('3 / Campfires, deck improvement and exploration')
fig('campfire-probe','Counterfactual probe on one actual campfire state: vary only consistent player HP fields across 25/50/75/100%; keep the deck, floor and options fixed. This is a local sensitivity probe, not a full benchmark. All five checkpoints choose Rest at all four HP levels.')
p('For selected shaped at full HP, Smith probability is <b>0.001335%</b>, approximately one Smith draw per 75,000 visits if that state and distribution stayed fixed. At 25% HP it is smaller still. Selected sparse is less saturated, with about 2.5% Smith probability at full HP in this state. Global training entropy around 1 nat does not reveal this phase-specific collapse.')
p('The latest deduplicated shaped traces contain seven campfire choices across two game seeds. All seven choose Rest; three are at full HP. Replayed sessions with the same model and seed are counted once. This is consistent with the sensitivity probe, but the tiny, selected live sample is not a general performance estimate.')
p('End-of-run permanent upgrades are rare: selected sparse averages 0.14 upgrades, with any upgraded card in 4.5% of holdout runs; selected shaped averages 0.67, with any upgraded card in 14.5%. These are end states, not counts of Smith actions. They also include relic/event upgrades, and some of the upgrade opportunities never complete due to the choice-loop issue.')
sub('Improve learning opportunities, not just the numeric reward')
p('After choice completion is reliable, use fresh initialization or a carefully tested curriculum. Include campfires at several HP fractions and decks with meaningful upgrade alternatives. Track campfire entropy, Rest-at-full-HP rate, Smith selections, completed upgrades and the eventual outcome separately. A small phase-specific entropy incentive can be tested; raising global entropy may mostly randomize combat.')
p('Keep Rest legal when it has real relic or other side benefits. A blanket full-HP ban hides the learning problem and can remove useful actions. Evaluate the behavior in both synthetic probes and naturally reached camps.',True)

page();heading('4 / Reward design: what to keep, what to test')
fig('reward-and-boss','Left: shaped reward contributions only; raw reward scales cannot be compared between profiles. Right: boss reach and conversion have different denominators (200 episodes versus boss visits).')
table(['Term','Existing shaped','Prepared deckbuilder','Rationale'],[
 ['Boss victory / death','+5 / -1','+5 / -1','Keep outcome as the target'],
 ['Permanent upgrade','0','+0.10','Pay for completed improvement'],
 ['HP-curve coefficient','1.0','0.5','Reduce immediate healing pressure'],
 ['Full-HP healing','0','0','No fictitious healing credit'],
 ['Adding cards / deck size','0','0','Avoid rewarding deck bloat'],
 ['Other shaping','Existing weights','Same weights','Isolate the two changes']],[109,99,110,191])
p('The deckbuilder profile is a reasonable <b>hypothesis</b>, not an empirically selected optimum. At 80 max HP, its immediate healing reward is 0 at 80 HP; +0.0094 for 77 to 80; +0.0902 for 40 to 64; and +0.1439 for 20 to 44. A permanent upgrade pays +0.10. Future survival remains part of the return; these are not fixed heal/Smith thresholds.')
p('The new implementation pays once per permanent card instance when an existing card becomes upgraded. Clicking Smith alone, acquiring an already-upgraded card, and temporary combat upgrades pay nothing. It does not reward indiscriminate removals or acquisitions. Picking the right card for the deck must still be learned from later outcomes.')
p('Dota 2 [R1] motivates useful-progress shaping and the quartic HP curve. Its numeric weights are game-specific, and some purchasing/ability decisions were scripted. It does not establish optimal reward coefficients for deck building. Avoid copying its HP coefficient of 2 as a prescription for this task.',True)
p('<b>Revise the earlier plan:</b> do not launch deckbuilder until multi-card completion is fixed. Then separate HP reduction and upgrade reward in an ablation; otherwise any improvement cannot be attributed to either change.',True)

page();heading('5 / Model, PPO and compute audit')
fig('optimizer-diagnostics','Thirty-rollout moving averages are descriptive. Each logged episode statistic is itself a rolling window of up to 200 episodes, so the chart points are strongly correlated.')
sub('Model capacity is not the first supported bottleneck')
p('All three policies use the same 1,436,552-parameter Transformer (width 128, three layers, four heads). There is no architecture ablation. The model already reaches the boss, while specific confirmation and campfire behaviors fail. Doubling width first would cost more without addressing those demonstrated mechanisms.')
p('Choice observations already include selected-card flags, selected count, confirmation availability and maximum count. The problem is not simply missing selected-state information. Useful additions to test are explicit choice purpose (discard/upgrade/remove/transform) and minimum count, plus a dedicated confirmation head or an atomic-set decision. Confirm currently reuses the End Turn head. Log truncation and unknown-ID rates before increasing observation capacities (40 choice cards, 60 pile cards, 16 options, 12 powers).')
sub('Optimization and horizon checks')
p('The custom trainer computes value loss as 0.5 x MSE and then multiplies by VALUE_COEF=0.5: effective critic coefficient 0.25 x MSE. The installed SB3 PPO computes MSE and multiplies by 0.5. Despite the source comment, they are not identical here. This is a verified implementation difference, not proof that it causes poor play. Test critic weight separately after fixing choice completion [R2].')
p('Learning rate stays at 0.0003; there is no KL target/early-stop telemetry in the custom update. Add approximate KL, explained variance, gradient norm and per-phase entropy. The observed clipping fraction is not an obvious optimizer explosion. LR decay or KL stopping are candidate stability experiments, not diagnosed cures.')
p('Gamma 0.995 has a return-discount horizon of about 200 decisions. A separate 0.998 trial extends that to about 500. GAE lambda remains 0.95; changing gamma does not by itself make the advantage estimator a 500-step estimator. Evaluate delayed deck benefits explicitly [R1].',True)
p('Logged evaluation intervals consume approximately 55% of sparse and 56% of shaped total wall time, including final/selected evaluations. Choice-loop removal and batched evaluation deserve attention before buying more capacity. GPU utilization was not logged as a historical time series, so a hardware bottleneck is not established.',True)

page();heading('6 / Ranked experiment plan and success criteria')
table(['Priority','Action','Evidence / acceptance criterion'],[
 ['P0 - correctness','Fix multi-choice completion; prevent repeated no-op choices; add progress/error categories.','Reproduce seeds 20019 and 20026 for sparse/shaped, complete discard and upgrade prompts, and pass focused transition tests.'],
 ['P0 - measurement','Re-evaluate saved checkpoints with corrected semantics; store results separately.','Report per-phase timeouts and all outcome changes on identical diagnostic seeds. Do not overwrite historical benchmark files.'],
 ['P1 - reward','Run HP x upgrade ablation on the corrected environment.','HP coefficient {1.0, 0.5}; upgrade reward {0, 0.10}; fixed model, PPO, environment seeds and budget.'],
 ['P1 - replication','Use at least three independent model seeds per reward arm.','Example model seeds 42/43/44. Separate environment-seed uncertainty from between-training-run variance.'],
 ['P2 - stability','Test longer horizon, LR decay and critic coefficient independently.','Retain validation-selected checkpoints; track KL, phase entropy and held-out win rate.'],
 ['P3 - architecture','Try dedicated choice/confirm heads or added choice-purpose features before a wider Transformer.','Compare matched compute and environment decisions; justify a capacity change with evidence.']],[79,189,241])
sub('A practical staged budget')
p('First run the four reward arms for 0.5M steps with one shared model seed as an exploratory screen (2M decisions total). Eliminate arms that cannot complete choices or still ignore healthy campfires. Then replicate the promising configuration and baseline with three model seeds, at 1M per run (6M decisions). Extend only credible candidates. This is a screen followed by replication; the screen alone cannot establish superiority.')
p('Checkpoints every 100,000 and validation every 250,000 decisions are sensible starting points. Do not blindly promote the final checkpoint. Use validation win rate as the main selection criterion; boss conversion, timeout rate and campfire behavior diagnose failures. Do not rank models by their differently scaled training rewards.')
sub('Pre-register evaluation before tuning again')
p('The old 20000-20199 holdout has now been inspected for checkpoint comparison and failure diagnosis. Treat it as development data going forward. Reserve a fresh 1000-seed final set (for example 30000-30999) and exclude it from training and selection. The prepared deckbuilder config still uses the old holdout and should be updated before the next experiment.')
p('Report the mean and spread across model seeds, seed-paired outcome differences and confidence intervals [R3]. At a 5% win rate, 200 environment seeds yield a broad interval; more episodes improve environment precision but cannot replace independent training runs. Candidate gates: clearly lower choice-loop timeouts, lower inappropriate full-HP Rest frequency, and an improvement in final win rate without a collapse in low-HP survival. Treat numerical gates as predeclared experiment criteria, not established performance guarantees.',True)

page();heading('7 / Methods, limits and reproducibility')
sub('Dataset and integrity checks')
p(f"The frozen snapshot contains {sum(d['rollouts'] for d in r['runs'].values()):,} rollout records and 45 periodic validation summaries from the three long experiments. Four completed-checkpoint holdout tables contain 800 rows but only 200 unique paired environment seeds. Random-baseline summaries repeat the same 50-seed random policy; they are not three independent baselines. Short smoke tests were excluded from performance comparisons.")
p('The audit checks row counts, distinct seeds, summary win/timeout totals and mutually exclusive success/timeout flags. Inputs are copied once, complete JSONL lines are retained, and SHA-256 hashes are stored in manifest.json. The ongoing survival run is truncated at the snapshot, not treated as complete. Working-tree source copies and hashes identify the implementation used for the probes; the repository contains uncommitted work.')
sub('Statistical methods')
p('Single-model win intervals use Wilson 95% intervals. Comparisons between checkpoints/profiles use 20,000 paired bootstrap resamples of the same 200 seed IDs (analysis RNG seed 1701). Paired win tests use exact two-sided binomial discordance (McNemar exact). P-values are exploratory and unadjusted for multiple comparisons. None of these methods estimates training-seed variance from the one available initialization per reward arm.')
p('The validation curve repeatedly reuses the same 50 seeds; points and rolling training means are correlated. Item/deck end-state associations have survival, route, encounter and choice-failure confounding. No causal card tier list, feature importance model or machine-learning classifier was fitted to the sparse wins. Such outputs would look precise while being unreliable.')
p('The HP probe changes one live campfire state, not a representative state distribution. Confirmation interventions use two selected failure seeds and two saved policies. Their four successful resolutions establish a specific mechanism, not a projected 81% speedup or a new general win rate. Real-game runs are few and simulator/Steam parity is not established; live bugs that were fixed earlier must not be mistaken for model strategy failures.')
sub('Reproduce and inspect')
p('Run scripts/training_audit.py to analyze the frozen snapshot; scripts/training_audit_probes.py for CPU policy probes; scripts/training_audit_interventions.py for the explicitly scripted confirmation diagnostic; and scripts/training_audit_charts.py for figures. build_training_audit_report.py creates this PDF using the bundled document Python runtime. No training parameters or running queue controls were changed by the audit.')
p('Inspect output/training-audit-20261019/analysis.json, evaluation-episodes.csv, policy-probes.json, confirmation-interventions.json and manifest.json. Original benchmark files remain in output/reward-comparison-20261019/runs/. CPU probes used the frozen checkpoints, whose saved metadata confirms 3,002,368 steps for both selected completed models.',True)
sub('Research references')
source('[R1] OpenAI et al. (2019). Dota 2 with Large Scale Deep Reinforcement Learning. Appendix G; section 4.5. https://arxiv.org/abs/1912.06680')
source('[R2] Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO. https://arxiv.org/abs/2005.12729. Local implementation comparison: sts2_env/entity/ppo.py versus installed stable_baselines3/ppo/ppo.py.')
source('[R3] Agarwal et al. (2021). Deep Reinforcement Learning at the Edge of the Statistical Precipice. https://arxiv.org/abs/2108.13264.')

def footer(c,doc):
    c.setStrokeColor(colors.HexColor('#ccd7df'));c.line(43,39,A4[0]-43,39)
    c.setFont('Audit',7.5);c.setFillColor(colors.HexColor('#617589'));c.drawString(43,27,'SpireFormer training audit | 19 October 2026 | frozen evidence, exploratory analysis');c.drawRightString(A4[0]-43,27,str(doc.page))
doc=SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=43,leftMargin=43,topMargin=39,bottomMargin=51,title='SpireFormer training audit - 19 October 2026',author='Codex',pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(PDF)
