"""Build publication figures and LaTeX tables from audited/identified evidence."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def rows(name):
    with (ROOT/'tables'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(fig,name):
    for ext in ('pdf','png'):
        dest=ROOT/'figures'/(name+'.'+ext)
        tmp=dest.with_name(dest.stem+'.building.'+ext)
        fig.savefig(tmp,dpi=240,bbox_inches='tight')
        if tmp.stat().st_size==0:raise RuntimeError('Empty figure: '+name)
        tmp.replace(dest)
    plt.close(fig)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.labelcolor':'#243746','text.color':'#243746'})
blue='#126B82';grey='#82929E';red='#AD3C43'
a=json.loads((ROOT/'reports/FINAL_INDEPENDENT_AUDIT.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(8.1,2.9),sharex=True)
for ax,e,label in zip(axes,a['endpoints'],['Overall Top-1','A61-A120 Top-1','Macro-F1, all 120']):
    d,l,m=[100*e[k] for k in ('delta','one_sided_lcb','margin')]
    ax.axvspan(m,.15,color=blue,alpha=.035);ax.axvline(m,color=red,ls='--',lw=1.3)
    ax.axvline(0,color=grey,lw=.7);ax.plot([l,d],[0,0],color=blue,lw=3)
    ax.plot(d,0,'o',color=blue,ms=6);ax.plot(l,0,'|',color=blue,ms=14,mew=2)
    ax.text(d,.18,f'{d:.4f}',ha='center',color=blue,fontweight='bold')
    ax.text(l,-.22,f'LCB {l:.4f}',ha='center',fontsize=8)
    ax.text(m,-.49,f'Margin {m:.1f}',ha='center',color=red,fontsize=8)
    ax.set(title=label,xlim=(-1.68,.16),ylim=(-.62,.55),yticks=[],xlabel='Stream-3 minus Full-4 (pp)')
    ax.spines['left'].set_visible(False)
fig.suptitle('Final XSet: prespecified non-inferiority criteria',fontsize=11,y=1.03)
fig.tight_layout();save(fig,'FINAL_NONINFERIORITY')

setup=rows('FINAL_PER_SETUP.csv');cl=rows('FINAL_PER_CLASS.csv')
fig,axes=plt.subplots(2,1,figsize=(8.1,6.1),gridspec_kw={'height_ratios':[1,1.15]})
xs=np.arange(16);ds=np.array([100*float(r['delta_top1']) for r in setup]);cols=[blue if int(r['a61_samples']) else grey for r in setup]
axes[0].bar(xs,ds,color=cols,width=.72);axes[0].axhline(0,color='#243746',lw=.8)
axes[0].set_xticks(xs,[f'S{int(r["setup"]):02d}' for r in setup]);axes[0].set(ylabel='Top-1 difference (pp)',title='(a) Setup effects; blue setups support A61-A120',ylim=(-.73,.49))
for i in [8,10]:axes[0].text(i,ds[i]+(.035 if ds[i]>0 else -.04),f'{ds[i]:+.3f}',ha='center',va='bottom' if ds[i]>0 else 'top',fontsize=8)
x=np.arange(1,121);d=np.array([100*float(r['delta_recall']) for r in cl])
axes[1].axvspan(60.5,120.5,color=blue,alpha=.06);axes[1].bar(x,d,color=[grey if v<=60 else blue for v in x],width=.82);axes[1].axhline(0,color='#243746',lw=.8)
axes[1].set(xlim=(.2,120.8),ylim=(-3.15,3.85),xlabel='Action ID',ylabel='Recall difference (pp)',title='(b) Class effects; 61 lower, 22 unchanged, 37 higher')
axes[1].set_xticks([1,10,20,30,40,50,60,70,80,90,100,110,120])
for i in [30,73]:axes[1].annotate(f'A{i:03d}: {d[i-1]:+.3f}',xy=(i,d[i-1]),xytext=(i+(8 if i==73 else 9),d[i-1]+(-.42 if i==73 else .32)),fontsize=8,ha='center',arrowprops={'arrowstyle':'-','color':grey})
fig.tight_layout(h_pad=2.0);save(fig,'FINAL_HETEROGENEITY')

hist=rows('HISTORICAL_TEMPORAL_TRADEOFF.csv');lat=json.loads((ROOT/'evidence/PREVIOUS_LATENCY_AUDIT.json').read_text())['latency']
fig,ax=plt.subplots(1,2,figsize=(8.1,3.35),gridspec_kw={'width_ratios':[1.2,1]})
for r in hist:
    k=r['variant'].split('_')[0].upper();v=float(r['mean_logical_read_reduction_pct']);d=float(r['delta_all_top1_pp'])
    ax[0].scatter(v,d,color=red if k=='K32' else grey,s=40,zorder=3)
    ax[0].annotate(k,(v,d),xytext=(5,7),textcoords='offset points',fontsize=9)
ax[0].axhline(0,lw=.7,color=grey);ax[0].set(xlabel='Logical skeleton-read reduction (%)',ylabel='Overall Top-1 difference (pp)',title='(a) Temporal reduction, seed 271828',xlim=(-4,104),ylim=(-9.5,1.3))
names=['Warm reuse','Fresh mapping'];values=[100*lat[k]['saving'] for k in ('warm_reuse','fresh_mapping')]
ax[1].bar(np.arange(2),values,color=blue,width=.58);ax[1].axhline(20,color=red,ls='--',lw=1.2)
for i,v in enumerate(values):ax[1].text(i,v+.65,f'{v:.2f}%',ha='center',color=blue,fontweight='bold')
ax[1].text(.96,20.55,'20% requirement',ha='right',transform=ax[1].get_yaxis_transform(),color=red,fontsize=8,bbox={'facecolor':'white','edgecolor':'none','pad':1.5})
ax[1].set_xticks(np.arange(2),names);ax[1].set(ylim=(0,30),ylabel='Paired latency saving (%)',title='(b) Stream-3, seed 271828')
fig.tight_layout(w_pad=2.0);save(fig,'TEMPORAL_AND_STREAM_EVIDENCE')

out=ROOT/'manuscript'
lines=[]
for e,label in zip(a['endpoints'],['Overall Top-1','A61--A120 Top-1','Macro-F1, all 120']):
    lines.append(label+' & '+' & '.join(f'{100*e[k]:.4f}' for k in ('reference','candidate','delta','one_sided_lcb','margin'))+r' \\')
(out/'final_endpoint_rows.tex').write_text('\n'.join(lines)+'\n')
lines=[]
for r in setup:
    lines.append(f"S{int(r['setup']):02d} & {r['samples']} & {r['a61_samples']} & {100*float(r['reference_top1']):.3f} & {100*float(r['candidate_top1']):.3f} & {100*float(r['delta_top1']):+.3f} & {r['lost_correct']} & {r['gained_correct']}"+r' \\')
(out/'setup_rows.tex').write_text('\n'.join(lines)+'\n')
lines=[]
for r in cl:
    lines.append(f"{r['action_id']} & {r['support']} & {100*float(r['reference_recall']):.3f} & {100*float(r['candidate_recall']):.3f} & {100*float(r['delta_recall']):+.3f} & {100*float(r['reference_f1']):.3f} & {100*float(r['candidate_f1']):.3f}"+r' \\')
(out/'class_rows.tex').write_text('\n'.join(lines)+'\n')
for prefix,columns,header,long in [
    ('final_endpoint','lrrrrr','Endpoint & Full-4 & Stream-3 & Difference & Lower bound & Margin',False),
    ('setup','lrrrrrrr',r'Setup & $n$ & A61 $n$ & Full-4 & Stream-3 & Difference & Lost & Gained',True),
    ('class','lrrrrrr',r'Action & $n$ & Recall Full-4 & Recall Stream-3 & Difference & F1 Full-4 & F1 Stream-3',True)]:
    env='longtable' if long else 'tabular'
    head='\\toprule\n'+header+' \\\\\n\\midrule\n'
    text='\\begin{'+env+'}{'+columns+'}\n'+head
    if long:text+='\\endfirsthead\n'+head+'\\endhead\n'
    text+=(out/(prefix+'_rows.tex')).read_text()+'\\bottomrule\n\\end{'+env+'}\n'
    (out/(prefix+'_table.tex')).write_text(text)
print('Created 3 PDF/PNG figure pairs and 3 exact LaTeX table fragments')
