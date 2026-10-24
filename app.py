"""FreeAgent dashboard. All results come from the saved modelling service."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from freeagent.data.ingestion import load_data
from freeagent.services.analysis_service import load,evaluate,standings,verify_dataset
from freeagent.valuation.displacement import Displacement,minutes_sensitivity
from freeagent.valuation.net_value import Costs,net_value,financial_sensitivity
from freeagent.cli import serializable
from freeagent.config import load_config
from freeagent.valuation.alternatives import alternatives

settings=load_config()

st.set_page_config(page_title='FreeAgent | Buying Points, Not Names',page_icon='⚽',layout='wide')
st.markdown('''<style>
.block-container {padding-top:5rem;max-width:1440px} h1 {letter-spacing:-1.8px}
[data-testid="stMetric"] {background:#112337;border:1px solid #24394b;padding:18px;border-radius:14px}
[data-testid="stMetricValue"] {color:#c6f65d} .eyebrow {color:#c6f65d;letter-spacing:4px;font-size:12px}
</style>''',unsafe_allow_html=True)
st.markdown('<p class="eyebrow">FREEAGENT / FOOTBALL INTELLIGENCE</p>',unsafe_allow_html=True)
st.title('Buying Points, Not Names.')
st.caption('Transfer research • Displacement-aware sporting value • Explicit financial assumptions')

@st.cache_data
def read_data(stamp):
    return load_data(Path('data'))

@st.cache_resource
def read_model(stamp):
    return load(Path('models/artifacts/core.joblib'))

page=st.sidebar.radio('Workspace',['Overview','Transfer evaluator','Player impact','Counterfactual analysis','Net Value','Club objectives','Backtesting','Explainability'])
st.sidebar.caption('Research prototype · v0.1.0')
st.sidebar.markdown('[Source: StatsBomb Open Data](https://github.com/statsbomb/open-data)')
st.sidebar.caption('Historical football data. Fees, wages, point prices and future minutes are user assumptions.')
match_path=Path('data/processed/matches.csv')
artifact=Path('models/artifacts/core.joblib')
if not match_path.exists():
    st.info('No historical dataset loaded. Run `.venv/bin/freeagent ingest`, then validate and train. No estimates or verdicts are available.')
    st.stop()
try:
    m,a=read_data((match_path.stat().st_mtime_ns,Path('data/processed/appearances.csv').stat().st_mtime_ns))
    bundle=read_model(artifact.stat().st_mtime_ns) if artifact.exists() else None
    if bundle: verify_dataset(bundle,m,a)
except (ValueError,FileNotFoundError,KeyError) as exc:
    st.error(f'Data or artifact could not be loaded: {exc}'); st.stop()
team_names={**dict(zip(m.home_id,m.home_name)),**dict(zip(m.away_id,m.away_name))} if 'home_name' in m else {t:t for t in set(m.home_id)|set(m.away_id)}

if page=='Overview':
    st.subheader('Research workspace')
    c=st.columns(4)
    for col,label,value in zip(c,['Historical matches','Players','Teams','Model cutoff'],[len(m),a.player_id.nunique(),len(team_names),bundle.metadata['cutoff'] if bundle else 'Not trained']): col.metric(label,value)
    source_label=", ".join(m.source.unique()) if "source" in m else "User CSV; provenance not supplied"
    st.success(f"Historical source: {source_label} · {m.date.min()} to {m.date.max()}")
    st.warning('EPA mapping and financial verdicts are not externally validated. This is a research system, not a causal transfer valuation or investment recommendation.')
    st.write('Coverage',m.groupby(['competition','season']).size().rename('matches').reset_index())
    q=Path('data/processed/quality.json')
    if q.exists():
        with st.expander('Data quality and provenance'): st.json(json.loads(q.read_text()))
    if bundle:
        with st.expander('Training metadata'): st.json(bundle.metadata)
    else: st.warning('Models not trained. Run the training command before evaluating transfers.')
    metrics=Path('reports/backtest/metrics.json')
    if metrics.exists():
        st.subheader('Last saved chronological evaluation')
        scores=json.loads(metrics.read_text())
        st.dataframe(pd.DataFrame({k:scores[k] for k in ['dixon_coles','poisson','frequency']}).T,use_container_width=True)
        st.caption(f"{scores['rows']} held-out matches across chronological folds. Lower scores are better.")
    else: st.info('No successful evaluation report yet.')
    st.stop()

if page=='Backtesting':
    directory=Path('reports/backtest')
    if not (directory/'metrics.json').exists(): st.info('No saved walk-forward report. Run the backtest CLI command.'); st.stop()
    st.subheader('Historical forecasts, evaluated forward in time')
    metrics=json.loads((directory/'metrics.json').read_text())
    st.dataframe(pd.DataFrame({k:metrics[k] for k in ['dixon_coles','poisson','frequency']}).T,use_container_width=True)
    st.dataframe(pd.read_csv(directory/'folds.csv'),use_container_width=True)
    calibration=pd.read_csv(directory/'calibration.csv')
    fig=px.scatter(calibration,x='predicted',y='observed',color='outcome',size='count',range_x=[0,1],range_y=[0,1],title='Reliability by outcome')
    fig.add_shape(type='line',x0=0,y0=0,x1=1,y1=1,line=dict(dash='dash',color='#C6F65D'))
    st.plotly_chart(fig,use_container_width=True)
    predictions=pd.read_csv(directory/'predictions.csv')
    st.plotly_chart(px.scatter(predictions,x='predicted_gd_conditional_minutes',y='observed_gd',color='cutoff',title='Held-out goal difference, conditional on observed minutes'),use_container_width=True)
    st.caption('The player-model diagnostic uses realised minutes as covariates. It is not a pre-match forecast. Match forecasts use only pre-cutoff training data. Causal EPA and financial-label validation are unavailable.')
    with st.expander('Failures / exclusions'): st.json(json.loads((directory/'failures.json').read_text()))
    st.download_button('Download predictions',predictions.to_csv(index=False),'predictions.csv')
    st.stop()

if bundle is None: st.warning('Models not trained. No valuation can be produced.'); st.stop()
impact=bundle.impact.report()

if page=='Player impact':
    st.subheader('Regularised player associations')
    st.caption('Goal difference per 90 minutes, conditional on teammates and team controls. Not causal effects. Bootstrap ranges are uncalibrated.')
    player=st.selectbox('Player',impact.player_id,format_func=lambda p:bundle.impact.names[p])
    r=impact.set_index('player_id').loc[player]
    c=st.columns(3); c[0].metric('Impact / 90',f'{r.impact:+.3f}'); c[1].metric('Historical minutes',f'{r.minutes:,.0f}'); c[2].metric('Eligible',str(r.eligible))
    st.write(f'Role: {r.position} · 90% bootstrap range: {r.lower:+.3f} to {r.upper:+.3f}')
    history=a[a.player_id==player].merge(m[['match_id','date']],on='match_id')
    history=history[history.date<bundle.metadata['cutoff']]
    st.plotly_chart(px.bar(history,x='date',y='minutes',color='team_id',title='Training-period appearance minutes'),use_container_width=True)
    st.dataframe(impact.sort_values('impact',ascending=False),use_container_width=True)
    st.json(bundle.impact.diagnostics)
    if r.eligible:
        with st.expander('Similar players with the same observed role'):
            st.dataframe(alternatives(impact,player),use_container_width=True)
            st.caption('Similarity uses impact, minutes and uncertainty. Costs are unavailable; this is not a purchase ranking.')
    st.stop()

if page=='Transfer evaluator':
    st.subheader('Build a transfer counterfactual')
    st.info('Historical scenario on real football data. Financial inputs and playing time are assumptions, not observed transfer facts.')
    season=st.selectbox('Target season',sorted(m.season.unique()))
    sm=m[m.season==season]
    future=sm[sm.date>=bundle.metadata['cutoff']][['match_id','date','home_id','away_id']]
    past=sm[sm.date<bundle.metadata['cutoff']]
    if future.empty: st.warning('No fixtures remain after this model cutoff. Train an earlier model.'); st.stop()
    club=st.selectbox('Destination club',bundle.match.teams,format_func=lambda t:team_names.get(t,t))
    nf=int(((future.home_id==club)|(future.away_id==club)).sum())
    training_apps=a[a.match_id.isin(past.match_id)]
    eligible=set(impact[impact.eligible].player_id)
    roster=bundle.metadata.get('roster_inferred_last_appearance',{})
    incumbents=sorted({p for p,t in roster.items() if t==club}&eligible) if roster else sorted(set(training_apps[training_apps.team_id==club].player_id)&eligible)
    candidates=sorted(eligible-set(incumbents))
    if not incumbents or not candidates: st.warning('Insufficient eligible players for this club.'); st.stop()
    incumbent=st.selectbox('Displaced incumbent',incumbents,format_func=lambda p:bundle.impact.names[p])
    incoming=st.selectbox('Incoming player',candidates,format_func=lambda p:bundle.impact.names[p])
    st.caption(f"Historical roles: {bundle.impact.positions[incoming]} → {bundle.impact.positions[incumbent]}. Confirm the sporting replacement assumption; no tactical-fit score is inferred.")
    with st.form('scenario'):
        left,right=st.columns(2)
        with left:
            incumbent_minutes=st.number_input('Incumbent baseline minutes',0,nf*90,nf*90,90)
            minutes=st.number_input('Incoming expected minutes',0,nf*90,nf*60,30)
            objective=st.selectbox('Club objective',['Expected points','Title','Qualification','Avoid relegation'])
            q=st.number_input('Qualification places',0,len(bundle.match.teams),min(settings['qualification_places'],len(bundle.match.teams)))
            rel=st.number_input('Relegation places',0,len(bundle.match.teams)-1,min(settings['relegation_places'],len(bundle.match.teams)-1))
            low=st.slider('Minimum availability fraction (scenario)',0.,1.,1.,.05)
        with right:
            fee=st.number_input('Transfer fee (€)',0.,step=100000.)
            wage=st.number_input('Annual wage (€)',0.,step=100000.)
            years=st.number_input('Contract years',1.,8.,4.,.5)
            horizon=st.number_input('Cost horizon in years',.01,1.,round(nf/max(1,int(((sm.home_id==club)|(sm.away_id==club)).sum())),2),.01,help='Match the benefit horizon. Default prorates one season by remaining club fixtures.')
            ppp=st.number_input('Assumed price per league point (€)',0.,step=100000.)
            reason=st.text_input('Basis for price per point',placeholder='Your documented budget or revenue scenario')
            sims=st.select_slider('Simulations per parameter draw',[100,250,500,1000],value=250)
        run=st.form_submit_button('Evaluate signing',type='primary')
    if run:
        if not reason.strip() or ppp<=0: st.error('Enter a positive price per point and its justification before generating a verdict.')
        else:
            try:
                with st.spinner('Simulating paired scenarios across parameter draws…'):
                    result=evaluate(bundle,future,club,Displacement(incoming,incumbent,minutes,incumbent_minutes,nf),Costs(fee,wage,years,horizon_years=horizon),ppp,n=sims,qualification=q,relegation=rel,initial=standings(past,bundle.match.teams),availability=(low,1.),seed=settings['seed'],tolerance=settings['financial_tolerance'],threshold=settings['verdict_probability'])
                result['minutes_sensitivity']=minutes_sensitivity(bundle.impact,bundle.match,future,club,Displacement(incoming,incumbent,minutes,incumbent_minutes,nf))
                st.session_state['result']=result
                st.session_state['assumptions']={'club':team_names.get(club,club),'incoming':bundle.impact.names[incoming],'incumbent':bundle.impact.names[incumbent],'objective':objective,'price_per_point':ppp,'justification':reason,'costs':dict(fee=fee,annual_wage=wage,contract_years=years,horizon_years=horizon)}
                st.success('Analysis saved for the Counterfactual, Net Value, Club objectives and Explainability screens.')
                st.metric('Expected points added',f"{result['epa_mean']:+.2f}")
            except (ValueError,RuntimeError) as exc: st.error(str(exc))
    st.stop()

if 'result' not in st.session_state: st.info('Run a transfer evaluation first. No default or fabricated results are shown.'); st.stop()
r=st.session_state['result']; assumptions=st.session_state['assumptions']
st.caption(f"Historical scenario: {assumptions['incoming']} → {assumptions['club']} · replacing {assumptions['incumbent']} · cutoff {r['metadata']['cutoff']}")
if page=='Counterfactual analysis':
    st.subheader('The value of the change')
    c=st.columns(3); c[0].metric('Without signing',f"{r['before_points'].mean():.2f} points"); c[1].metric('With signing',f"{r['after_points'].mean():.2f} points"); c[2].metric('Expected points added',f"{r['epa_mean']:+.2f}")
    fig=go.Figure()
    for values,name,color in [(r['before_points'],'Without signing','#4361EE'),(r['after_points'],'With signing','#C6F65D')]: fig.add_histogram(x=values,name=name,opacity=.6,marker_color=color)
    fig.update_layout(barmode='overlay',xaxis_title='Final league points'); st.plotly_chart(fig,use_container_width=True)
    st.plotly_chart(px.histogram(x=r['epa_samples'],labels={'x':'Expected EPA per parameter draw'},color_discrete_sequence=['#30CBEA']),use_container_width=True)
    st.json(r['displacement'])
    st.caption(f"Paired simulation EPA check: {r['simulation_epa_mean']:+.3f}; exact expected EPA: {r['epa_mean']:+.3f}.")
    if 'minutes_sensitivity' in r: st.plotly_chart(px.line(pd.DataFrame(r['minutes_sensitivity']),x='minutes',y='epa',markers=True,title='Minutes sensitivity at fitted point parameters'),use_container_width=True)
    st.caption('Identical fixtures and shared random uniforms; incoming minutes replace incumbent minutes one for one. Other squad minutes remain unchanged.')
elif page=='Net Value':
    f=r['finance']; st.subheader(f"{f['verdict']} · uncertainty-aware verdict")
    c=st.columns(3); c[0].metric('Mean Net Value',f"€{f['mean']/1e6:+.2f}m"); c[1].metric('Positive value probability',f"{f['positive_probability']:.0%}"); c[2].metric('90% scenario range',f"€{f['interval'][0]/1e6:.1f}m — €{f['interval'][1]/1e6:.1f}m")
    st.plotly_chart(px.histogram(x=f['samples']/1e6,labels={'x':'Net Value (€m)'},color_discrete_sequence=['#C6F65D']),use_container_width=True)
    st.write('Verdict probabilities',f['category_probabilities']); st.caption(f['verdict_note'])
    st.write('Accounting costs (€)',f['costs'])
    st.caption(f"Best Buy / Overpaid require ≥{settings['verdict_probability']:.0%} probability above / below a €{settings['financial_tolerance']/1e6:g}m tolerance. Fair includes inconclusive cases. These are research defaults, not validated financial thresholds.")
    st.write('Price-per-point basis:',assumptions['justification'])
    sensitivity=[]
    for factor in [.5,.75,1.,1.25,1.5]:
        price=assumptions['price_per_point']*factor
        sensitivity.append({'price_per_point':price,'mean_net_value':net_value(r['epa_samples'],Costs(**assumptions['costs']),price)['mean']})
    st.plotly_chart(px.line(pd.DataFrame(sensitivity),x='price_per_point',y='mean_net_value',markers=True,title='Price-per-point sensitivity (€)'),use_container_width=True)
    st.plotly_chart(px.line(pd.DataFrame(financial_sensitivity(r['epa_samples'],Costs(**assumptions['costs']),assumptions['price_per_point'])),x='factor',y='mean_net_value',color='parameter',markers=True,title='Fee, wage and point-price sensitivity'),use_container_width=True)
    with st.expander('Match-outcome variability, separately from expected EPA'):
        st.plotly_chart(px.histogram(x=r['predictive_finance']['samples']/1e6,labels={'x':'Coupled realised Net Value (€m)'}),use_container_width=True)
        st.caption('This distribution depends on the chosen coupling between unobservable scenarios. It is not an empirically identified causal distribution.')
elif page=='Club objectives':
    st.subheader(assumptions['objective'])
    table=r['objectives'].loc[['title','qualification','relegation']]
    st.dataframe(table.style.format('{:.1%}'),use_container_width=True)
    st.plotly_chart(px.bar(table[['without','with']].reset_index().melt(id_vars='index'),x='index',y='value',color='variable',barmode='group',color_discrete_sequence=['#4361EE','#C6F65D']),use_container_width=True)
    st.caption('Negative relegation-probability change is beneficial. Qualification places are configured by the user. Exact rank ties use seeded lots after points, GD and goals scored.')
elif page=='Explainability':
    st.subheader('Every component is auditable')
    st.write('1. Ridge estimates a net goal-difference association from signed minutes, opponent/team controls and home advantage.')
    st.write('2. Incoming minus incumbent coefficient × displaced minutes / (90 × remaining fixtures) gives a club goal-difference shift.')
    st.write('3. Half that shift increases scoring; half reduces conceding. This split is an assumption, not separately identified attack and defence skill.')
    st.write('4. Paired simulations provide objective changes and a Monte Carlo EPA check. Exact integration of each fixture’s expected points removes Monte Carlo noise from reported expected EPA.')
    st.write('5. EPA × your price per point − allocated fee and wages produces Net Value.')
    st.json({'minutes':r['displacement'],'mean_GD_shift':float(np.mean(r['goal_difference_shifts'])),'rate_floor_activations':r['rate_floor_count'],'financial_assumptions':assumptions})
    for warning in r['warnings']: st.warning(warning)
    st.download_button('Download full audit report',json.dumps({'assumptions':assumptions,'result':r},default=serializable,indent=2),'freeagent-analysis.json',mime='application/json')
