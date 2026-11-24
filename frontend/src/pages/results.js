import {plot} from '../components/common/chart.js';
import {datasetField, feedback, field} from '../components/common/fields.js';
import {empty, metrics, panel, pill} from '../components/common/panels.js';
import {table} from '../components/common/table.js';
import {REVIEW, STATUS} from '../constants/enums.js';
import {handler, poll} from '../hooks/action.js';
import {session} from '../stores/session.js';
import {workspace} from '../stores/workspace.js';
import {button, el, h, link, p} from '../utils/dom.js';
import {date, value} from '../utils/format.js';

export async function results(route, refresh) {
  const {spec,runs,datasets} = workspace;
  const id = route.id || runs[0]?.id;
  if (!id) return panel(spec.pages[3],empty('还没有可查看的运行结果。','#run','前往运行'));
  const job = await workspace.getRun(id);
  const source = datasets.find(item => item.id === job.dataset_id);
  const message = feedback();
  const finished = item => item.status === 'complete' || item.status === 'failed';
  if (!finished(job)) {
    const original = location.hash;
    const status = el('p',{role:'status'},'后台正在计算，完成后会自动展示。');
    poll(()=>workspace.getRun(id),finished,()=>location.hash===original)
      .then(result=>{if (result && status.isConnected && location.hash===original) refresh();
        else if(status.isConnected) status.textContent='仍在执行，请点击刷新检查。';})
      .catch(error=>{if(status.isConnected) status.textContent=error.message;});
    return panel(job.name,pill(STATUS[job.status],job.status),status,button('刷新',{class:'quiet',onClick:refresh}));
  }
  const download = button('下载结果 CSV',{class:'quiet','data-testid':'export',onClick:event=>
    handler(download,message,()=>workspace.download(`/runs/${job.id}/export`,`${spec.short_name}-${job.id}.csv`))(event)});
  const submit = button('生成复盘',{type:'submit',class:'primary'});
  const history = el('label',{class:'field run-picker',for:'run-picker'},el('span',{},'切换运行记录'),
    el('select',{id:'run-picker',onChange:event=>{location.hash=`#results/${event.target.value}`;}},
      (runs.some(item=>item.id===job.id) ? runs : [job,...runs]).map(item=>el('option',{value:item.id,selected:item.id===job.id},item.name))));
  const result = job.result;
  const analysis = result ? el('section',{class:'analysis','data-testid':'analysis-complete'},
    el('div',{class:'result-head'},p(job.name,'eyeline'),h(2,result.headline),p(result.note,'intro-copy'),
      el('div',{class:'actions'},pill(STATUS[job.status],job.status),download,message)),
    metrics(result.metrics),
    result.charts.map(plot),
    (result.sections || []).map(item=>panel(item.title,p(item.text))),
    result.tables.map(item=>panel(item.title,table(item.columns,item.rows,{caption:item.title}))))
    : panel(job.name,pill(STATUS[job.status],job.status),p(job.error,'feedback'),link('返回调整参数',`#run/${job.dataset_id}`,{class:'quiet'}));
  const inputs = spec.params.map(param=>el('div',{},el('dt',{},param.label),el('dd',{},value(job.params[param.key]))));
  const roster = result?.roster;
  const children = runs.filter(item=>item.parent_id===job.id).sort((a,b)=>b.id-a.id);
  const replays = roster ? (job.parent_id
    ? link(`本记录按原名单运行 ${job.parent_id} 复盘`, `#results/${job.parent_id}`, {class:'quiet'})
    : session.canWrite ? el('form',{class:'fields','data-testid':'replay-form',
      onSubmit:handler(submit,message,async event => {
        const data = new FormData(event.target);
        const item = await workspace.replayRun(job.id,{dataset_id:Number(data.get('dataset_id')),name:data.get('name')});
        location.hash = `#results/${item.id}`;
      })},
      p('用原名单复核最新抽检版本：补传来料与历史更正只更新当前误报、漏报，不改变参评批次和入选结果。','subtle'),
      datasetField(datasets,job.dataset_id), field({key:'name',label:'复盘名称',type:'string',required:false,max_length:120},`${job.name} · 复盘`),
      submit, message)
    : p('按原名单复盘需要分析员或复核主管权限。','subtle')) : null;
  const replayLinks = children.length ? el('ul',{class:'method-list'},children.map(item=>
    el('li',{},link(`${item.name}（${item.status==='complete'?'已完成':STATUS[item.status]}）`,`#results/${item.id}`)))) : null;
  return el('div',{},el('div',{class:'two-col'},el('div',{class:'wide-panel'},analysis),
    el('aside',{class:'sidebar-panel'},panel('运行来源',history,
      source && link(source.name,`#inputs/${source.id}`),p(date(job.finished_at),'subtle'),
      el('dl',{class:'dictionary'},inputs)),
    roster && panel(job.parent_id ? '原名单复盘' : '按原名单复盘', replays, replayLinks),
    result && panel(spec.decision, pill(REVIEW[job.review],job.review),
      link('打开复核记录',`#review/${job.id}`,{class:'quiet'})))));
}
