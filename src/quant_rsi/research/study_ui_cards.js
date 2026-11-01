/* Read-only DOM views. Provider text is never interpreted as HTML. */
const StudyCards = (() => {
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const number = value => {
    if (value == null || typeof value === 'boolean' || String(value).trim() === '') return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
  };
  const fmt = (value, digits = 4) => number(value) === null ? 'N/A' : number(value).toFixed(digits);
  const pct = value => number(value) === null ? 'N/A' : `${(number(value) * 100).toFixed(2)}%`;
  const title = key => String(key).replaceAll('_', ' ');
  const utcTime = value => {
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? 'Invalid recorded time' : `${date.toISOString().slice(0,16).replace('T',' ')} UTC`;
  };
  const valueText = value => value == null ? 'Not provided' : Array.isArray(value) ? value.join(', ') : typeof value === 'object' ? JSON.stringify(value) : String(value);
  const note = text => el('p', 'view-note', text || '');
  const badge = (text, cls = '') => el('span', `view-badge ${cls}`, text);
  function link(url, text) {
    try {
      const parsed = new URL(url);
      if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password) return null;
      const a = el('a', 'source-link', text || parsed.hostname);
      a.href = parsed.href; a.target = '_blank'; a.rel = 'noopener noreferrer';
      return a;
    } catch { return null; }
  }
  function table(headers, rows, caption) {
    const wrap = el('div', 'view-table-wrap'), t = el('table', 'view-table');
    if (caption) t.append(el('caption', '', caption));
    const head = el('thead'), tr = el('tr');
    headers.forEach(h => { const th = el('th', '', h); th.scope = 'col'; tr.append(th); });
    head.append(tr); t.append(head);
    const body = el('tbody');
    rows.forEach(row => { const r = el('tr'); row.forEach(v => {const td = el('td'); td.append(v instanceof Node ? v : document.createTextNode(valueText(v))); r.append(td);}); body.append(r); });
    t.append(body); wrap.append(t); return wrap;
  }
  function metrics(items) {
    const row = el('div', 'view-metrics');
    items.forEach(([label, value, hint, cls]) => {
      const card = el('div', `view-metric ${cls || ''}`);
      card.append(el('span', '', label), el('strong', '', value));
      if (hint) card.append(el('small', '', hint));
      row.append(card);
    });
    return row;
  }
  function pills(items) {
    const list = el('div', 'view-pills');
    (items || []).forEach(item => list.append(badge(item)));
    return list;
  }
  function article(item) {
    const card = el('article', 'news-card');
    const head = el('div', 'news-meta'), source = link(item.url);
    head.append(source || el('span', '', 'Source URL not provided'));
    if (item.status) head.append(badge(item.status, item.status === 'available' ? 'good' : 'caution'));
    card.append(head, el('h3', '', item.title || item.symbol || 'Retained news item'));
    if (item.reason) card.append(note(`Read result · ${title(item.reason)}${item.fetch_attempted === false ? ' · No request sent' : ''}`));
    if (item.final_url && item.final_url !== item.url) {
      const destination = link(item.final_url, 'Final requested URL');
      if (destination) card.append(destination);
    }
    if (item.excerpt) card.append(el('p', 'news-excerpt', item.excerpt));
    if (item.excerpt_shortened) card.append(note('Excerpt shortened; full retained text is in the trace artifact.'));
    const times = el('div', 'news-times');
    times.append(el('span', '', `Published · ${item.published_at || item.published_date || 'not provided'}`));
    if (item.received_at) times.append(el('span', '', `Received · ${item.received_at}`));
    card.append(times);
    if (item.id) {const id = badge(item.id, 'evidence-id'); id.title = item.id; card.append(id);}
    if (item.content_extent) card.append(note(title(item.content_extent)));
    return card;
  }
  function articles(items) {
    const grid = el('div', 'news-grid');
    (items || []).forEach(item => grid.append(article(item)));
    if (!items?.length) grid.append(note('No retained articles for this stage.'));
    return grid;
  }
  function relationships(view) {
    const wrap = el('div'), map = el('div', 'relationship-map'), peers = el('div', 'relationship-peers');
    const center = el('div', 'relationship-target'); center.append(el('strong', '', view.target), el('small', '', 'TARGET')); map.append(center);
    const entries = [
      ...(view.stocks || []).map(symbol => ({symbol, relationship: 'Configured or simulated context', context: true})),
      ...(view.research_candidates || []).map(item => ({...item, context: false}))
    ];
    entries.slice(0, 8).forEach(item => {
      const peer = el('article', `relationship-peer ${item.context ? '' : 'unverified'}`);
      const heading = el('div'); heading.append(el('strong', '', item.symbol), badge(item.context ? 'Context peer' : 'Unverified suggestion', item.context ? '' : 'caution'));
      peer.append(heading, el('p', '', item.relationship || item.name || 'Relationship not specified'));
      if (item.evidence_ids?.length) peer.append(pills(item.evidence_ids));
      peers.append(peer);
    });
    if (!entries.length) peers.append(note('No related companies were supplied.'));
    map.append(peers); wrap.append(map);
    if(view.price_features?.length)wrap.append(el('h3','view-section-title','Related price availability'),table(['Symbol','Status','Reason'],view.price_features.map(p=>[p.symbol,title(p.status),p.reason])));
    if (entries.length > 8) wrap.append(note(`Showing 8 of ${entries.length} relationships; see raw data for all entries.`));
    wrap.append(note(view.reason), note(`Source · ${view.provider || 'not provided'}. Suggestions are not verified business relationships.`));
    return wrap;
  }
  function factor(item, heading = true) {
    const card = el('article', 'factor-card');
    const head = el('div', 'factor-head'); head.append(badge(item.kind || 'factor', item.kind === 'news' ? 'caution' : ''), el('span', 'operator', item.recipe?.op || 'No recipe'));
    card.append(head);if(heading)card.append(el('h3', '', item.factor_id));
    if (item.description) card.append(el('p', 'factor-description', item.description));
    const parameters = Object.entries(item.recipe || {}).filter(([key]) => !['op','expression','inputs'].includes(key));
    if (parameters.length) card.append(table(['Parameter', 'Value'], parameters.map(([key, value]) => [title(key), key === 'window_days' ? `${value} calendar days` : key === 'window' ? `${value} sessions` : valueText(value)])));
    if (item.input_kind) card.append(note(`Input · ${title(item.input_kind)}`));
    if (item.reuse_lifetime) card.append(table(['Event recipe lifetime','Value'],[
      ['First admitted',utcTime(item.reuse_lifetime.created_at)],
      ['Reuse until',utcTime(item.reuse_lifetime.expires_at)],
      ['Original news evidence',(item.reuse_lifetime.evidence_news_ids || []).join(', ')]
    ]));
    if (item.definition_owner) card.append(note(`${title(item.definition_scope)} · ${item.definition_owner} · ${item.definition_id}`));
    if (item.source) card.append(table(['Source','Entity','Field'],[[item.source.source_id,item.source.entity,item.source.field]]));
    if (['financial_expression','business_expression'].includes(item.recipe?.op)) card.append(table(['Input alias','Source factor'],(item.recipe.inputs||[]).map(i=>[i.name,i.factor_id])));
    if (item.recipe?.expression?.nodes) card.append(table(['Node','Operation','Inputs'],item.recipe.expression.nodes.map(n=>[n.name,n.op,n.leaf || (n.inputs||[]).join(', ')])));
    if (item.input_factor_ids) card.append(note(`Bound inputs · ${item.input_factor_ids.join(', ')}`));
    if (item.runnable === false) card.append(badge('Not executable in this path', 'caution'));
    if (item.availability_reason) card.append(note(title(item.availability_reason)));
    if (item.reasons?.length) card.append(note(item.reasons.map(title).join('; ')));
    return card;
  }
  function evidencePath(view) {
    const wrap = el('div', 'evidence-path');
    wrap.append(el('h3', 'view-section-title', 'News → hypothesis → fitted inputs'));
    wrap.append(pills([`${view.citations.length} cited sources`, `${view.news_columns_fitted} news columns fitted`]));
    const rationale = el('details', 'observation-details');
    rationale.append(el('summary', '', 'Read hypothesis and source citations'));
    if (view.hypothesis) rationale.append(el('p', 'hypothesis-text', view.hypothesis));
    if (view.citations.length) rationale.append(table(['Cited source', 'Role'], view.citations.map(c => [link(c.url, c.title) || c.title, 'Hypothesis context'])));
    wrap.append(rationale);
    wrap.append(table(['Factor', 'Input', 'Execution', 'Reason'], view.factors.map(f => [f.factor_id, title(f.input_kind), badge(title(f.status), f.status === 'fitted' ? 'good' : 'caution'), title(f.reason)])));
    const coverage = view.news_coverage || {};
    if (coverage.original_training_rows !== undefined) wrap.append(note(`Headline history coverage · ${coverage.available_training_rows}/${coverage.original_training_rows} training rows · ${coverage.available_evaluation_rows}/${coverage.original_evaluation_rows} development rows. Coverage is measured before factor selection.`));
    wrap.append(note(view.notice));
    return wrap;
  }
  function continuity(view) {
    const wrap = el('div');
    wrap.append(el('h3', 'view-section-title', 'Continuing the previous research'));
    wrap.append(pills(Object.entries(view.counts).map(([status, count]) => `${count} ${title(status)}`)));
    if (!view.entries.length) wrap.append(note('First run in this history. No previous outcome feedback.'));
    else wrap.append(table(['Target', 'Status / reason', 'Predicted', 'Actual', 'Price error', 'Outcome feedback supplied'], view.entries.map(e => [e.target_session, `${title(e.status)} · ${title(e.reason || '')}`, fmt(e.predicted_price), fmt(e.actual_price), fmt(e.price_error), e.provided_as_feedback ? 'Yes' : 'No'])));
    if (view.previous_study_id && view.selected_factor_ids !== null) {
      wrap.append(table(['Selected factors', 'IDs'], [['Previous run', view.previous_factor_ids.join(', ')], ['This run', view.selected_factor_ids.join(', ')]]));
      wrap.append(pills([...view.added.map(f => `+ ${f}`), ...view.removed.map(f => `− ${f}`)]));
      if (!view.added.length && !view.removed.length) wrap.append(note('Selected factor IDs are unchanged from the previous run.'));
    }
    wrap.append(note(view.notice));
    return wrap;
  }
  function changes(items) {
    const list = el('div', 'change-list');
    if (!items?.length) list.append(note('No factor comparison is available.'));
    (items || []).forEach(change => {
      const labels = {added: 'Added', removed: 'Removed', retained: 'Retained', changed: 'Parameters changed', replaced: 'Recipe replacement'};
      const card = el('article', `change-card change-${change.type}`);
      card.append(badge(labels[change.type] || change.type));
      if (change.before && change.after && change.before.factor_id !== change.after.factor_id) {
        card.append(el('h3', '', `${change.before.factor_id} → ${change.after.factor_id}`));
        card.append(note('Different factor IDs; compared because this operator has one old and one new recipe.'));
      } else card.append(el('h3', '', (change.after || change.before).factor_id));
      if (change.before && change.after && change.type !== 'retained') {
        const before = change.before.recipe || {}, after = change.after.recipe || {};
        const keys = [...new Set([...Object.keys(before), ...Object.keys(after)])];
        card.append(table(['Parameter', 'Before', 'After'], keys.map(key => {
          const changed = JSON.stringify(before[key]) !== JSON.stringify(after[key]);
          return [title(key), valueText(before[key]), changed ? el('strong', 'parameter-change', valueText(after[key])) : valueText(after[key])];
        })));
      } else if (change.type !== 'retained') card.append(factor(change.after || change.before, false));
      list.append(card);
    });
    return list;
  }
  const svgEl = (tag, attributes = {}) => {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value)); return node;
  };
  function chart(rows, series, label, unit) {
    const wrap = el('figure', 'data-chart'), caption = el('figcaption', '', label);
    wrap.append(caption);
    const legend = el('div', 'chart-legend');
    series.forEach(s => {const item = el('span', '', s.label); item.style.setProperty('--series', s.color); legend.append(item);});
    wrap.append(legend);
    const values = rows.flatMap(row => series.map(s => number(row[s.key]))).filter(v => v !== null);
    if (!rows.length || !values.length) {wrap.append(note('No usable observations. Missing values are not drawn as zero.')); return wrap;}
    let lo = Math.min(...values), hi = Math.max(...values);const pad = (hi - lo) * .12 || Math.max(Math.abs(hi) * .01, .01);lo -= pad;hi += pad;
    const svg = svgEl('svg', {viewBox: '0 0 860 250', role: 'img', 'aria-label': label});
    const x = i => 64 + i * 776 / Math.max(rows.length - 1, 1), y = v => 204 - (v - lo) * 180 / (hi - lo);
    for (let i = 0; i < 4; i++) {
      const v = lo + (hi - lo) * i / 3, yy = y(v);
      svg.append(svgEl('line', {x1: 64, x2: 840, y1: yy, y2: yy, class: 'chart-grid'}));
      const text = svgEl('text', {x: 54, y: yy + 4, 'text-anchor': 'end'});text.textContent = fmt(v, 2);svg.append(text);
    }
    series.forEach(s => {
      let d = '', connected = false;
      rows.forEach((row, i) => {const v = number(row[s.key]);if(v === null){connected = false;return;}d += `${connected ? 'L' : 'M'}${x(i)},${y(v)} `;connected = true;});
      svg.append(svgEl('path', {d, stroke: s.color, 'stroke-width': 2.2, fill: 'none', 'stroke-dasharray': s.dash || '', 'data-series': s.key}));
      // Isolated observations remain visible even when adjacent values are missing.
      rows.forEach((row, i) => {const v = number(row[s.key]);if(v !== null && number(rows[i-1]?.[s.key]) === null && number(rows[i+1]?.[s.key]) === null)svg.append(svgEl('circle',{cx:x(i),cy:y(v),r:3,fill:s.color}));});
    });
    [0, rows.length - 1].forEach((i, n) => {const t = svgEl('text',{x:x(i),y:234,'text-anchor':n?'end':'start'});t.textContent=rows[i].date;svg.append(t);});
    const cursor = svgEl('line', {x1:64,x2:64,y1:20,y2:205,class:'chart-cursor'});svg.append(cursor);wrap.append(svg);
    const inspector = el('div', 'chart-inspector'); inspector.setAttribute('aria-live', 'polite');
    const slider = el('input', 'chart-slider');slider.type='range';slider.min=0;slider.max=rows.length-1;slider.value=0;slider.step=1;slider.setAttribute('aria-label',`Inspect ${label} observation`);
    function inspect(i) {
      const row = rows[i];cursor.setAttribute('x1',x(i));cursor.setAttribute('x2',x(i));slider.value=i;
      inspector.replaceChildren(el('strong','',row.origin ? `${row.origin} → ${row.date}` : row.date));
      series.forEach(s=>{const item=el('span','',`${s.label} ${fmt(row[s.key])}`);item.style.color=s.color;inspector.append(item);});
    }
    slider.oninput=()=>inspect(Number(slider.value));
    svg.onpointermove=event=>{const box=svg.getBoundingClientRect(),xx=(event.clientX-box.left)*860/box.width;inspect(Math.max(0,Math.min(rows.length-1,Math.round((xx-64)/776*Math.max(rows.length-1,1)))));};
    wrap.append(inspector,slider,note(`${unit} · Move over the chart or use the slider. Missing observations remain gaps.`));inspect(0);
    return wrap;
  }
  function prediction(view) {
    const wrap=el('div'), revised=view.attempt==='revised', initial=view.initial_metrics?.mae_price, current=view.metrics?.mae_price;
    const tiles=revised ? [['Initial MAE',fmt(initial)],['Revised MAE',fmt(current)]] : [['Price MAE',fmt(current)]];
    tiles.push(['Unchanged-price MAE',fmt(view.baseline_metrics?.mae_price)],['Observations',valueText(view.metrics?.sample_count)]);
    wrap.append(metrics(tiles));
    if(view.evidence_path)wrap.append(evidencePath(view.evidence_path));
    if(view.input_usage?.length){wrap.append(el('h3','view-section-title','Inputs actually used'),table(['Source','Status','Executed factors / reason'],view.input_usage.map(u=>[u.source,title(u.status),u.factor_ids?.length?u.factor_ids.join(', '):title(u.reason)])));}
    if(revised && number(initial)!==null && number(current)!==null){const delta=number(current)-number(initial);wrap.append(el('p',`metric-delta ${delta>0?'worse':delta<0?'better':''}`,`MAE change ${delta>0?'+':''}${fmt(delta)} · ${delta>0?'higher error':delta<0?'lower error':'unchanged error'} in reported development metrics`));}
    const series=[{key:'observed',label:'Observed',color:'#e7eff4'}];
    if(revised)series.push({key:'initial',label:'Initial',color:'#83b5ff'});
    series.push({key:'prediction',label:revised?'Revised':'Initial',color:revised?'#62efc5':'#83b5ff'},{key:'baseline',label:'Unchanged price',color:'#d4b97d',dash:'4 5'});
    wrap.append(chart(view.rows,series,'Development price predictions',view.currency || 'Quote-price units'));
    wrap.append(note(`${view.horizon || 'Unspecified'}-session horizon. Aligned by origin and target session; unmatched or conflicting outcomes are not compared.`));
    const data=el('details','observation-details');data.append(el('summary','', 'View observation table'));
    data.append(table(['Origin','Target','Observed',...(revised?['Initial']:[]),revised?'Revised':'Initial','Unchanged price'],view.rows.map(r=>[r.origin,r.date,fmt(r.observed),...(revised?[fmt(r.initial)]:[]),fmt(r.prediction),fmt(r.baseline)])));
    wrap.append(data,note(view.note || 'Retrospective development observations; not independent validation.'));return wrap;
  }
  function feedback(view) {
    const wrap=el('div'), m=view.metrics||{};
    wrap.append(metrics([['Price MAE',fmt(m.mae_price)],['Price RMSE',fmt(m.rmse_price)],['Observations',valueText(m.sample_count)]]));
    wrap.append(el('h3','view-section-title','Largest development errors'));
    if(view.errors.length)wrap.append(table(['Target session','Observed','Predicted','Signed error'],view.errors.map(r=>[r.date,fmt(r.observed),fmt(r.prediction),`${r.error>0?'+':''}${fmt(r.error)}`])));
    else wrap.append(table(['Observation','Value'],Object.entries(view.latest||{}).map(([k,v])=>[title(k),valueText(v)])));
    wrap.append(note('Sorted by absolute error. Signed error = predicted price − observed price. These observations are reused in the next hypothesis.'));return wrap;
  }
  function terminal(view) {
    const wrap=el('div');
    wrap.append(el('h3','view-section-title','Reserved terminal evaluation'),note(`Selected round ${view.selected_round} · ${view.sample_count} reserved observations · same fitted model, no refit`));
    if(view.status!=='completed'){
      wrap.append(note('Unavailable: selected inputs are missing. No model substitution or row dropping.'),table(['Origin','Missing factors'],(view.missing_inputs||[]).map(r=>[r.origin_session,r.factor_ids.join(', ')])),note(view.limitation));return wrap;
    }
    const forecasts=[['Selected model',view.metrics],...Object.entries(view.baselines)];
    wrap.append(table(['Terminal forecast','Price MAE','Price RMSE','Observations'],forecasts.map(([name,m])=>[name,fmt(m.mae_price),fmt(m.rmse_price),m.sample_count])));
    wrap.append(table(['Terminal account','Net return','Sharpe','Max drawdown'],view.accounts.map(a=>[a.label,pct(a.net_return),fmt(a.sharpe_ratio),pct(a.max_drawdown)])));
    wrap.append(chart(view.rows,[{key:'observed',label:'Observed',color:'#e7eff4'},{key:'prediction',label:'Selected model',color:'#62efc5'},{key:'baseline',label:'Unchanged price',color:'#d4b97d',dash:'4 5'}],'Terminal price predictions','Quote-price units'));
    wrap.append(chart(view.nav,view.accounts.map((a,i)=>({key:a.label,label:a.label,color:['#62efc5','#d4b97d'][i]})),'Terminal account NAV','Account currency units'));
    wrap.append(note('Costs included. Terminal accounts start separately; do not concatenate development and terminal NAV. Short-sample Sharpe uses 252 sessions/year and zero risk-free rate.'),note(view.limitation));return wrap;
  }
  function summary(view) {
    const wrap=el('div'), s=view.selection;
    if(s){wrap.append(metrics([['Initial selection score',fmt(s.initial_score)],[s.selected_score!==undefined?'Selected score':'Revised selection score',fmt(s.selected_score!==undefined?s.selected_score:s.revised_score)],['Selected attempt',s.selected_attempt||'Unavailable',s.metric==='sharpe'?'Net strategy Sharpe':'Price MAE']]));wrap.append(note(s.reason));}
    if(view.text)wrap.append(el('p','hypothesis-text',view.text));
    if(view.continuity)wrap.append(continuity(view.continuity));
    (view.evidence_paths||[]).forEach(path=>{
      const details=el('details','observation-details');
      details.open=path.round===(s?.selected_round||1);
      details.append(el('summary','',`Round ${path.round}${path.round===s?.selected_round?' · Selected model':''} · News and fitted inputs`),evidencePath(path));
      wrap.append(details);
    });
    if(view.rounds?.length){
      wrap.append(note(`Stopped: ${view.stop_reason}`),table(['Round','Factors','Price MAE','Net return','Sharpe','Accepted'],view.rounds.map(r=>[r.round,r.factor_ids.join(', '),fmt(r.mae_price),pct(r.net_return),fmt(r.sharpe_ratio),r.accepted?'Yes':'No'])));
    }
    if(view.accounts?.length){
      wrap.append(table(['Account','Net return','Sharpe','Max drawdown'],view.accounts.map(a=>[a.label,pct(a.net_return),number(a.sharpe_ratio)===null?`N/A · ${title(a.risk_statistics?.status||'unavailable')}`:fmt(a.sharpe_ratio),pct(a.max_drawdown)])));
      const colors=['#83b5ff','#62efc5','#cba6ff','#d4b97d'];
      wrap.append(chart(view.nav||[],view.accounts.map((a,i)=>({key:a.label,label:a.label,color:colors[i]})),'Hypothetical account NAV','Account currency units'));
    }
    wrap.append(note('Costs are included in account results. Development feedback and selection reuse these observations; this is not an independent improvement claim.'));
    if(view.terminal)wrap.append(terminal(view.terminal));return wrap;
  }
  function render(host, view, emptyText) {
    let content=el('div');
    if(!view){content=el('p','visual-empty',emptyText || 'Waiting for this stage.');}
    else if(view.kind==='relationships')content=relationships(view);
    else if(view.kind==='news'){
      if(view.query)content.append(el('h3','view-section-title',view.query));
      if(view.note)content.append(note(view.note));
      if(view.query_origin)content.append(badge(title(view.query_origin)));
      content.append(articles(view.items),note(`Showing ${view.items.length} of ${view.count} items. Current news is context, not backdated training evidence.`));
    } else if(view.kind==='hypothesis'){
      content.append(badge('Research hypothesis'),el('p','hypothesis-text',view.text||'No hypothesis text provided.'),el('h3','view-section-title','Candidate factors'),pills(view.factors));
      if(view.feedback_used)content.append(note('This revision uses development feedback.'));
      content.append(el('h3','view-section-title','Supporting evidence'),articles(view.evidence));
    } else if(view.kind==='findings'){
      (view.findings||[]).forEach(f=>{content.append(el('p','hypothesis-text',f.text),articles(f.evidence));});
      if(view.related?.length)content.append(el('h3','view-section-title','Unverified company suggestions'),pills(view.related.map(c=>`${c.symbol} · ${c.relationship}`)));
      if(view.followup)content.append(note(`Follow-up query · ${view.followup}`));
      if(view.recovery_query)content.append(note(`Recovery search · ${view.recovery_query}. ${view.recovery_reason}`));
    } else if(view.kind==='factors'){
      content.append(note(view.catalog?'Factor catalog · inclusion does not mean execution.':'Selected model inputs'),note(view.note));
      const grid=el('div','factor-grid');view.factors.forEach(f=>grid.append(factor(f)));content.append(grid);
      if(view.count>view.factors.length)content.append(note(`Showing ${view.factors.length} of ${view.count} definitions.`));
    } else if(view.kind==='factor_changes'){
      const counts={};view.changes.forEach(c=>counts[c.type]=(counts[c.type]||0)+1);
      content.append(pills(Object.entries(counts).map(([k,v])=>`${v} ${k}`)),note(view.note),changes(view.changes));
    } else if(view.kind==='prediction')content=prediction(view);
    else if(view.kind==='feedback')content=feedback(view);
    else if(view.kind==='summary')content=summary(view);
    else if(view.kind==='continuity')content=continuity(view);
    else if(view.kind==='terminal')content=terminal(view);
    else if(view.kind==='engine_comparison'){
      content.append(note(view.note),table(['Window','Engine','Price MAE','Net return','Sharpe','Max drawdown'],view.rows.map(row=>[row.window,row.engine,fmt(row.mae),pct(row.net_return),fmt(row.sharpe_ratio),pct(row.max_drawdown)])));
    }
    else if(view.kind==='iteration'){
      content.append(metrics([['Round',view.round],['Selected round',view.incumbent_round],['Fits',`${view.completed_fits} / ${view.fit_budget}`]]),note(view.decision?.reason||'Initial baseline retained.'),table(['Metric','Value'],[['Price MAE',fmt(view.development_metrics?.mae_price)],['Net return',pct(view.strategy_metrics?.net_return)],['Sharpe',fmt(view.strategy_metrics?.sharpe_ratio)]]),pills([view.accepted?'Accepted':'Retained incumbent',...(view.factor_changes?.activated||[]).map(f=>`+ ${f}`),...(view.factor_changes?.deactivated||[]).map(f=>`− ${f}`)]));
    }
    else if(view.kind==='facts'){content.append(table(['Field','Value'],Object.entries(view.facts).map(([k,v])=>[title(k),valueText(v)])));if(view.note)content.append(note(view.note));}
    else content.append(note('A visual summary is not available for this stage. Inspect its raw data below.'));
    if(view?.sample_coverage?.length){
      content.append(el('h3','view-section-title','Financial variation on research rows'),
        table(['Factor','Available training rows','Distinct training values','Distinct development values'],view.sample_coverage.map(r=>[r.factor_id,`${r.training_available_rows} / ${r.training_rows}`,r.training_distinct_values,r.development_distinct_values])),
        note('Repeated daily values are not independent financial observations; distinct values are not a count of filings. A constant training feature cannot explain differences between training rows. Coverage does not establish predictive value.'));
    }
    if(view?.coverage_gaps?.length){
      const gaps=el('div','factor-grid');
      view.coverage_gaps.forEach(g=>{
        const card=el('article','factor-card coverage-gap-card');
        card.append(el('h3','',g.factor_id),badge(`${g.missing_required_rows} missing required rows`,'caution'),
          el('p','factor-description',g.description),el('strong','','Next step'),
          el('p','factor-description',g.next_step),note(`Example required cutoff · ${g.example_cutoff}`));
        gaps.append(card);
      });
      content.append(el('h3','view-section-title','Why financial factors are unavailable'),gaps,
        note('One primary reason per missing required row; source failures may coexist. Suggestions do not guarantee coverage. Use a new collection and study; do not fill missing values or move publication dates.'));
    }
    host.replaceChildren(content);
  }
  return {render};
})();
