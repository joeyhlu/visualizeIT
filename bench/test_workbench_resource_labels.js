// Codex-owned actual timing renderer controls; DOM, browser, model and GPU are not used.
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const source=fs.readFileSync('bench/web/workbench/workbench.js','utf8');
const start=source.indexOf('  const CACHE_MODES = '),end=source.indexOf('  function frameUrl',start);
assert(start>=0&&end>start);
const ctx={};vm.runInNewContext(source.slice(start,end)+';this.render=renderTimingProvenance;',ctx);
const roles=['primary_neural','frozen_baseline'];let passed=0;
function display(tel,spec={roles}){const lines={};ctx.render(tel===undefined?{}:{report:{telemetry:tel}},spec,(k,v)=>lines[k]=v);return lines;}
const G='Legacy GPU field; run scope unverified';
const cached={run_cache_mode:'cached',run_wall_ms:11237.9,gpu_used:false,
  cache:Object.fromEntries(roles.map(r=>[r,{status:'cache_hit'}])),
  cached_original_timings:Object.fromEntries(roles.map(r=>[r,{source:'cached_original_run'}])),frames:[]};
let lines=display(cached);assert.match(lines['Per-frame timings'],/historical.*primary_neural, frozen_baseline.*not new inference/);
assert.equal(lines[G],'false (recorded legacy value)');assert.equal(lines['Recorded pipeline wall (excludes outer command validation)'],'11.2 s');passed++;
const cold={run_cache_mode:'cold',gpu_used:true,cache:Object.fromEntries(roles.map(r=>[r,{status:'miss'}])),frames:roles.map(role=>({role}))};
lines=display(cold);assert.match(lines['Per-frame timings'],/from this run/);assert(!lines['Per-frame timings'].includes('historical'));passed++;
const mixed=structuredClone(cached);mixed.run_cache_mode='partially_cached';mixed.cache.frozen_baseline.status='miss';mixed.frames=[{role:'frozen_baseline'}];
lines=display(mixed);assert.match(lines['Per-frame timings'],/historical.*primary_neural\);.*from this run \(frozen_baseline\)/);passed++;
const unsupported=structuredClone(cached);delete unsupported.cached_original_timings;
lines=display(unsupported);assert.match(lines['Per-frame timings'],/^unknown/);passed++;
lines=display(undefined);assert.match(lines['Timing provenance'],/not recorded/);passed++;
for(const wall of [-1,'1000',NaN,Infinity,true])assert.equal(display({...cached,run_wall_ms:wall})['Recorded pipeline wall (excludes outer command validation)'],'not recorded');passed++;
lines=display({...cached,gpu_used:'false',run_cache_mode:'invented'});assert.equal(lines[G],'not recorded');assert.match(lines['Recorded cache mode'],/unknown/);passed++;
lines=display(cached,{});assert.match(lines['Per-frame timings'],/historical/);passed++;
for(const [value,want] of [[true,'true (recorded legacy value)'],[false,'false (recorded legacy value)'],[undefined,'not recorded'],[null,'not recorded'],[0,'not recorded'],[1,'not recorded'],['false','not recorded'],[{},'not recorded']]){
  lines=display({...cached,gpu_used:value});assert.equal(lines[G],want);assert(!('Recorded GPU use' in lines));
  assert(!/GPU idle|no GPU|GPU used:|did not use GPU/i.test(Object.values(lines).join(' ')));
}passed++;
const fields=['process_peak_rss_bytes','gpu_peak_allocated_bytes','gpu_peak_reserved_bytes'];
const labels=['Host process peak RSS (not a per-run peak)','Process CUDA allocator peak allocated (cumulative; epoch unrecorded)','Process CUDA allocator peak reserved (cumulative; epoch unrecorded)'];
for(const value of [0,1024,null,undefined,true,false,-1,1.5,NaN,Infinity,'100',{},2**53]){
  const tel={...cached};fields.forEach(key=>tel[key]=value);lines=display(tel);
  const want=Number.isSafeInteger(value)&&value>=0?`${value} B`:'not recorded';
  for(const label of labels)assert.equal(lines[label],want);
}passed++;
assert.equal(lines['Memory scope'],'Process memory observations do not establish GPU work or per-run memory peaks.');passed++;
console.log(JSON.stringify({passed,failed:0,scope:'11 pure renderer controls, including retained eight cache/timing cases; no browser or inference'}));
