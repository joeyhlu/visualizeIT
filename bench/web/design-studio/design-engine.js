/* Local finite recipe grammar. No network, model, DOM, or image dependency. */
(function (root, factory) {
  const engine = factory();
  if (typeof module === 'object' && module.exports) module.exports = engine;
  else root.DesignEngine = engine;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const PATTERNS = ['solid', 'stripes', 'checker', 'dots', 'waves', 'grid'];
  const NAMED_COLORS = Object.freeze({cream:'#f4ebd8',ink:'#202820',green:'#42634b',orange:'#dc693f',pink:'#e5a1ac',blue:'#376aa0',red:'#bd3f3f',yellow:'#e9bb42',purple:'#795b9b',black:'#111111',white:'#ffffff',navy:'#203957',teal:'#287e80',brown:'#805b41',gray:'#808080',grey:'#808080'});
  const DEFAULT = Object.freeze({pattern:'stripes',colors:['#f4ebd8','#42634b'],text:'',scale:1,rotation_deg:-25,seed:731,width:1024,height:1024});
  const clone = value => JSON.parse(JSON.stringify(value));
  function normalizeRecipe(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError('Recipe must be an object.');
    const keys = ['pattern','colors','text','scale','rotation_deg','seed','width','height'];
    if (Object.keys(value).some(key => !keys.includes(key)) || keys.some(key => !Object.hasOwn(value,key))) throw new TypeError('Recipe fields do not match the local design format.');
    if (!PATTERNS.includes(value.pattern)) throw new TypeError('Choose a supported pattern.');
    if (!Array.isArray(value.colors) || value.colors.length < 2 || value.colors.length > 5 || value.colors.some(c => typeof c !== 'string' || !/^#[0-9a-f]{6}$/i.test(c))) throw new TypeError('Use two to five six-digit hex colors.');
    if (typeof value.text !== 'string' || Array.from(value.text).length > 80) throw new TypeError('Text must contain at most 80 characters.');
    for (const [key,min,max,integer] of [['scale',.25,4,false],['rotation_deg',-180,180,false],['seed',0,4294967295,true],['width',256,1024,true],['height',256,1024,true]]) {
      if (typeof value[key] !== 'number' || !Number.isFinite(value[key]) || value[key]<min || value[key]>max || (integer && !Number.isInteger(value[key]))) throw new TypeError(`${key} must be ${integer?'an integer':'a finite number'} from ${min} to ${max}.`);
    }
    return {...value,colors:value.colors.map(c=>c.toLowerCase())};
  }
  function parsePrompt(prompt, base = DEFAULT) {
    if (typeof prompt !== 'string' || Array.from(prompt).length > 500) throw new TypeError('Prompt must contain at most 500 characters.');
    const recipe = normalizeRecipe(clone(base));
    const warnings = [];
    let words = prompt;
    const quoted = [...words.matchAll(/"([^"\n]*)"|'([^'\n]*)'/g)];
    if (quoted.length) {
      const text = quoted[0][1] ?? quoted[0][2];
      recipe.text = Array.from(text).slice(0,80).join('');
      if (Array.from(text).length>80) warnings.push('Text was shortened to 80 characters.');
      if (quoted.length>1) warnings.push('Only the first quoted text is used.');
      words = words.replace(/"([^"\n]*)"|'([^'\n]*)'/g,' ');
    } else {
      const textMatch = /\b(?:text|say|saying)\s*:?\s+(.+)$/i.exec(words);
      if (textMatch) {
        recipe.text = Array.from(textMatch[1].trim()).slice(0,80).join('');
        if (Array.from(textMatch[1].trim()).length>80) warnings.push('Text was shortened to 80 characters.');
        words = words.slice(0,textMatch.index);
      }
    }
    words = words.toLowerCase();
    const colorMatches = [...words.matchAll(/#[0-9a-z]+|\b(?:cream|ink|green|orange|pink|blue|red|yellow|purple|black|white|navy|teal|brown|gray|grey)\b/g)];
    const colors = [];
    for (const match of colorMatches) {
      const color = NAMED_COLORS[match[0]] || (/^#[0-9a-f]{6}$/.test(match[0]) ? match[0] : null);
      if (color && !colors.includes(color)) colors.push(color);
      else if (!color) warnings.push(`Ignored invalid color ${match[0]}; use #RRGGBB.`);
    }
    if (colors.length) {
      recipe.colors = colors.slice(0,5);
      if (recipe.colors.length===1) recipe.colors.push(recipe.colors[0]==='#ffffff'?'#202820':'#ffffff');
      if (colors.length>5) warnings.push('Only the first five colors are used.');
    }
    words=words.replace(/#[0-9a-z]+|\b(?:cream|ink|green|orange|pink|blue|red|yellow|purple|black|white|navy|teal|brown|gray|grey)\b/g,' ');
    const aliases={stripe:'stripes',striped:'stripes',checkerboard:'checker',checkered:'checker',dot:'dots',dotted:'dots',wave:'waves',wavy:'waves'};
    const found=[];
    words=words.replace(/\b(solid|stripes?|striped|checker|checkerboard|checkered|dots?|dotted|waves?|wavy|grid)\b/g,word=>{found.push(aliases[word]||word);return ' ';});
    if (found.length) recipe.pattern=found[0];
    if (new Set(found).size>1) warnings.push('Multiple patterns found; the first one is used.');
    words=words.replace(/\b(scale|rotation|rotate|seed)\s*:?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:degrees?|deg)?/g,(all,key,n)=>{
      const field=key==='rotate'||key==='rotation'?'rotation_deg':key;
      const [min,max]=field==='scale'?[.25,4]:field==='seed'?[0,4294967295]:[-180,180];
      const raw=Number(n); const bounded=Math.max(min,Math.min(max,field==='seed'?Math.trunc(raw):raw));
      recipe[field]=bounded;
      if(raw!==bounded) warnings.push(`${field} adjusted to ${bounded} (allowed ${min}–${max}).`);
      return ' ';
    });
    words=words.replace(/\b(?:a|an|the|and|with|in|on|of|make|create|draw|pattern|colors?|background|please|text|say|saying|using)\b/g,' ');
    const unsupportedWords=[...new Set(words.match(/[\p{L}\p{N}#]+/gu)||[])];
    if(unsupportedWords.length) warnings.push(`Unsupported wording: ${unsupportedWords.join(', ')}. Only the listed pattern, color, text and numeric terms affect the artwork.`);
    if(!prompt.trim()) warnings.push('Empty prompt: the current recipe was kept.');
    return {recipe:normalizeRecipe(recipe),warnings,unsupportedWords};
  }
  function seededRandom(seed) {
    let state=seed>>>0;
    return ()=>{state=(state+0x6D2B79F5)>>>0;let t=state;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};
  }
  function draw(canvas, input) {
    const r=normalizeRecipe(input),ctx=canvas.getContext('2d');
    if(!ctx) throw new Error('Canvas drawing is unavailable in this browser.');
    canvas.width=r.width;canvas.height=r.height;
    const rand=seededRandom(r.seed),cell=96*r.scale,extent=Math.hypot(r.width,r.height)/2+cell*2;
    ctx.fillStyle=r.colors[0];ctx.fillRect(0,0,r.width,r.height);
    ctx.save();ctx.translate(r.width/2,r.height/2);ctx.rotate(r.rotation_deg*Math.PI/180);
    const phaseX=rand()*cell,phaseY=rand()*cell;
    ctx.translate(phaseX,phaseY);
    const start=-Math.ceil(extent/cell),end=Math.ceil(extent/cell);
    if(r.pattern==='solid') { /* Color zero is the background. */ }
    else if(r.pattern==='stripes') {
      for(let i=start;i<=end;i++){ctx.fillStyle=r.colors[((i%r.colors.length)+r.colors.length)%r.colors.length];ctx.fillRect(i*cell,-extent*2,cell+1,extent*4);}
    } else if(r.pattern==='checker'||r.pattern==='dots'||r.pattern==='grid') {
      for(let y=start;y<=end;y++)for(let x=start;x<=end;x++) {
        const index=r.pattern==='checker'?((x+y)%r.colors.length+r.colors.length)%r.colors.length:1+Math.floor(rand()*(r.colors.length-1));
        ctx.fillStyle=r.colors[index];ctx.strokeStyle=r.colors[index];
        if(r.pattern==='checker')ctx.fillRect(x*cell,y*cell,cell+1,cell+1);
        if(r.pattern==='dots'){ctx.beginPath();ctx.arc(x*cell,y*cell,cell*(.17+rand()*.13),0,Math.PI*2);ctx.fill();}
        if(r.pattern==='grid'){ctx.lineWidth=Math.max(2,cell*.045);ctx.strokeRect(x*cell,y*cell,cell,cell);}
      }
    } else if(r.pattern==='waves') {
      ctx.lineWidth=cell*.45;
      for(let y=start;y<=end;y++) {
        ctx.strokeStyle=r.colors[1+((y% (r.colors.length-1)+r.colors.length-1)%(r.colors.length-1))];ctx.beginPath();
        for(let x=-extent*2;x<=extent*2;x+=8){const yy=y*cell+Math.sin(x/(cell*.75)+phaseX)*cell*.3;if(x===-extent*2)ctx.moveTo(x,yy);else ctx.lineTo(x,yy);}ctx.stroke();
      }
    }
    ctx.restore();
    if(r.text){
      let size=Math.min(r.width,r.height)*.12;ctx.textAlign='center';ctx.textBaseline='middle';
      ctx.font=`700 ${size}px system-ui, sans-serif`;
      const measured=ctx.measureText(r.text).width;
      if(measured>r.width*.84){size*=r.width*.84/measured;ctx.font=`700 ${size}px system-ui, sans-serif`;}
      const width=ctx.measureText(r.text).width;
      ctx.fillStyle=r.colors[0];ctx.fillRect((r.width-width)/2-size*.28,r.height/2-size*.72,width+size*.56,size*1.44);
      ctx.fillStyle=r.colors[1];ctx.fillText(r.text,r.width/2,r.height/2);
    }
    return r;
  }
  return Object.freeze({PATTERNS,NAMED_COLORS,DEFAULT,normalizeRecipe,parsePrompt,seededRandom,draw});
});
