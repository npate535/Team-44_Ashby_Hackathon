// Team 44 triage worklist. Reads cached JSON only (data/cases.json).
// Case shape follows the plan's frozen case record; fields like tier, risk, rd, set, ood
// stand in for the model output contract.
var TIERS={urgent:0,unsure:1,quality:2,routine:3};
var LAB={urgent:"Urgent",unsure:"AI unsure",quality:"Quality",routine:"Routine, not cleared"};
var C=[];
var ST={tab:"w",mode:"ai",sel:null,pause:false,cancel:{},dec:{},hide:false};
var used=function(c){return c.el/c.tg};
var aged=function(c){return used(c)>=.75&&c.lane!=="frozen"};
function sorted(){
 var a=C.slice();
 if(ST.mode==="arrival")return a.sort(function(x,y){return y.el-x.el});
 return a.sort(function(x,y){
  var fx=x.lane==="frozen",fy=y.lane==="frozen";
  if(fx!==fy)return fx?-1:1;
  if(fx)return (x.tg-x.el)-(y.tg-y.el);
  var ax=aged(x),ay=aged(y);
  if(ax!==ay)return ax?-1:1;
  if(TIERS[x.tier]!==TIERS[y.tier])return TIERS[x.tier]-TIERS[y.tier];
  return used(y)-used(x);
 });
}
function fmt(m){m=Math.max(0,Math.round(m));if(m<120)return m+" min";var h=m/60;return (h<48?h.toFixed(1)+" h":(h/24).toFixed(1)+" d")}
function rem(c){var r=c.tg-c.el;return r<0?"Overdue "+fmt(-r):fmt(r)+" left"}
function rng(s){return function(){s=(s*9301+49297)%233280;return s/233280}}
function mosaic(c){
 var r=rng(c.seed*97+1),o="",n=128,k=Math.round(n*c.tf*1.6);
 var cx=3+Math.floor(r()*10),cy=1+Math.floor(r()*6),cells=[];
 for(var i=0;i<n;i++){var x=i%16,y=Math.floor(i/16),d=Math.hypot(x-cx,y-cy);cells.push({i:i,d:d+r()*2.5})}
 var hot={};cells.sort(function(a,b){return a.d-b.d}).slice(0,k).forEach(function(e){hot[e.i]=1});
 for(var j=0;j<n;j++){var s=hot[j]?.55+r()*.45:r()*.2;if(c.tier==="quality")s=r()*.12;
  o+='<i style="background:color-mix(in srgb,var(--eos) '+Math.round(s*100)+'%,var(--bl))" title="Tile '+(j+1)+' tumor score '+s.toFixed(2)+'"></i>'}
 return o;
}
function acts(c){
 var a=[];
 if(c.lane==="frozen"){
  a.push({id:"OR-1",t:"Escalate timer, page pathologist at 5 min left",why:"Frozen lane: no stains, no time. Notice only.",auto:"Automatic notice"});
  a.push({id:"BLOCKED",t:"preorder_ihc(MMR panel)",why:"Guard rejected: no standing order allows stains in the frozen lane.",blk:1});
  return a;
 }
 if(c.tier==="urgent"){
  if(c.scarce)a.push({id:"CRC-2",t:"Propose MMR without cutting",why:"Tissue is scarce. Save it for molecular testing.",auto:"Pathologist one-click"});
  else a.push({id:"CRC-1",t:"Pre-order MMR stains (MLH1, PMS2, MSH2, MSH6)",why:"Urgent colon case above the stain threshold with enough tissue.",auto:"Automatic, cancel window open"});
  a.push({id:"CRC-3",t:"Open draft colorectal checklist",why:"AI may suggest tumor present and histologic type only. Depth, margins, nodes and grade stay blank.",auto:"Draft only"});
 }
 if(c.tier==="unsure")a.push({id:"GEN-1",t:"Request deeper levels",why:"Readers disagree. Deeper levels, no stains.",auto:"Pathologist one-click"});
 if(c.tier==="quality")a.push({id:"GEN-2",t:"Request re-scan",why:"Most tiles are out of focus or debris.",auto:"Automatic, non-diagnostic"});
 return a;
}
function row(c){
 var u=Math.min(100,used(c)*100),hot=c.lane==="frozen"?(c.tg-c.el<=5):aged(c);
 var tags=[];if(aged(c)&&ST.mode==="ai")tags.push('<div class="aged">Aged: promoted at 75% of target</div>');
 var n=acts(c).filter(function(a){return !a.blk&&!ST.cancel[c.id+a.id]}).length;
 return '<tr class="row" tabindex="0" data-id="'+c.id+'"><td><b>'+c.id+'</b><div class="sub">'+c.q+'</div>'+tags.join("")+'</td><td>'+(c.lane==="frozen"?"Frozen section":c.lane==="biopsy"?"Biopsy":"Resection")+'<div class="sub">'+c.site+(c.room?" · "+c.room:"")+'</div></td><td><span class="tier t-'+c.tier+'">'+LAB[c.tier]+'</span><div class="sub">'+(c.risk===null?"no score":c.risk+"% calibrated")+'</div></td><td><span class="timer'+(hot?" hot":"")+'">'+rem(c)+'</span><div class="pbar"><i class="'+(hot?"hot":"")+'" style="width:'+u+'%"></i></div></td><td>'+(ST.dec[c.id]?"Signed out":n?n+" agent action"+(n>1?"s":""):"None")+'</td></tr>';
}
function worklist(){
 var h='<div class="bar"><div class="seg" role="group" aria-label="Sort order"><button data-m="arrival" aria-pressed="'+(ST.mode==="arrival")+'">Arrival order</button><button data-m="ai" aria-pressed="'+(ST.mode==="ai")+'">AI priority</button></div><div class="note">'+(ST.mode==="ai"?"Frozen sections by minutes left, then aged cases, then tier, then share of target used.":"Read roughly first come, first served.")+'</div></div>';
 h+='<div class="tblw"><table><thead><tr><th>Case</th><th>Lane</th><th>AI tier</th><th>Timer</th><th>Agent</th></tr></thead><tbody>'+sorted().map(row).join("")+'</tbody></table></div>';
 h+='<div class="card" style="margin-top:14px"><h2>Tasks</h2>'+tasks()+'</div>';
 return h;
}
function tasks(){
 var t=[];
 C.forEach(function(c){
  if(ST.dec[c.id])return;
  if(c.lane==="frozen")t.push("Read "+c.room+" frozen ("+c.id+"), "+rem(c));
  acts(c).forEach(function(a){if(!a.blk&&a.auto&&a.auto.indexOf("cancel")>-1&&!ST.cancel[c.id+a.id])t.push("Confirm or cancel pre-ordered MMR stains, "+c.id)});
  if(c.tier==="quality")t.push("Re-scan requested, "+c.id+" ("+c.ood+")");
 });
 return t.length?'<ul style="margin:0;padding-left:18px">'+t.map(function(x){return "<li>"+x+"</li>"}).join("")+"</ul>":'<div class="note">No open tasks. New tasks appear when the agent or a timer creates them.</div>';
}
function detail(c){
 var A=acts(c),dec=ST.dec[c.id],look=[["TUM","t"],["TUM","t"],["TUM","t"],["STR","s"],["TUM","t"]];
 if(c.tier==="routine"||c.tier==="quality")look=[["NORM","n"],["LYM","n"],["NORM","n"],["MUC","n"],["ADI","n"]];
 var concepts=c.tier==="urgent"?["malignant glands with nuclear atypia","desmoplastic stroma","necrotic debris"]:c.tier==="quality"?["blurred tissue","background glass"]:c.tier==="unsure"?["crowded glands","reactive atypia","inflamed mucosa"]:["normal crypts","lymphoid aggregate","adipose tissue"];
 var h='<button class="back" id="bk">Back to worklist</button><div class="grid"><div>';
 h+='<div class="card"><h2>'+c.id+' · '+c.q+'</h2><div class="sub">'+c.lane+' · '+c.site+' · '+rem(c)+'</div>';
 var hide=ST.hide&&!dec;
 if(hide)h+='<p class="note">AI opinion hidden. Record your first impression to reveal the tier, score and evidence.</p><div class="btns"><button class="btn p" data-r="1">Benign</button><button class="btn p" data-r="1">Malignant</button><button class="btn p" data-r="1">Unsure</button></div>';
 else{
  h+='<h3>Tile mosaic, colored by tumor score</h3><div class="mos">'+mosaic(c)+'</div><div class="legend">low<b></b>high</div>';
  h+='<h3>Concept words (what tiles resemble, not causes)</h3><div class="chips">'+concepts.map(function(x){return '<span class="chip">'+x+"</span>"}).join("")+'</div>';
  h+='<h3>Five look-alike labeled reference tiles</h3><div class="look">'+look.map(function(l){return '<div><i style="background:color-mix(in srgb,var(--'+(l[1]==="t"?"eos":l[1]==="s"?"hema":"mut")+') '+(l[1]==="n"?30:55)+'%,var(--bl))"></i>'+l[0]+'</div>'}).join("")+'</div>';
 }
 h+='</div></div><div>';
 h+='<div class="card"><h2>AI read</h2><span class="tier t-'+c.tier+'">'+LAB[c.tier]+'</span>';
 if(!hide)h+='<div style="margin-top:8px"><div class="kv"><span>Calibrated risk</span><b>'+(c.risk===null?"n/a":c.risk+"%")+'</b></div><div class="kv"><span>Conformal set (5% error)</span><b>'+c.set+'</b></div><div class="kv"><span>Readers agree</span><b>'+c.rd+'</b></div><div class="kv"><span>Input check</span><b>'+c.ood+'</b></div><div class="kv"><span>Stamp</span><span class="sub">CONCH · mock v0 · preprocessing mock</span></div></div>';
 if(c.tier==="routine")h+='<p class="note">Routine means read in turn. It does not mean cleared.</p>';
 h+='</div><div class="card" style="margin-top:14px"><h2>Agent actions</h2>';
 if(!A.length)h+='<div class="note">No actions. The agent never acts on routine cases.</div>';
 A.forEach(function(a){
  var k=c.id+a.id,off=ST.cancel[k];
  h+='<div class="act'+(a.blk?" blk":"")+(off?" off":"")+'"><b>'+a.id+'</b> '+a.t+'<div class="sub">'+a.why+(a.auto?" · "+a.auto:"")+'</div>'+(a.blk||off?"":'<div class="btns" style="margin-top:6px"><button class="btn d" data-c="'+k+'">Cancel</button></div>')+'</div>';
 });
 h+='</div><div class="card" style="margin-top:14px"><h2>Pathologist approval</h2><div class="note">The only place a diagnosis is made.</div>';
 if(dec)h+='<div class="verdict">'+dec+'</div>';
 else h+='<div class="btns"><button class="btn p" data-d="Agreed with AI tier. Signed out.">Agree and sign out</button><button class="btn" data-d="Overridden: wrong tier.">Override: wrong tier</button><button class="btn" data-d="Overridden: stain not needed.">Override: stain not needed</button></div>';
 h+='</div></div></div>';
 return h;
}
function orboard(){
 var f=C.filter(function(c){return c.lane==="frozen"}),S=["Received","Cutting","Being read","Result called"];
 return '<div class="or">'+f.map(function(c){
  var done=ST.dec[c.id];var st=done?3:c.step;
  return '<div class="card"><div class="sub">'+c.room+' · surgeon view</div><div class="big">'+(done?"Called":rem(c).replace(" left",""))+'</div><div class="steps">'+S.map(function(_,i){return '<span class="'+(i<st||(done&&i===3)?"done":i===st?"on":"")+'"></span>'}).join("")+'</div><b>'+S[st]+'</b><div class="sub">'+c.id+' · '+c.q+'</div></div>'}).join("")+
 '<div class="card"><div class="sub">OR-2 · surgeon view</div><div class="big">Idle</div><div class="note">No specimen sent.</div></div></div>';
}
function draw(){
 if(!C.length)return;
 document.getElementById("sim").textContent="+"+Math.floor(C[0].el)+" min";
 var h=ST.sel?detail(C.filter(function(c){return c.id===ST.sel})[0]):ST.tab==="w"?worklist():orboard();
 document.getElementById("app").innerHTML=h;
 document.getElementById("tw").setAttribute("aria-selected",ST.tab==="w");
 document.getElementById("to").setAttribute("aria-selected",ST.tab==="o");
}
document.getElementById("tw").onclick=function(){ST.tab="w";ST.sel=null;draw()};
document.getElementById("to").onclick=function(){ST.tab="o";ST.sel=null;draw()};
document.getElementById("pp").onclick=function(e){ST.pause=!ST.pause;e.target.setAttribute("aria-pressed",ST.pause);e.target.textContent=ST.pause?"Resume clock":"Pause clock";};
document.getElementById("app").addEventListener("click",function(e){
 var t=e.target,r=t.closest("tr.row");
 if(t.dataset.m){ST.mode=t.dataset.m}
 else if(t.dataset.c){ST.cancel[t.dataset.c]=1}
 else if(t.dataset.d){ST.dec[ST.sel]=t.dataset.d}
 else if(t.dataset.r){ST.hide=false}
 else if(t.id==="bk"){ST.sel=null}
 else if(r){ST.sel=r.dataset.id}
 else return;
 draw();
});
document.getElementById("app").addEventListener("keydown",function(e){
 var r=e.target.closest&&e.target.closest("tr.row");
 if(r&&(e.key==="Enter"||e.key===" ")){e.preventDefault();ST.sel=r.dataset.id;draw()}
});
setInterval(function(){if(ST.pause)return;C.forEach(function(c){c.el+=1});
 var a=document.activeElement;if(a&&a.tagName==="BUTTON"&&ST.sel)return;
 if(!ST.sel||ST.tab==="o")draw();else draw();},1000);
fetch("data/cases.json")
 .then(function(r){if(!r.ok)throw new Error(r.status);return r.json()})
 .then(function(d){C=d;draw()})
 .catch(function(){document.getElementById("app").innerHTML='<div class="card"><h2>Cases did not load</h2><p>Serve the app folder over HTTP (see README), then reload. Opening index.html directly blocks the fetch.</p></div>'});

