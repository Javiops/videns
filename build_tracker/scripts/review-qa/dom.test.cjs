// Execute the actual static page in a DOM with real click/key event dispatch.
// External resources are disabled; this is interaction QA, not visual rendering.
const assert=require("node:assert/strict");
const fs=require("node:fs");
const path=require("node:path");
const {JSDOM,VirtualConsole}=require("jsdom");
const root=path.resolve(__dirname,"../..");
const canonical=path.resolve(root,"../videns-site");
const site=fs.existsSync(path.join(canonical,"review.html"))?canonical:path.join(root,"site");
const read=(dir,name)=>fs.readFileSync(path.join(dir,name),"utf8");
for(const name of ["review.html","review-playback.js","review-faker-data.js","review-faker-evidence.json"]){
  assert.equal(read(site,name).trimEnd(),read(path.join(root,"site"),name).trimEnd(),`published source drift: ${name}`);
}
assert.equal(read(site,"review-playback.js"),read(path.join(root,"web"),"review-playback.js"));
const errors=[],consoleCapture=new VirtualConsole();
consoleCapture.on("jsdomError",error=>errors.push(error));
const dom=new JSDOM(read(site,"review.html"),{runScripts:"outside-only",url:"https://review.test/review.html",virtualConsole:consoleCapture});
const window=dom.window,document=window.document;
Object.assign(window,{Blob,Response,DecompressionStream});
document.body.dataset.staticDemo="";
window.eval(read(site,"review-playback.js"));
window.eval(read(site,"review-faker-data.js"));
window.eval(read(site,"review.html").match(/<script>([\s\S]*?)<\/script>/)[1]
  + "\nwindow.__reviewEvaluate = expression => eval(expression);");
const q=selector=>document.querySelector(selector);
const evaluate=expression=>window.__reviewEvaluate(expression);
const at=(ts,eventExpression="null")=>evaluate(`setTime(${ts},${eventExpression})`);

(async()=>{
  await evaluate("showDemo()");
  assert.ok(q("#review").classList.contains("ready"),q("#state").textContent);
  assert.match(q("#title").textContent,/Faker.*Cassiopeia/);
  assert.equal(document.querySelectorAll(".player").length,10);
  assert.equal(document.querySelectorAll(".structure").length,30);
  assert.equal(q("#seek"),null,"the map replaces the slider");
  const data=evaluate("data");
  const death=data.events.find(e=>e.kind==="kill"&&e.actors[1]===3);
  const deathButton=document.querySelector(`.ev.kill[data-ts="${death.ts}"]`);
  assert.ok(deathButton,"Faker death can be selected in the event rail");
  deathButton.click();
  assert.equal(evaluate("now"),death.ts);
  assert.equal(q('.marker[data-map-player="3"]'),null,"no living Faker marker at death");
  const corpse=q('.death-marker[data-map-player="3"]');
  assert.ok(corpse,"death location is visible");
  const deathStyle=corpse.getAttribute("style");
  assert.ok(q('.player[data-player="3"]').classList.contains("is-down"));
  assert.equal(q(`.ev.kill[data-ts="${death.ts}"]`).getAttribute("aria-current"),"true");
  at(death.ts+1);
  assert.equal(q('.death-marker[data-map-player="3"]').getAttribute("style"),deathStyle);
  const alive=data.observations.find(f=>f.ts>death.ts&&f.players[3].health>0);
  at(alive.ts-1);
  assert.equal(q('.marker[data-map-player="3"]'),null);
  at(alive.ts);
  assert.ok(q('.marker[data-map-player="3"]'),"positive health restores the marker");
  assert.equal(q('.death-marker[data-map-player="3"]'),null);
  at(death.ts);
  assert.equal(q('.death-marker[data-map-player="3"]').getAttribute("style"),deathStyle,"backward seek restores death");

  q('[data-filter="base"]').click();q("#next").click();
  assert.equal(evaluate("selectedEvent.kind"),"base","step respects the active filter");
  q('[data-filter="all"]').click();
  const visit=data.decisions[6];
  q(`.ev.shop[data-ts="${visit.ts_end}"]`).click();
  assert.equal(evaluate("now"),visit.ts_end,"full purchase is shown at shop completion");
  assert.match(q("#comparison-status").textContent,/comparable|shops/);
  assert.ok(q('#comparison-status a[href="review-faker-evidence.json"]'));
  q('.marker[data-map-player="3"],.death-marker[data-map-player="3"]').click();
  assert.equal(q('.player[data-player="3"]').getAttribute("aria-pressed"),"true");
  const beforeStep=evaluate("now");
  document.dispatchEvent(new window.KeyboardEvent("keydown",{key:"ArrowRight",bubbles:true}));
  assert.ok(evaluate("now")>beforeStep,"keyboard stepping works");

  const first=data.decisions[0];
  if(first.ts<first.ts_end){at(first.ts);assert.equal(q("#actual-buy").querySelectorAll("img").length,0);}
  at(first.ts_end);assert.equal(q("#actual-buy").querySelectorAll("img").length,first.bought.length);
  const inventory=document.createElement("div");
  inventory.innerHTML=evaluate("itemSlots([2003,2003,1036,1036,1001,1028,1027,1052,1033])");
  assert.equal(inventory.querySelector(".quantity").textContent,"2","potions stack with quantity");
  assert.equal(inventory.querySelectorAll('.slot:not(.empty)').length,8,"nonstackable copies stay separate");
  inventory.innerHTML=evaluate("itemSlots([1036,1036,1036,1036,1036,1036,1036,1036,1036])");
  assert.equal(inventory.querySelectorAll("img").length,9,"no silent ninth-item truncation");
  at(data.duration_ms);
  assert.ok(q('.structure.nexus.red.down'));
  assert.equal(q("#next").disabled,true);
  assert.equal(errors.length,0,errors.map(error=>error.message).join("\n"));
  console.log(`Review DOM interactions passed against ${site}/review.html`);
})().catch(error=>{console.error(error);process.exitCode=1;}).finally(()=>window.close());
