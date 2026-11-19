// Boundary tests for deaths, re-observation and seeking. Uses only the public
// Faker fixture, never a training/validation/test payload.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const zlib = require("node:zlib");
const {create, lastAt} = require("../web/review-playback.js");

const snapshot = (ts, health, x, secondHealth=100) => ({ts,players:{
  1:{health,health_max:100,position:{x,y:x},level:1,cs:0},
  2:{health:secondHealth,health_max:100,position:{x:20,y:20},level:1,cs:0},
}});
const death = (ts, victim=1, x=50) => ({ts,kind:"kill",actors:[victim===1?2:1,victim],assists:[],position:{x,y:x}});
const base = {
  duration_ms:200000, players:[{participant_id:1},{participant_id:2}],
  observations:[snapshot(0,100,10),snapshot(60000,0,999),snapshot(120000,100,5),snapshot(180000,100,25)],
  events:[death(30000),death(70000,2),death(150000)],
  inventory_states:[{ts:0,players:{1:[1],2:[]}},{ts:40000,players:{1:[2],2:[]}}],
  structures:[{key:"inhibitor",destroyed_at:null,down_intervals:[{from:50000,to:100000}]}],
};
const core = create(base);
assert.equal(lastAt([{ts:10}],9),null,"a future frame must never be returned");
assert.deepEqual(core.stateAt(29999).players[1].position,{x:10,y:10},"no future-position interpolation");
for(const ts of [30000,30001,60000,70000,119999]){
  const p=core.stateAt(ts).players[1];
  assert.equal(p.position,null,`dead player has no living marker at ${ts}`);
  assert.deepEqual(p.death_position,{x:50,y:50},"death marker stays at the recorded location");
}
assert.equal(core.stateAt(70000).players[1].life,"death_recorded","a posthumous kill cannot prove revival");
assert.deepEqual(core.stateAt(120000).players[1].position,{x:5,y:5},"resume only at observed living frame");
assert.equal(core.stateAt(150000).players[1].position,null,"second death invalidates the newer position");
assert.equal(core.stateAt(179999).players[1].position,null);
assert.deepEqual(core.stateAt(180000).players[1].position,{x:25,y:25});
assert.deepEqual(core.stateAt(29999).players[1].position,{x:10,y:10},"backward seeking does not retain future state");
assert.deepEqual(core.stateAt(39999).players[1].inventory,[1]);
assert.deepEqual(core.stateAt(40000).players[1].inventory,[2]);
assert.equal(core.stateAt(49999).structures[0].down,false);
assert.equal(core.stateAt(50000).structures[0].down,true);
assert.equal(core.stateAt(100000).structures[0].down,false);
const simultaneous=create({...base,observations:[snapshot(0,100,10),snapshot(30000,100,80)]});
assert.equal(simultaneous.stateAt(30000).players[1].position,null,"death wins a same-timestamp living sample");
const missing=create({...base,observations:[snapshot(0,null,10),snapshot(60000,0,999)]});
assert.equal(missing.stateAt(0).players[1].position,null,"missing health must not become living evidence");
assert.equal(missing.stateAt(60000).players[1].position,null);

const script=fs.readFileSync(path.join(__dirname,"../web/review-faker-data.js"),"utf8");
const data=JSON.parse(zlib.gunzipSync(Buffer.from(script.match(/"([A-Za-z0-9+/=]+)"/)[1],"base64")));
assert.equal(data.match_id,"KR_8386011226");
assert.ok(data.observations.every(f=>Object.values(f.players).every(p=>typeof p.health==="number")),"fixture must carry source health");
const real=create(data), deaths=data.events.filter(e=>e.kind==="kill");
let zeroHealthChecks=0, positiveHealthChecks=0;
for(const e of deaths){
  const id=e.actors[1];
  const nextAlive=data.observations.find(f=>f.ts>e.ts&&f.players[id]?.health>0);
  for(const ts of [e.ts,e.ts+1,...(nextAlive?[nextAlive.ts-1]:[])]){
    assert.equal(real.stateAt(ts).players[id].position,null,`${id} cannot move alive after death ${e.ts}`);
  }
  assert.deepEqual(real.stateAt(e.ts+1).players[id].death_position,e.position);
  if(nextAlive&&!deaths.some(other=>other.actors[1]===id&&other.ts===nextAlive.ts)){
    assert.deepEqual(real.stateAt(nextAlive.ts).players[id].position,nextAlive.players[id].position);
  }
}
for(const frame of data.observations){
  for(const [id,sample] of Object.entries(frame.players)){
    if(sample.health===0){assert.equal(real.stateAt(frame.ts).players[id].position,null);zeroHealthChecks++;}
    else if(!deaths.some(e=>String(e.actors[1])===id&&e.ts===frame.ts)){
      assert.deepEqual(real.stateAt(frame.ts).players[id].position,sample.position);positiveHealthChecks++;
    }
  }
}
console.log(`Playback boundaries passed: ${deaths.length} deaths, ${zeroHealthChecks} zero-health and ${positiveHealthChecks} positive-health frame checks.`);
