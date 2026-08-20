const chatArea = document.getElementById('chatArea');
const input = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
const healthEl = document.getElementById('health');
const emptyState = document.getElementById('emptyState');
const SESSION_STORAGE_KEY = 'dosewise_session_id';
let sessionId = localStorage.getItem(SESSION_STORAGE_KEY) || crypto.randomUUID();
localStorage.setItem(SESSION_STORAGE_KEY, sessionId);

function addMessage(text, who='assistant', meta=null, rtl=false){
  // hide empty state when first message is added
  if(emptyState) emptyState.style.display = 'none';
  const el = document.createElement('div');
  el.className = 'msg ' + (who==='user' ? 'user' : 'assistant');
  if(rtl) { el.classList.add('rtl'); el.setAttribute('dir','rtl'); }

  if(who === 'assistant'){
    const row = document.createElement('div'); row.className='assistant-row';
    const avatar = document.createElement('div'); avatar.className='assistant-avatar'; avatar.innerText='DW';
    const bubble = document.createElement('div'); bubble.className='content'; bubble.innerHTML = escapeHtml(text);
    row.appendChild(avatar); row.appendChild(bubble); el.appendChild(row);
  } else {
    el.innerHTML = `<div class='content'>${escapeHtml(text)}</div>`;
  }

  if(meta){
    const m = document.createElement('div'); m.className='meta'; m.innerText = meta; el.appendChild(m);
  }
  chatArea.appendChild(el);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function escapeHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}

function showInteractionCard(retrievals){
  retrievals.forEach(r=>{
    const card = document.createElement('div'); card.className='card interaction';
    const pair = document.createElement('div'); pair.className='pair'; pair.innerText = `${r.drug_a} + ${r.drug_b}`;
    card.appendChild(pair);
    const sevLine = document.createElement('div'); sevLine.className='meta-line';
    if(r.interaction_level){
      const lvl = String(r.interaction_level).toLowerCase();
      sevLine.innerHTML = 'Severity: ' + `<span class="badge ${lvl}">${escapeHtml(r.interaction_level)}</span>`;
    } else {
      sevLine.innerText = 'Severity: Not identified in DDInter';
    }
    card.appendChild(sevLine);
    const src = document.createElement('div'); src.className='meta-line'; src.innerText = 'Source: DDInter'; card.appendChild(src);
    const evidence = document.createElement('div'); evidence.className='meta-line';
    const references = (r.chunk_ids || []).join(', ');
    evidence.innerText = references ? `Evidence: DDInter · Record: ${references}` : 'Evidence: DDInter · Record: none';
    card.appendChild(evidence);
    chatArea.appendChild(card);
  })
}

function showDetectedMeds(extracted=[], verified=[]){
  if((extracted||[]).length===0 && (verified||[]).length===0) return;
  const box = document.createElement('div'); box.className='card';
  const title = document.createElement('div'); title.className='pair'; title.innerText='Detected medications'; box.appendChild(title);
  const list = document.createElement('div'); list.className='interaction';
  // try to pair extracted -> resolved by index
  const max = Math.max(extracted.length, verified.length);
  for(let i=0;i<max;i++){
    const e = extracted[i]||''; const v = verified[i]||'';
    const row = document.createElement('div'); row.className='meta-line'; row.innerText = e ? `${e}${v? ' → '+v : ''}` : (v||'');
    list.appendChild(row);
  }
  box.appendChild(list);
  chatArea.appendChild(box);
}

async function postMessage(message){
  // prevent duplicate sends
  if(sendBtn.disabled) return;
  addMessage(message,'user',null, detectRTL(message));
  sendBtn.disabled = true; sendBtn.innerText='Sending...';
  // show typing indicator
  const typingEl = document.createElement('div'); typingEl.className='msg assistant'; typingEl.id='typing'; typingEl.innerHTML = `<div class="assistant-row"><div class="assistant-avatar">DW</div><div class="content typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div></div>`;
  chatArea.appendChild(typingEl);
  chatArea.scrollTop = chatArea.scrollHeight;
  try{
    const resp = await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message, session_id: sessionId})});
    if(!resp.ok){
      // show friendly error
      addMessage('Something went wrong while checking the medications. Please try again.','assistant');
      return;
    }
    const data = await resp.json();
    // remove typing
    const t = document.getElementById('typing'); if(t) t.remove();

    // If clarification required
    if(data.needs_clarification){
      addMessage(data.response,'assistant', null, detectRTL(data.response));
      return;
    }

    // show detected/resolved meds compactly
    showDetectedMeds(data.extracted || [], data.verified_generics || []);

    // show structured retrievals then response
    if(data.retrievals && data.retrievals.length>0){
      showInteractionCard(data.retrievals);
    }

    addMessage(data.response,'assistant',null, detectRTL(data.response));

  }catch(e){
    const t = document.getElementById('typing'); if(t) t.remove();
    addMessage('Something went wrong while checking the medications. Please try again.','assistant');
  }finally{sendBtn.disabled=false; sendBtn.innerText='Send'}
}

function detectRTL(text){
  if(!text) return false;
  // simple Arabic script detection
  return /[\u0600-\u06FF\u0750-\u077F]/.test(text);
}

sendBtn.addEventListener('click', ()=>{
  const v = input.value.trim(); if(!v) return; postMessage(v); input.value='';
});
clearBtn.addEventListener('click', async () => {
  chatArea.innerHTML = '';
  if (emptyState) emptyState.style.display = 'flex';
  
  // Inform backend to clear session state (optional, but good practice)
  try {
    await fetch('/api/session', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId })
    });
  } catch (e) {}
  
  // Generate a completely new session id to ensure isolation
  sessionId = crypto.randomUUID();
  localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
});

input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});

// wire example buttons
document.querySelectorAll('.example').forEach(b=>{
  b.addEventListener('click', ()=>{
    input.value = b.innerText; input.focus();
  })
});

// health check
async function pollHealth(){
  try{
    const r = await fetch('/health'); if(r.ok){ const j=await r.json(); healthEl.innerText = j.status==='ok' ? 'Backend: ready' : 'Backend: unknown'; }
  }catch(e){ healthEl.innerText='Backend: unreachable' }
}
pollHealth(); setInterval(pollHealth,5000);
