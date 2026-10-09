import streamlit as st
import json, time, base64, io
from datetime import datetime
from groq import Groq
from duckduckgo_search import DDGS
import requests
from gtts import gTTS

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="AgentScope AI — Autonomous Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ PREMIUM CSS ============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    * { font-family: 'Space Grotesk', sans-serif; }
    .stApp {
        background: radial-gradient(ellipse at top, #1b1035 0%, #0d0a18 55%, #000000 100%);
    }
    .hero-title {
        font-size: 3rem; font-weight: 700; text-align: center; margin-bottom: 0;
        background: linear-gradient(135deg, #ffd700 0%, #ff9d00 40%, #b366ff 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    .hero-subtitle { text-align: center; color: #999; font-size: 1.05rem; margin: 8px 0 28px; }
    .tool-card {
        background: rgba(179, 102, 255, 0.05);
        border: 1px solid rgba(179, 102, 255, 0.18);
        border-radius: 14px; padding: 16px; text-align: center;
        backdrop-filter: blur(10px);
    }
    .tool-card .icon { font-size: 1.8rem; }
    .tool-card .title { color: #ffd700; font-weight: 600; font-size: 0.9rem; margin-top: 6px; }
    .tool-card .desc { color: #888; font-size: 0.75rem; margin-top: 3px; }
    .stButton > button {
        background: linear-gradient(135deg, #ffd700, #ff8c00);
        color: #000; font-weight: 600; border: none; border-radius: 10px;
    }
    .stButton > button:hover { color: #000; box-shadow: 0 8px 24px rgba(255, 215, 0, 0.3); }
    .footer { text-align: center; color: #555; font-size: 0.85rem; margin-top: 40px; padding: 20px;
        border-top: 1px solid rgba(255,255,255,0.05); }
    .footer b { color: #ffd700; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============ MODELS ============
MODELS = {
    "⚡ GPT-OSS 20B (Fast)": "openai/gpt-oss-20b",
    "🧠 GPT-OSS 120B (Smart)": "openai/gpt-oss-120b",
    "🦙 Llama 3.3 70B": "llama-3.3-70b-versatile",
}
VISION_MODEL = "llama-3.2-90b-vision-preview"

SYSTEM_PROMPT = """You are AgentScope AI, an autonomous assistant with real tools.
TOOLS: web_search (current info/news), get_weather (any city), calculator (math),
convert_currency (live rates), get_datetime (current date/time), save_note/list_notes (reminders).
RULES:
- Use tools when needed, but be EFFICIENT: max 2-3 web searches per question, then synthesize the answer.
- Don't repeat the same search with tiny changes — one good search is enough.
- You may chain different tools for complex questions.
- Cite web sources with URLs when using web_search.
- If no tool is needed, answer directly from knowledge.
- Be concise, helpful, and a little witty. Reply in the user's language."""

# ============ TOOL FUNCTIONS ============
def web_search(query, max_results=5):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        return "\n".join([f"- {r.get('title')}: {r.get('body')} ({r.get('href')})" for r in results])
    except Exception as e:
        return f"Search failed: {str(e)[:150]}"

def get_weather(city):
    try:
        g = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1", timeout=10).json()
        if not g.get("results"):
            return f"City '{city}' not found."
        lat, lon = g["results"][0]["latitude"], g["results"][0]["longitude"]
        name = g["results"][0].get("name", city)
        w = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            timeout=10).json()["current"]
        codes = {0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",45:"Foggy",48:"Icy fog",
                 51:"Light drizzle",61:"Light rain",63:"Rain",65:"Heavy rain",71:"Light snow",
                 80:"Light showers",95:"Thunderstorm",96:"Storm with hail"}
        return f"{name}: {codes.get(w['weather_code'],'Unknown')}, {w['temperature_2m']}°C, Humidity {w['relative_humidity_2m']}%, Wind {w['wind_speed_10m']} km/h"
    except Exception as e:
        return f"Weather failed: {str(e)[:150]}"

def calculator(expression):
    import ast, operator
    allowed = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
               ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
               ast.USub: operator.neg, ast.UAdd: operator.pos}
    def ev(node):
        if isinstance(node, ast.Expression): return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](ev(node.operand))
        raise ValueError("Invalid expression")
    try:
        return str(ev(ast.parse(expression, mode="eval")))
    except Exception as e:
        return f"Calc error: {str(e)[:100]}"

def convert_currency(amount, from_currency, to_currency):
    try:
        fc, tc = from_currency.upper(), to_currency.upper()
        r = requests.get(f"https://open.er-api.com/v6/latest/{fc}", timeout=10).json()
        rate = r["rates"].get(tc)
        if not rate:
            return f"Currency {tc} not found."
        return f"{amount} {fc} = {round(float(amount)*rate, 2)} {tc} (live rate: {rate})"
    except Exception as e:
        return f"Currency failed: {str(e)[:150]}"

def get_datetime():
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")

def save_note(note):
    st.session_state.notes.append({"note": note, "time": get_datetime()})
    return f"✅ Note saved! Total notes: {len(st.session_state.notes)}"

def list_notes():
    if not st.session_state.notes:
        return "No notes saved yet."
    return "\n".join([f"{i+1}. {n['note']} ({n['time']})" for i, n in enumerate(st.session_state.notes)])

TOOL_FUNCS = {
    "web_search": web_search, "get_weather": get_weather, "calculator": calculator,
    "convert_currency": convert_currency, "get_datetime": get_datetime,
    "save_note": save_note, "list_notes": list_notes,
}

TOOLS = [
    {"type": "function", "function": {"name": "web_search",
        "description": "Search the web for current information, news, facts.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_weather",
        "description": "Get current weather for any city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calculator",
        "description": "Evaluate math expressions. Supports + - * / ** % and parentheses.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "convert_currency",
        "description": "Convert money between currencies with live rates.",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"}, "from_currency": {"type": "string"}, "to_currency": {"type": "string"}},
            "required": ["amount", "from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "get_datetime",
        "description": "Get the current date and time.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "save_note",
        "description": "Save a note or reminder for the user.",
        "parameters": {"type": "object", "properties": {"note": {"type": "string"}}, "required": ["note"]}}},
    {"type": "function", "function": {"name": "list_notes",
        "description": "List all saved notes.",
        "parameters": {"type": "object", "properties": {}}}},
]

# ============ AGENT LOOP ============
def run_agent(client, model, messages):
    tool_log = []
    all_models = [model] + [m for m in MODELS.values() if m != model]
    for _ in range(10):
        resp = None
        for m_id in all_models:
            try:
                resp = client.chat.completions.create(
                    model=m_id, messages=messages, tools=TOOLS, tool_choice="auto",
                    temperature=0.3, max_tokens=1200)
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    continue
                raise
        if resp is None:
            return "Rate limit! 1 minute ruko phir try karo.", tool_log
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or "No response.", tool_log
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            fn = TOOL_FUNCS.get(name)
            result = str(fn(**args)) if fn else f"Unknown tool: {name}"
            tool_log.append({"tool": name, "args": args, "result": result[:400]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return "Maine kaafi tools use kiye lekin jawab complete nahi ho saka. Sawal thoda simple karke dobara poocho!", tool_log

def analyze_image(client, img_bytes, question):
    b64 = base64.b64encode(img_bytes).decode()
    try:
        r = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": question or "Describe this image in detail."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
            max_tokens=1200)
        return r.choices[0].message.content
    except Exception as e:
        return f"⚠️ Image analysis unavailable right now. ({str(e)[:120]})"

def speak(text):
    try:
        tts = gTTS(text[:600], lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None

def stream_words(text):
    for w in text.split(" "):
        yield w + " "
        time.sleep(0.015)

# ============ SESSION STATE ============
if "notes" not in st.session_state:
    st.session_state.notes = []
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active" not in st.session_state:
    cid = str(int(time.time() * 1000))
    st.session_state.chats[cid] = {"title": "New Chat", "messages": []}
    st.session_state.active = cid

def new_chat():
    cid = str(int(time.time() * 1000))
    st.session_state.chats[cid] = {"title": "New Chat", "messages": []}
    st.session_state.active = cid

chat = st.session_state.chats[st.session_state.active]

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### 🔑 Setup")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
            st.success("✅ API key loaded")
        except Exception:
            st.info("👆 API key dalo (console.groq.com)")

    st.markdown("### ⚡ Model")
    model_label = st.selectbox("Choose model", list(MODELS.keys()))
    model_id = MODELS[model_label]
    voice = st.checkbox("🔊 Voice replies", value=False)

    st.markdown("### 💬 Chats")
    if st.button("➕ New Chat", use_container_width=True):
        new_chat()
        st.rerun()
    for cid, c in reversed(list(st.session_state.chats.items())):
        label = ("🟡 " if cid == st.session_state.active else "⚪ ") + c["title"][:30]
        if st.button(label, key=f"chat_{cid}", use_container_width=True):
            st.session_state.active = cid
            st.rerun()

    st.markdown("### 🖼️ Vision")
    img_file = st.file_uploader("Image upload karo", type=["png", "jpg", "jpeg"])
    img_q = st.text_input("Image ke baare me poocho", placeholder="Is tasveer me kya hai?")
    if st.button("🔍 Analyze Image", use_container_width=True):
        if not img_file:
            st.warning("Pehle image upload karo!")
        elif not api_key:
            st.warning("API key chahiye!")
        else:
            st.session_state.pending_image = (img_file.getvalue(), img_q)
            st.rerun()

    if st.session_state.notes:
        st.markdown(f"### 📝 Notes ({len(st.session_state.notes)})")
        for n in st.session_state.notes[-3:]:
            st.caption(f"• {n['note'][:50]}")

# ============ HERO ============
st.markdown('<div class="hero-title">🤖 AgentScope AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Autonomous agent with real tools &nbsp;•&nbsp; Web • Weather • Math • Currency • Notes • Vision 🚀</div>', unsafe_allow_html=True)

if not chat["messages"]:
    c1, c2, c3 = st.columns(3)
    tools_show = [("🔍", "Web Search", "Taza khabren & facts"), ("🌤️", "Live Weather", "Kisi bhi sheher ka"),
                  ("🧮", "Calculator", "Instant math"), ("💱", "Currency", "Live rates"),
                  ("📝", "Notes", "Reminders save karo"), ("🖼️", "Vision", "Image analyze karo")]
    for i, (ic, ti, de) in enumerate(tools_show):
        with [c1, c2, c3][i % 3]:
            st.markdown(f'<div class="tool-card"><div class="icon">{ic}</div><div class="title">{ti}</div><div class="desc">{de}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    s1, s2 = st.columns(2)
    suggestions = ["🌤️ Karachi ka mausam kaisa hai?", "🔍 Latest AI news batao",
                   "💱 100 USD kitne PKR hain?", "🧮 (15000 * 12) / 100 kitna hai?"]
    for i, s in enumerate(suggestions):
        if [s1, s2][i % 2].button(s, use_container_width=True, key=f"sug_{i}"):
            st.session_state.pending_prompt = s
            st.rerun()

# ============ IMAGE HANDLING ============
pending_img = st.session_state.pop("pending_image", None)
if pending_img and api_key:
    img_bytes, q = pending_img
    chat["messages"].append({"role": "user", "content": q or "Is tasveer me kya hai?", "image": img_bytes})
    with st.chat_message("user"):
        st.image(img_bytes, width=300)
        st.markdown(q or "Is tasveer me kya hai?")
    with st.chat_message("assistant"):
        with st.spinner("🖼️ Image analyze ho rahi hai..."):
            client = Groq(api_key=api_key)
            ans = analyze_image(client, img_bytes, q)
        st.markdown(ans)
    chat["messages"].append({"role": "assistant", "content": ans})
    if len(chat["messages"]) == 2:
        chat["title"] = "🖼️ " + (q or "Image analysis")[:35]
    st.rerun()

# ============ DISPLAY HISTORY ============
for m in chat["messages"]:
    with st.chat_message(m["role"]):
        if m.get("image"):
            st.image(m["image"], width=300)
        st.markdown(m["content"])
        if m.get("tools"):
            with st.expander(f"🔧 Agent ne {len(m['tools'])} tool(s) use kiye"):
                for t in m["tools"]:
                    st.markdown(f"**`{t['tool']}`** `{json.dumps(t['args'])}`")
                    st.caption(t["result"][:350])
        if m.get("audio"):
            st.audio(m["audio"], format="audio/mp3")

# ============ REGENERATE ============
if chat["messages"] and chat["messages"][-1]["role"] == "assistant":
    if st.button("🔄 Regenerate"):
        chat["messages"].pop()
        st.session_state.do_regen = True
        st.rerun()

# ============ INPUT ============
prompt = st.chat_input("AgentScope se kuch bhi poocho...")
if not prompt:
    prompt = st.session_state.pop("pending_prompt", None)

do_regen = st.session_state.pop("do_regen", False)
if do_regen:
    um = [m for m in chat["messages"] if m["role"] == "user"]
    prompt = um[-1]["content"] if um else None

if prompt:
    if not api_key:
        st.warning("⚠️ Pehle Groq API key dalo (sidebar me)")
        st.stop()

    if not do_regen:
        chat["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        if len(chat["messages"]) == 1:
            chat["title"] = prompt[:38]

    client = Groq(api_key=api_key)
    agent_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in chat["messages"]:
        if m["role"] in ("user", "assistant") and not m.get("image"):
            agent_msgs.append({"role": m["role"], "content": m["content"]})

    with st.chat_message("assistant"):
        with st.status("🤖 Agent soch raha hai...", expanded=False) as status:
            t0 = time.time()
            try:
                answer, tool_log = run_agent(client, model_id, agent_msgs)
            except Exception as e:
                answer, tool_log = f"❌ Error: {str(e)[:200]}", []
            dt = time.time() - t0
            status.update(label=f"✅ {dt:.1f}s • {len(tool_log)} tool(s) • {model_label}", state="complete", expanded=False)
        st.write_stream(stream_words(answer))
        if tool_log:
            with st.expander(f"🔧 Agent ne {len(tool_log)} tool(s) use kiye"):
                for t in tool_log:
                    st.markdown(f"**`{t['tool']}`** `{json.dumps(t['args'])}`")
                    st.caption(t["result"][:350])
        audio_bytes = speak(answer) if voice else None
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")

    chat["messages"].append({"role": "assistant", "content": answer, "tools": tool_log, "audio": audio_bytes})

# ============ FOOTER ============
st.markdown('<div class="footer">Crafted with ❤️ by <b>Engr. Muhammad Sheharyar Khan</b> &nbsp;•&nbsp; Powered by Agentic AI + Groq</div>', unsafe_allow_html=True)
