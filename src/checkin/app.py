"""
Interface Gradio — "Comment vas-tu ce matin ?"
Phase 2 — Mental Health Signal Detector
"""

import os
import requests
import gradio as gr

API_URL = os.getenv("API_URL", "http://localhost:8000")

LEVEL_ICONS = {"green": "✅", "yellow": "💛", "red": "🆘"}

# HTML des emojis avec inline styles — indépendant du thème Gradio
# Chaque bouton est enveloppé dans un wrapper relatif pour positionner le tooltip
_EMOJI_HTML = """
<style>
.emo-wrap { position:relative; display:inline-block; }
.emo-tooltip {
    position:absolute; bottom:calc(100% + 10px); left:50%;
    transform:translateX(-50%) scale(0.85);
    background:#333; color:#fff;
    padding:6px 14px; border-radius:20px;
    font-size:0.92rem; font-weight:600; white-space:nowrap;
    pointer-events:none; opacity:0;
    transition:opacity 0.18s ease, transform 0.18s ease;
    box-shadow:0 3px 10px rgba(0,0,0,0.25);
    z-index:99;
}
/* Petite flèche sous le tooltip */
.emo-tooltip::after {
    content:''; position:absolute; top:100%; left:50%;
    transform:translateX(-50%);
    border:6px solid transparent;
    border-top-color:#333;
}
.emo-wrap:hover .emo-tooltip {
    opacity:1;
    transform:translateX(-50%) scale(1);
}
</style>

<div id="emoji-row" style="
    display:flex; gap:18px; justify-content:center;
    flex-wrap:wrap; margin:18px 0; padding:8px 0;">

  <div class="emo-wrap">
    <span class="emo-tooltip">Joie · Enthousiasme</span>
    <button onclick="selectEmoji('😄',this)" style="
        font-size:3rem; padding:14px 22px; border-radius:24px;
        border:3px solid #c8f7c5; background:#eafaf1;
        box-shadow:0 5px 0 #a9dfb2; cursor:pointer;
        transition:transform 0.12s,box-shadow 0.12s; outline:none;">😄</button>
  </div>

  <div class="emo-wrap">
    <span class="emo-tooltip">Sérénité · Bien-être</span>
    <button onclick="selectEmoji('🙂',this)" style="
        font-size:3rem; padding:14px 22px; border-radius:24px;
        border:3px solid #d5f5e3; background:#f0faf4;
        box-shadow:0 5px 0 #a9dfb2; cursor:pointer;
        transition:transform 0.12s,box-shadow 0.12s; outline:none;">🙂</button>
  </div>

  <div class="emo-wrap">
    <span class="emo-tooltip">Neutralité · Incertitude</span>
    <button onclick="selectEmoji('😐',this)" style="
        font-size:3rem; padding:14px 22px; border-radius:24px;
        border:3px solid #fef9e7; background:#fffde7;
        box-shadow:0 5px 0 #f7dc6f; cursor:pointer;
        transition:transform 0.12s,box-shadow 0.12s; outline:none;">😐</button>
  </div>

  <div class="emo-wrap">
    <span class="emo-tooltip">Tristesse · Fatigue</span>
    <button onclick="selectEmoji('😔',this)" style="
        font-size:3rem; padding:14px 22px; border-radius:24px;
        border:3px solid #fde8d8; background:#fef5ec;
        box-shadow:0 5px 0 #f0a07a; cursor:pointer;
        transition:transform 0.12s,box-shadow 0.12s; outline:none;">😔</button>
  </div>

  <div class="emo-wrap">
    <span class="emo-tooltip">Détresse · Désespoir</span>
    <button onclick="selectEmoji('😢',this)" style="
        font-size:3rem; padding:14px 22px; border-radius:24px;
        border:3px solid #fadbd8; background:#fef0ee;
        box-shadow:0 5px 0 #e98c85; cursor:pointer;
        transition:transform 0.12s,box-shadow 0.12s; outline:none;">😢</button>
  </div>

</div>

<script>
function selectEmoji(emoji, btn) {
    // Remettre tous les boutons à l'état non-sélectionné
    document.querySelectorAll('#emoji-row button').forEach(b => {
        b.style.transform   = 'translateY(0) scale(1)';
        b.style.opacity     = '0.45';
        b.style.boxShadow   = '0 5px 0 rgba(0,0,0,0.12)';
        b.style.border      = '3px solid #ddd';
        b.style.filter      = 'grayscale(40%)';
    });
    // Effacer les coches précédentes
    document.querySelectorAll('#emoji-row .emo-check').forEach(c => c.remove());

    // Animer le bouton sélectionné
    btn.style.transform  = 'translateY(-8px) scale(1.28)';
    btn.style.opacity    = '1';
    btn.style.boxShadow  = '0 12px 0 rgba(0,0,0,0.18), 0 0 0 4px #fff, 0 0 0 7px #4CAF50';
    btn.style.border     = '3px solid #4CAF50';
    btn.style.filter     = 'grayscale(0%)';

    // Ajouter une coche sous le bouton sélectionné
    const check = document.createElement('div');
    check.className = 'emo-check';
    check.textContent = '✓';
    check.style.cssText = `
        text-align:center; font-size:1.1rem; font-weight:900;
        color:#4CAF50; margin-top:4px; line-height:1;
        animation: popIn 0.2s ease;
    `;
    btn.parentElement.appendChild(check);

    // Envoyer la valeur au composant Gradio caché
    const tb = document.querySelector('#emoji-hidden textarea');
    if (tb) {
        tb.value = emoji;
        tb.dispatchEvent(new Event('input', {bubbles: true}));
    }
}
</script>

<style>
@keyframes popIn {
    0%   { transform: scale(0); opacity: 0; }
    70%  { transform: scale(1.3); opacity: 1; }
    100% { transform: scale(1); opacity: 1; }
}
</style>
"""


# ---------------------------------------------------------------------------
# Appel API
# ---------------------------------------------------------------------------

def call_checkin(emoji: str | None, text: str, step: int = 1) -> dict:
    payload = {"step": step}
    if emoji and emoji.strip():
        payload["emoji"] = emoji.strip()
    if text and text.strip():
        payload["text"] = text.strip()
    try:
        r = requests.post(f"{API_URL}/checkin", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "API hors ligne — lance `bash scripts/run_api.sh`"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Formatage réponse
# ---------------------------------------------------------------------------

def format_response(data: dict) -> str:
    if "error" in data:
        return f"⚠️ **Erreur :** {data['error']}"

    level = data.get("level", "green")

    if level == "green":
        visual  = "☀️"
        hearts  = "💚 💚 💚 💚 💚"
        need_label = "Tu vas bien !"
    elif level == "yellow":
        visual  = "⛅"
        hearts  = "💛 💛 💛 🤍 🤍"
        need_label = "Un peu de soutien peut aider"
    else:
        visual  = "🌧️"
        hearts  = "🧡 🧡 🧡 🧡 🧡"
        need_label = "J'ai besoin d'aide"

    lines = [
        f"## {visual}",
        "**Est-ce que j'ai besoin d'aide ?**",
        f"### {hearts}",
        f"*{need_label}*",
        "",
        f"**{data['message']}**",
        "",
    ]

    if data.get("tip"):
        lines += [data["tip"], ""]

    if data.get("follow_up"):
        lines += [f"---\n🤔 **{data['follow_up']}**", ""]

    if data.get("resources"):
        lines += ["---\n### 📋 Ressources disponibles\n"]
        for res in data["resources"]:
            lines += [
                f"**{res['title']}**",
                res["description"],
                f"→ [{res['action']}]({res['url']})",
                "",
            ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def handle_emoji_selected(emoji: str, state: dict):
    if not emoji or not emoji.strip():
        return state, "_Clique sur un emoji pour commencer_"
    state["emoji"] = emoji.strip()
    state["step"] = 1
    labels = {"😄": "Très bien", "🙂": "Bien", "😐": "Mouais", "😔": "Pas terrible", "😢": "Vraiment pas bien"}
    label = labels.get(emoji.strip(), emoji.strip())
    return state, f"Tu as choisi **{label} {emoji.strip()}** — ajoute quelques mots si tu veux, puis clique **Envoyer**."


def handle_send(text: str, state: dict):
    emoji = state.get("emoji")
    step  = state.get("step", 1)

    if not emoji and not (text and text.strip()):
        return "Choisis d'abord un emoji ou écris quelque chose 🙏", text, state

    data = call_checkin(emoji=emoji, text=text, step=step)
    response_md = format_response(data)

    if data.get("follow_up") and step == 1:
        state["step"] = 2
        state["emoji"] = None
    else:
        state["emoji"] = None
        state["step"] = 1

    return response_md, "", state


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="Comment vas-tu ce matin ?") as demo:

    state = gr.State({"emoji": None, "step": 1})

    gr.Markdown("""
# 💬 Comment vas-tu ce matin ?
*Un moment pour toi — rapide, bienveillant, confidentiel.*
---
""")

    gr.Markdown("### Comment tu te sens en ce moment ?")
    gr.HTML(_EMOJI_HTML)

    # Textbox cachée — reçoit la valeur de l'emoji via JS
    emoji_hidden = gr.Textbox(value="", visible=False, elem_id="emoji-hidden")

    emoji_status = gr.Markdown("_Clique sur un emoji pour commencer_")

    with gr.Row():
        text_input = gr.Textbox(
            placeholder="Tu peux ajouter quelques mots… (optionnel)",
            label="",
            lines=2,
            max_lines=4,
            scale=5,
        )
        send_btn = gr.Button("Envoyer ➤", variant="primary", scale=1)

    gr.Markdown("---")
    response_box = gr.Markdown("_Ta réponse apparaîtra ici_")

    gr.Markdown("""
---
<small>⚠️ Cette application est un outil de bien-être, pas un dispositif médical.
En cas d'urgence, appelle le **15 (SAMU)** ou le **3114**.</small>
""")

    # Bindings
    emoji_hidden.change(
        fn=handle_emoji_selected,
        inputs=[emoji_hidden, state],
        outputs=[state, emoji_status],
    )
    send_btn.click(
        fn=handle_send,
        inputs=[text_input, state],
        outputs=[response_box, text_input, state],
    )
    text_input.submit(
        fn=handle_send,
        inputs=[text_input, state],
        outputs=[response_box, text_input, state],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
