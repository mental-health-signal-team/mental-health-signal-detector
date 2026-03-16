"""
Contenu conversationnel de l'app "Comment vas-tu ce matin ?"

Trois niveaux de réponse selon le score de détresse :
  - VERT  (score < 0.35) : encouragement + tip du jour
  - JAUNE (0.35 ≤ score < 0.65) : questions d'approfondissement
  - ROUGE (score ≥ 0.65) : écoute empathique + ressources d'aide

Inspiré des témoignages Jade, Sarah, Bruno, Enzo (Mon Soutien Psy / Assurance Maladie 2026)
et du Journal de bien-être émotionnel pour ados.
"""

# ---------------------------------------------------------------------------
# Mapping emoji → score initial
# ---------------------------------------------------------------------------

EMOJI_SCORES: dict[str, float] = {
    "😄": 0.05,   # Très bien
    "🙂": 0.20,   # Bien
    "😐": 0.45,   # Mouais
    "😔": 0.65,   # Pas terrible
    "😢": 0.85,   # Vraiment pas bien
}

EMOJI_LABELS: dict[str, str] = {
    "😄": "Très bien",
    "🙂": "Bien",
    "😐": "Mouais",
    "😔": "Pas terrible",
    "😢": "Vraiment pas bien",
}

# ---------------------------------------------------------------------------
# Réponses niveau VERT — encouragement
# ---------------------------------------------------------------------------

GREEN_RESPONSES = [
    "Super ! Profite de cette belle énergie aujourd'hui. 🌱",
    "Content de l'entendre ! Les bonnes journées, ça se cultive.",
    "C'est bien de le reconnaître quand ça va. Continue comme ça ! ✨",
    "Bonne nouvelle ! Tu es sur la bonne longueur d'onde ce matin.",
]

GREEN_TIPS = [
    "💡 **Tip du jour :** Prends 2 minutes pour noter une chose qui te rend reconnaissant(e) ce matin.",
    "💡 **Tip du jour :** Une petite marche de 10 min aujourd'hui peut booster ton humeur toute la journée.",
    "💡 **Tip du jour :** Partage un moment sympa avec quelqu'un que tu apprécies — ça multiplie le bien-être.",
    "💡 **Tip du jour :** Hydrate-toi bien dès le matin — ça influence vraiment l'énergie et la concentration.",
    "💡 **Tip du jour :** Commence ta journée par la tâche qui te tient le plus à cœur, pas par les emails.",
]

# ---------------------------------------------------------------------------
# Questions de suivi niveau JAUNE — approfondissement
# ---------------------------------------------------------------------------

YELLOW_FOLLOWUP_QUESTIONS = [
    "Depuis combien de temps tu te sens comme ça ?",
    "Est-ce qu'il y a quelque chose de précis qui te pèse en ce moment ?",
    "As-tu réussi à dormir correctement cette nuit ?",
    "Est-ce que c'est plutôt dans ta tête ou aussi dans ton corps (fatigue, tension) ?",
    "Y a-t-il quelqu'un à qui tu peux en parler autour de toi ?",
]

YELLOW_RESPONSES = [
    "Je t'entends. C'est courageux de le reconnaître. 💛",
    "Merci de me le dire. Ces moments difficiles, ça arrive à tout le monde — même si on ne le dit pas toujours.",
    "C'est ok de ne pas être au top. L'important c'est de ne pas rester seul(e) avec ça.",
    "Tu as bien fait de répondre ce matin. On va essayer d'y voir un peu plus clair ensemble.",
]

YELLOW_TIPS = [
    "💡 **Rappel :** Parfois, mettre des mots sur ce qu'on ressent suffit à l'alléger un peu.",
    "💡 **Rappel :** Si cette sensation dure depuis plusieurs jours, c'est un signal à prendre au sérieux — pas à ignorer.",
    "💡 **Rappel :** Tu n'as pas besoin d'aller très mal pour avoir le droit de demander de l'aide.",
]

# ---------------------------------------------------------------------------
# Réponses niveau ROUGE — empathie + ressources
# ---------------------------------------------------------------------------

RED_RESPONSES = [
    "Je suis là. Ce que tu ressens est réel, et tu mérites d'être soutenu(e). 🤝",
    "Merci de me faire confiance avec ça. Ce n'est pas rien de dire que ça ne va pas.",
    "Tu n'es pas seul(e) dans ce que tu traverses. Des milliers de personnes vivent des moments similaires — et s'en sortent.",
    "Ce que tu ressens a un nom, et des professionnels sont formés pour t'accompagner. Tu n'as pas à traverser ça seul(e).",
]

RED_FOLLOWUP_QUESTIONS = [
    "Est-ce que tu as des pensées noires ou des idées de te faire du mal ?",
    "As-tu quelqu'un de confiance auprès de qui tu peux être ce soir ?",
    "Est-ce que tu as déjà parlé à un médecin ou un professionnel de santé de ce que tu ressens ?",
]

# ---------------------------------------------------------------------------
# Ressources d'aide — Mon Soutien Psy + numéros de crise
# ---------------------------------------------------------------------------

RESOURCES = {
    "mon_soutien_psy": {
        "title": "🏥 Mon Soutien Psy — Assurance Maladie",
        "description": (
            "Jusqu'à **12 séances** par an avec un psychologue, remboursées à 60% par l'Assurance Maladie. "
            "Sans avance de frais pour les bénéficiaires de la CSS. "
            "Accessible dès 3 ans."
        ),
        "action": "Trouver un psychologue partenaire",
        "url": "https://monsoutienpsy.ameli.fr",
    },
    "medecin": {
        "title": "👨‍⚕️ Parler à son médecin traitant",
        "description": (
            "Ton médecin traitant peut te remettre un courrier d'accompagnement "
            "pour accéder à Mon Soutien Psy, ou t'orienter vers d'autres ressources adaptées."
        ),
        "action": "Prendre rendez-vous",
        "url": "https://www.doctolib.fr",
    },
    "crisis_3114": {
        "title": "📞 3114 — Numéro national de prévention du suicide",
        "description": (
            "Disponible **24h/24, 7j/7**. Des professionnels de santé formés répondent "
            "et peuvent vous orienter vers les soins adaptés. Gratuit depuis tout téléphone."
        ),
        "action": "Appeler le 3114",
        "url": "https://www.3114.fr",
    },
    "fil_sante_jeunes": {
        "title": "💬 Fil Santé Jeunes — 0 800 235 236",
        "description": (
            "Pour les 12-25 ans. Gratuit, anonyme, disponible de 9h à 23h. "
            "Chat disponible sur filsantejeunes.com."
        ),
        "action": "Accéder au chat",
        "url": "https://www.filsantejeunes.com",
    },
}

# ---------------------------------------------------------------------------
# Messages d'entrée selon l'heure
# ---------------------------------------------------------------------------

def get_greeting(hour: int) -> str:
    if 5 <= hour < 12:
        return "Bonjour ☀️"
    elif 12 <= hour < 18:
        return "Bonjour 🌤"
    else:
        return "Bonsoir 🌙"


OPENING_MESSAGE = (
    "{greeting} — **Comment vas-tu ce matin ?**\n\n"
    "Choisis l'emoji qui correspond le mieux à ton état en ce moment :\n\n"
    "😄 Très bien &nbsp;&nbsp; 🙂 Bien &nbsp;&nbsp; 😐 Mouais &nbsp;&nbsp; 😔 Pas terrible &nbsp;&nbsp; 😢 Vraiment pas bien\n\n"
    "_Tu peux aussi écrire quelques mots si tu veux — ou juste cliquer sur un emoji._"
)
