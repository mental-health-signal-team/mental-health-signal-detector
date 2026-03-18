"""
Contenu thérapeutique — port Python de frontend/src/data/solutions.ts

Contient les messages empathiques, phrases de clôture, micro-actions
et ressources structurées par niveau de triage.
"""

from src.solutions.schemas import MicroAction, Resource

# ─── Ressources ───────────────────────────────────────────────────────────────

RESOURCES: dict[str, Resource] = {
    "line3114": Resource(
        id="3114",
        label="Numéro national prévention suicide",
        detail="3114 — Gratuit, confidentiel, 24h/24",
        type="phone",
        href="tel:3114",
        urgent=True,
    ),
    "samu": Resource(
        id="samu",
        label="SAMU — Urgences médicales",
        detail="15 — Urgences médicales",
        type="phone",
        href="tel:15",
        urgent=True,
    ),
    "filSanteJeunes": Resource(
        id="fil-sante-jeunes",
        label="Fil Santé Jeunes",
        detail="0 800 235 236 — Gratuit, anonyme, 9h–23h",
        type="phone",
        href="tel:0800235236",
    ),
    "monSoutienPsy": Resource(
        id="mon-soutien-psy",
        label="Mon Soutien Psy",
        detail="Séances remboursées avec psychologue conventionné",
        type="website",
        href="https://www.has-sante.fr/jcms/p_3333615/fr/mon-soutien-psy",
    ),
    "psycom": Resource(
        id="psycom",
        label="Psycom — Annuaire psy",
        detail="Trouver un professionnel de santé mentale",
        type="website",
        href="https://www.psycom.org",
    ),
    "medecinTraitant": Resource(
        id="medecin-traitant",
        label="Médecin traitant",
        detail="Premier interlocuteur — peut orienter vers un spécialiste",
        type="person",
    ),
    "proche": Resource(
        id="proche",
        label="Parler à un proche de confiance",
        detail="Famille, ami — le simple fait d'en parler aide",
        type="person",
    ),
    "procheEnfant": Resource(
        id="proche-enfant",
        label="Parler à un adulte de confiance",
        detail="Parent, professeur, infirmier scolaire...",
        type="person",
    ),
    "enfanceEnDanger": Resource(
        id="119",
        label="119 — Allô Enfance en danger",
        detail="Gratuit, confidentiel, 24h/24 — enfants, ados et jeunes majeurs en danger",
        type="phone",
        href="tel:119",
        urgent=True,
    ),
    "antiHarcelement": Resource(
        id="3018",
        label="3018 — Cyberharcèlement",
        detail="Gratuit, anonyme, confidentiel, 7j/7 · 9h–23h — réseaux sociaux et internet",
        type="phone",
        href="tel:3018",
    ),
    "harcelementScolaire": Resource(
        id="3020",
        label="3020 — Harcèlement scolaire",
        detail="Gratuit, lun–ven · 9h–20h — harcèlement à l'école",
        type="phone",
        href="tel:3020",
    ),
    "violencesFemmes": Resource(
        id="3919",
        label="3919 — Arrêtons les violences",
        detail="Gratuit, anonyme, 24h/24 — femmes victimes de violences",
        type="phone",
        href="tel:3919",
        urgent=True,
    ),
}

# ─── Ressources par niveau de triage ─────────────────────────────────────────

RESOURCES_BY_LEVEL: dict[int, dict[str, list[Resource]]] = {
    0: {"kids": [], "adult": []},
    1: {"kids": [], "adult": []},
    2: {
        "kids": [RESOURCES["procheEnfant"], RESOURCES["antiHarcelement"], RESOURCES["harcelementScolaire"], RESOURCES["filSanteJeunes"]],
        "adult": [RESOURCES["proche"], RESOURCES["monSoutienPsy"], RESOURCES["psycom"]],
    },
    3: {
        "kids": [RESOURCES["line3114"], RESOURCES["enfanceEnDanger"], RESOURCES["antiHarcelement"], RESOURCES["harcelementScolaire"], RESOURCES["procheEnfant"]],
        "adult": [RESOURCES["medecinTraitant"], RESOURCES["line3114"], RESOURCES["violencesFemmes"], RESOURCES["monSoutienPsy"]],
    },
    4: {
        "kids": [RESOURCES["line3114"], RESOURCES["enfanceEnDanger"], RESOURCES["filSanteJeunes"], RESOURCES["procheEnfant"]],
        "adult": [RESOURCES["line3114"], RESOURCES["violencesFemmes"], RESOURCES["samu"], RESOURCES["proche"]],
    },
}

# ─── Micro-actions ────────────────────────────────────────────────────────────

MICRO_ACTIONS: dict[str, MicroAction] = {
    "breathing478": MicroAction(
        id="breathing-478",
        title="Respiration 4-7-8",
        description="Inspirez 4s, bloquez 7s, expirez 8s. Répétez 3 fois.",
        duration="2 min",
    ),
    "breathingCoherent": MicroAction(
        id="breathing-coherent",
        title="Respiration cohérente",
        description="Inspirez 5s, expirez 5s. Régule le système nerveux.",
        duration="5 min",
    ),
    "grounding54321": MicroAction(
        id="grounding-54321",
        title="Ancrage 5-4-3-2-1",
        description="5 choses vues, 4 entendues, 3 touchées, 2 senties, 1 goûtée.",
        duration="3 min",
    ),
    "bodyScan": MicroAction(
        id="body-scan",
        title="Scan corporel",
        description="Fermez les yeux. Observez chaque partie de votre corps sans jugement.",
        duration="5 min",
    ),
    "pleasantActivity": MicroAction(
        id="pleasant-activity",
        title="Activité agréable",
        description="Faites quelque chose qui vous fait plaisir, même 10 minutes.",
        duration="10 min",
    ),
    "journaling": MicroAction(
        id="journaling",
        title="Journal émotionnel",
        description="Écrivez librement ce que vous ressentez, sans vous censurer.",
        duration="10 min",
    ),
    "journalingKids": MicroAction(
        id="journaling-kids",
        title="Mon carnet de ressentis",
        description="Dessine ou écris ce que tu ressens. Pas besoin que ce soit parfait !",
        duration="5 min",
    ),
    "thoughtChallenge": MicroAction(
        id="thought-challenge",
        title="Challenger une pensée",
        description="Quelle est la preuve pour cette pensée ? Quelle est la preuve contre ?",
        duration="10 min",
    ),
    "pause": MicroAction(
        id="pause",
        title="Pause intentionnelle",
        description="Posez ce que vous faites. Respirez. Revenez dans le moment présent.",
        duration="2 min",
    ),
}

# ─── Messages empathiques par émotion × niveau ───────────────────────────────
# Structure : { emotionId: { level: { "kids": str, "adult": str } } }

_PROFILE_MESSAGES: dict[str, dict[int, dict[str, str]]] = {
    "sadness": {
        1: {
            "kids": "Je comprends. Parfois, on a le cœur un peu lourd, et c'est normal.",
            "adult": "Je comprends. Avoir le cœur lourd peut être difficile à porter.",
        },
        2: {
            "kids": "Ta tristesse est réelle et elle mérite d'être entendue. Tu n'es pas seul.",
            "adult": "La tristesse que vous portez mérite attention. Vous n'êtes pas seul(e).",
        },
        3: {
            "kids": "Ce que tu ressens est lourd. Des personnes formées peuvent t'aider à traverser ça.",
            "adult": "Ce que vous traversez est difficile. Un professionnel peut vous accompagner.",
        },
        4: {
            "kids": "Tu n'es pas seul. Des personnes sont là pour toi maintenant, 24h/24.",
            "adult": "Vous n'êtes pas seul(e). Des personnes disponibles maintenant peuvent vous aider.",
        },
    },
    "fear": {
        1: {
            "kids": "L'inquiétude, ça arrive. C'est courageux d'en parler.",
            "adult": "L'inquiétude est une réaction normale. Vous avez bien fait d'en prendre conscience.",
        },
        2: {
            "kids": "Cette peur est importante. Tu mérites d'être accompagné pour la traverser.",
            "adult": "Ce que vous ressentez mérite d'être entendu. Vous n'êtes pas seul(e).",
        },
        3: {
            "kids": "Ce que tu vis est trop lourd à porter seul. Des adultes peuvent t'aider.",
            "adult": "Ce niveau d'inquiétude mérite un regard professionnel bienveillant.",
        },
        4: {
            "kids": "Tu traverses quelque chose de très difficile. Des personnes sont là pour toi.",
            "adult": "Vous traversez une situation très difficile. Des personnes sont disponibles maintenant.",
        },
    },
    "stress": {
        1: {
            "kids": "Le stress, ça arrive à tout le monde. On peut apprendre à le gérer.",
            "adult": "Le stress ponctuel est normal. Quelques outils peuvent faire la différence.",
        },
        2: {
            "kids": "Ce stress est trop présent. Prenons le temps de souffler ensemble.",
            "adult": "Ce niveau de stress mérite attention. Prenons le temps de l'aborder.",
        },
        3: {
            "kids": "Ce que tu ressens est épuisant. Des personnes peuvent t'aider à alléger ça.",
            "adult": "Ce que vous portez est lourd. Un accompagnement professionnel peut aider.",
        },
        4: {
            "kids": "Tu n'es pas seul. Des personnes sont là pour toi maintenant.",
            "adult": "Vous n'êtes pas seul(e). Des personnes sont disponibles immédiatement.",
        },
    },
    "anger": {
        1: {
            "kids": "La colère, c'est une émotion normale. Elle a un message pour toi.",
            "adult": "La colère est légitime. Elle signale souvent quelque chose d'important.",
        },
        2: {
            "kids": "Cette colère est forte. Respirons ensemble et essayons de comprendre.",
            "adult": "Cette intensité mérite attention. Identifier la source peut aider.",
        },
        3: {
            "kids": "Ce que tu vis est difficile à gérer seul. Des personnes peuvent t'aider.",
            "adult": "Ce niveau d'intensité mérite un regard professionnel.",
        },
        4: {
            "kids": "Tu n'es pas seul. Des personnes sont là pour toi maintenant.",
            "adult": "Vous n'êtes pas seul(e). Des personnes sont disponibles immédiatement.",
        },
    },
    "tiredness": {
        1: {
            "kids": "La fatigue, ça arrive. Ton corps te dit qu'il a besoin de repos.",
            "adult": "La fatigue est un signal. Votre corps et votre esprit méritent du repos.",
        },
        2: {
            "kids": "Cette fatigue est lourde. Prends soin de toi — tu le mérites.",
            "adult": "Cette fatigue persistante mérite attention. Vous n'êtes pas seul(e).",
        },
        3: {
            "kids": "Tu es épuisé et c'est important. Des personnes peuvent t'aider.",
            "adult": "Cet épuisement mérite un accompagnement professionnel.",
        },
        4: {
            "kids": "Tu n'es pas seul. Des personnes sont là pour toi maintenant.",
            "adult": "Vous n'êtes pas seul(e). Des personnes sont disponibles immédiatement.",
        },
    },
    "joy": {
        0: {
            "kids": "C'est super que tu te sentes joyeux ! Continue à faire ce qui te rend heureux.",
            "adult": "C'est merveilleux de vous sentir joyeux. Continuez à cultiver ce qui vous fait du bien.",
        },
    },
    "calm": {
        0: {
            "kids": "Bravo pour ce moment de calme ! Continue à prendre soin de toi.",
            "adult": "C'est précieux de ressentir cette sérénité. Profitez-en pleinement.",
        },
    },
    "pride": {
        0: {
            "kids": "Tu as raison d'être fier ! Continue comme ça, tu fais de belles choses !",
            "adult": "Votre fierté est justifiée. Reconnaître vos réussites est important.",
        },
    },
}

# ─── Phrases de clôture par émotion × niveau ─────────────────────────────────

_PROFILE_CLOSINGS: dict[str, dict[int, dict[str, str]]] = {
    "sadness": {
        1: {"kids": "Tu n'es pas seul.", "adult": "Vous n'êtes pas seul(e)."},
        2: {"kids": "Un pas à la fois.", "adult": "Un pas à la fois."},
        3: {"kids": "Des personnes peuvent t'aider.", "adult": "Des professionnels peuvent vous accompagner."},
        4: {"kids": "Des personnes sont là pour toi maintenant.", "adult": "Ne restez pas seul(e)."},
    },
    "fear": {
        1: {"kids": "Tu peux en parler.", "adult": "Vous pouvez en parler."},
        2: {"kids": "Tu n'es pas seul.", "adult": "Vous n'êtes pas seul(e)."},
        3: {"kids": "Des personnes peuvent t'aider.", "adult": "Un professionnel peut vous accompagner."},
        4: {"kids": "Des personnes sont là pour toi.", "adult": "Ne restez pas seul(e)."},
    },
    "stress": {
        1: {"kids": "Tu fais de ton mieux.", "adult": "Vous faites de votre mieux."},
        2: {"kids": "Un pas à la fois.", "adult": "Un pas à la fois."},
        3: {"kids": "Des personnes peuvent t'aider.", "adult": "Un accompagnement est possible."},
        4: {"kids": "Tu n'es pas seul.", "adult": "Vous n'êtes pas seul(e)."},
    },
    "anger": {
        1: {"kids": "Tu as le droit de ressentir ça.", "adult": "Votre ressenti est légitime."},
        2: {"kids": "Un pas à la fois.", "adult": "Un pas à la fois."},
        3: {"kids": "Des personnes peuvent t'aider.", "adult": "Un accompagnement est possible."},
        4: {"kids": "Tu n'es pas seul.", "adult": "Vous n'êtes pas seul(e)."},
    },
    "tiredness": {
        1: {"kids": "Tu mérites du repos.", "adult": "Vous méritez du repos."},
        2: {"kids": "Prends soin de toi.", "adult": "Prenez soin de vous."},
        3: {"kids": "Des personnes peuvent t'aider.", "adult": "Un accompagnement est possible."},
        4: {"kids": "Tu n'es pas seul.", "adult": "Vous n'êtes pas seul(e)."},
    },
    "joy": {0: {"kids": "Continue comme ça !", "adult": "Continuez à en prendre soin."}},
    "calm": {0: {"kids": "Profite de ce moment !", "adult": "Profitez de ce moment."}},
    "pride": {0: {"kids": "Tu peux être fier !", "adult": "Soyez fier(e) de vous."}},
}

# ─── Actions recommandées par profil clinique × niveau ───────────────────────

_PROFILE_ACTIONS: dict[str, dict[int, dict[str, list[MicroAction]]]] = {
    "wellbeing": {
        0: {
            "kids": [MICRO_ACTIONS["journalingKids"], MICRO_ACTIONS["pleasantActivity"]],
            "adult": [MICRO_ACTIONS["journaling"], MICRO_ACTIONS["pleasantActivity"]],
        },
    },
    "adjustment": {
        1: {
            "kids": [MICRO_ACTIONS["breathingCoherent"], MICRO_ACTIONS["journalingKids"]],
            "adult": [MICRO_ACTIONS["breathingCoherent"], MICRO_ACTIONS["journaling"]],
        },
        2: {
            "kids": [MICRO_ACTIONS["grounding54321"], MICRO_ACTIONS["pause"]],
            "adult": [MICRO_ACTIONS["grounding54321"], MICRO_ACTIONS["pause"]],
        },
    },
    "anxiety": {
        2: {
            "kids": [MICRO_ACTIONS["breathing478"], MICRO_ACTIONS["grounding54321"]],
            "adult": [MICRO_ACTIONS["breathing478"], MICRO_ACTIONS["grounding54321"]],
        },
        3: {
            "kids": [MICRO_ACTIONS["breathingCoherent"], MICRO_ACTIONS["pause"]],
            "adult": [MICRO_ACTIONS["breathingCoherent"], MICRO_ACTIONS["bodyScan"]],
        },
    },
    "burnout": {
        2: {
            "kids": [MICRO_ACTIONS["pause"], MICRO_ACTIONS["pleasantActivity"]],
            "adult": [MICRO_ACTIONS["pause"], MICRO_ACTIONS["pleasantActivity"]],
        },
        3: {
            "kids": [MICRO_ACTIONS["breathingCoherent"], MICRO_ACTIONS["pause"]],
            "adult": [MICRO_ACTIONS["breathingCoherent"], MICRO_ACTIONS["pause"]],
        },
    },
    "depression": {
        2: {
            "kids": [MICRO_ACTIONS["pleasantActivity"], MICRO_ACTIONS["journalingKids"]],
            "adult": [MICRO_ACTIONS["pleasantActivity"], MICRO_ACTIONS["thoughtChallenge"]],
        },
        3: {
            "kids": [MICRO_ACTIONS["breathingCoherent"], MICRO_ACTIONS["pause"]],
            "adult": [MICRO_ACTIONS["breathingCoherent"], MICRO_ACTIONS["pause"]],
        },
    },
    "crisis": {
        4: {
            "kids": [MICRO_ACTIONS["breathingCoherent"]],
            "adult": [MICRO_ACTIONS["breathingCoherent"]],
        },
    },
}

# ─── Fallbacks génériques par niveau ─────────────────────────────────────────

FALLBACK_MESSAGES: dict[int, dict[str, str]] = {
    0: {"kids": "Tu vas bien et c'est précieux. Continue à prendre soin de toi !",
        "adult": "Vous êtes dans un bon équilibre. Prenez le temps de l'apprécier."},
    1: {"kids": "Ce que tu ressens est normal. Voici quelque chose qui peut t'aider.",
        "adult": "Ce que vous ressentez est compréhensible. Quelques petites actions peuvent faire la différence."},
    2: {"kids": "Je sens que c'est difficile en ce moment. Tu n'es pas seul.",
        "adult": "Ce que vous traversez mérite attention et soutien."},
    3: {"kids": "Ce que tu ressens est important. Des personnes formées peuvent t'aider.",
        "adult": "Vous traversez quelque chose de difficile. Un professionnel peut vous accompagner."},
    4: {"kids": "Tu traverses un moment très difficile. Tu n'es pas seul — des personnes sont là pour toi maintenant.",
        "adult": "Vous traversez une situation très difficile. Vous n'êtes pas seul(e). Des personnes sont disponibles immédiatement."},
}

FALLBACK_CLOSINGS: dict[int, dict[str, str]] = {
    0: {"kids": "Prends soin de toi.", "adult": "Prenez soin de vous."},
    1: {"kids": "Tu n'es pas seul.", "adult": "Vous n'êtes pas seul(e)."},
    2: {"kids": "Un pas à la fois.", "adult": "Un pas à la fois."},
    3: {"kids": "Des personnes peuvent t'aider.", "adult": "Des professionnels peuvent vous accompagner."},
    4: {"kids": "Des personnes sont là pour toi maintenant.", "adult": "Ne restez pas seul(e). Des personnes sont disponibles immédiatement."},
}

FALLBACK_ACTIONS: dict[str, list[MicroAction]] = {
    "kids": [MICRO_ACTIONS["breathingCoherent"], MICRO_ACTIONS["grounding54321"]],
    "adult": [MICRO_ACTIONS["breathingCoherent"], MICRO_ACTIONS["grounding54321"]],
}


def get_message(emotion_id: str, level: int, mode: str) -> str:
    emotion_msgs = _PROFILE_MESSAGES.get(emotion_id, {})
    # Cherche le niveau exact, puis remonte
    for lvl in range(level, -1, -1):
        entry = emotion_msgs.get(lvl)
        if entry:
            return entry.get(mode, entry.get("adult", ""))
    return FALLBACK_MESSAGES[level][mode]


def get_closing(emotion_id: str, level: int, mode: str) -> str:
    emotion_closings = _PROFILE_CLOSINGS.get(emotion_id, {})
    for lvl in range(level, -1, -1):
        entry = emotion_closings.get(lvl)
        if entry:
            return entry.get(mode, entry.get("adult", ""))
    return FALLBACK_CLOSINGS[level][mode]


def get_actions(clinical_profile: str, level: int, mode: str) -> list[MicroAction]:
    profile_actions = _PROFILE_ACTIONS.get(clinical_profile, {})
    for lvl in range(level, -1, -1):
        entry = profile_actions.get(lvl)
        if entry:
            return entry.get(mode, entry.get("adult", []))
    return FALLBACK_ACTIONS[mode]
