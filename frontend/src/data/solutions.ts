/**
 * Bibliothèque de contenu thérapeutique — Phase 3
 * Organisée par niveau (0–4) × émotion × mode (kids / adult)
 *
 * Sources cliniques :
 * - CBT : Beck, 1979 / Behavioural Activation (Jacobson)
 * - ACT : Hayes, 2004
 * - Grounding : Dialectical Behaviour Therapy (Linehan)
 * - Stepped-care : NICE Guidelines (depression / anxiety)
 * - Ressources France : 3114, Mon Soutien Psy (décret 2022), Psycom
 */

import type { MicroAction, Resource } from "../types/solutions";

// ─── RESSOURCES ──────────────────────────────────────────────────────────────

export const RESOURCES: Record<string, Resource> = {
  line3114: {
    id: "3114",
    label: "Numéro national prévention suicide",
    detail: "3114 — Gratuit, confidentiel, 24h/24",
    type: "phone",
    href: "tel:3114",
    urgent: true,
  },
  samu: {
    id: "samu",
    label: "SAMU — Urgences médicales",
    detail: "15 ou 112",
    type: "phone",
    href: "tel:15",
    urgent: true,
  },
  filSanteJeunes: {
    id: "fil-sante-jeunes",
    label: "Fil Santé Jeunes",
    detail: "0 800 235 236 — Gratuit, anonyme",
    type: "phone",
    href: "tel:0800235236",
    urgent: false,
  },
  monSoutienPsy: {
    id: "mon-soutien-psy",
    label: "Mon Soutien Psy",
    detail: "Séances remboursées avec un psychologue",
    type: "website",
    href: "https://monpsy.sante.gouv.fr",
    urgent: false,
  },
  psycom: {
    id: "psycom",
    label: "Psycom — Annuaire psy",
    detail: "Trouver un professionnel de santé mentale",
    type: "website",
    href: "https://www.psycom.org",
    urgent: false,
  },
  medecinTraitant: {
    id: "medecin-traitant",
    label: "Médecin traitant",
    detail: "Premier interlocuteur — orientation spécialisée",
    type: "person",
    urgent: false,
  },
  proche: {
    id: "proche",
    label: "Parler à un proche de confiance",
    detail: "Parent, ami, membre de la famille",
    type: "person",
    urgent: false,
  },
  procheEnfant: {
    id: "proche-enfant",
    label: "Parler à un adulte de confiance",
    detail: "Parent, professeur, infirmière scolaire",
    type: "person",
    urgent: false,
  },
  enfanceEnDanger: {
    id: "119",
    label: "119 — Allô Enfance en danger",
    detail: "Gratuit, confidentiel, 24h/24 — enfants, ados et jeunes majeurs en danger",
    type: "phone",
    href: "tel:119",
    urgent: true,
  },
  antiHarcelement: {
    id: "3018",
    label: "3018 — Cyberharcèlement",
    detail: "Gratuit, anonyme, confidentiel, 7j/7 · 9h–23h — réseaux sociaux et internet",
    type: "phone",
    href: "tel:3018",
    urgent: false,
  },
  harcelementScolaire: {
    id: "3020",
    label: "3020 — Harcèlement scolaire",
    detail: "Gratuit, lun–ven · 9h–20h — harcèlement à l'école",
    type: "phone",
    href: "tel:3020",
    urgent: false,
  },
  violencesFemmes: {
    id: "3919",
    label: "3919 — Arrêtons les violences",
    detail: "Gratuit, anonyme, 24h/24 — femmes victimes de violences",
    type: "phone",
    href: "tel:3919",
    urgent: true,
  },
};

// ─── MICRO-ACTIONS ────────────────────────────────────────────────────────────

export const MICRO_ACTIONS: Record<string, MicroAction> = {

  // — Respiration & ancrage —
  breathing478: {
    id: "breathing-478",
    title: "Respiration 4-7-8",
    description: "Inspire 4 secondes · retiens 7 secondes · expire 8 secondes. Répète 4 fois.",
    duration: "2 min",
    brick: "mindfulness",
  },
  breathingCoherent: {
    id: "breathing-coherent",
    title: "Respiration cohérente",
    description: "Inspire doucement 5 secondes par le nez · expire 5 secondes par la bouche. Continue 3 minutes.",
    duration: "3 min",
    brick: "mindfulness",
  },
  grounding54321: {
    id: "grounding-54321",
    title: "Ancrage 5-4-3-2-1",
    description: "Nomme : 5 choses que tu vois · 4 que tu entends · 3 que tu touches · 2 odeurs · 1 goût.",
    duration: "3 min",
    brick: "mindfulness",
  },
  bodyScan: {
    id: "body-scan",
    title: "Scan corporel rapide",
    description: "Ferme les yeux. Pars des pieds, remonte lentement vers la tête. Note les tensions sans les juger.",
    duration: "3 min",
    brick: "mindfulness",
  },

  // — Activation comportementale (CBT) —
  pleasantActivity: {
    id: "pleasant-activity",
    title: "Une petite chose agréable",
    description: "Choisis une activité simple qui te fait du bien (musique, promenade, dessin…) et planifie-la aujourd'hui.",
    duration: "5 min de planif",
    brick: "cbt_activation",
  },
  journaling: {
    id: "journaling",
    title: "Journal des émotions",
    description: "Écris 3 lignes : ce qui s'est passé · ce que tu as ressenti · ce dont tu as besoin.",
    duration: "5 min",
    brick: "cbt_activation",
  },
  journalingKids: {
    id: "journaling-kids",
    title: "Mon carnet de ressentis",
    description: "Dessine ou écris : qu'est-ce qui s'est passé aujourd'hui ? Comment tu t'es senti ? Qu'est-ce qui t'aurait aidé ?",
    duration: "5 min",
    brick: "cbt_activation",
  },
  successJournal: {
    id: "success-journal",
    title: "Journal des réussites",
    description: "Note 3 choses que tu as bien faites aujourd'hui — même petites. Les réussites méritent d'être reconnues.",
    duration: "3 min",
    brick: "cbt_activation",
  },

  // — Restructuration cognitive (CBT) —
  thoughtChallenge: {
    id: "thought-challenge",
    title: "Questionner la pensée négative",
    description: "Identifie une pensée difficile. Demande-toi : est-ce un fait ou une interprétation ? Quelle preuve j'ai pour et contre ?",
    duration: "5–10 min",
    brick: "cbt_restructuring",
  },
  thoughtChallengeKids: {
    id: "thought-challenge-kids",
    title: "Est-ce vraiment vrai ?",
    description: "Pense à quelque chose de triste ou de difficile. Demande-toi : est-ce que c'est vraiment comme ça ? Y a-t-il une autre façon de voir ?",
    duration: "5 min",
    brick: "cbt_restructuring",
  },
  identifyTrigger: {
    id: "identify-trigger",
    title: "Identifier le déclencheur",
    description: "Qu'est-ce qui a déclenché cette émotion ? Situation · pensée · sensation physique ? Écrire aide à démêler.",
    duration: "5 min",
    brick: "cbt_restructuring",
  },

  // — ACT —
  actObserve: {
    id: "act-observe",
    title: "Observer sans juger",
    description: "Pose ta main sur ta poitrine. Dis-toi : 'Je remarque que je ressens de l'anxiété.' Tu n'es pas cette émotion — tu l'observes.",
    duration: "2 min",
    brick: "act",
  },
  actValues: {
    id: "act-values",
    title: "Ce qui compte pour moi",
    description: "Écris une valeur importante pour toi (famille, liberté, créativité…). Comment cette valeur peut-elle guider une petite action aujourd'hui ?",
    duration: "5 min",
    brick: "act",
  },

  // — Psychoéducation —
  eduStress: {
    id: "edu-stress",
    title: "Comprendre le stress",
    description: "Le stress est une réponse d'alerte normale. Il devient problématique quand il dure trop longtemps. Ton corps te signale qu'il a besoin d'aide.",
    duration: "2 min",
    brick: "psychoeducation",
  },
  eduBurnout: {
    id: "edu-burnout",
    title: "Comprendre l'épuisement",
    description: "L'épuisement s'installe progressivement. Ce n'est pas une faiblesse — c'est un signal que tes ressources sont mobilisées au maximum. Il est temps de les recharger.",
    duration: "2 min",
    brick: "psychoeducation",
  },
  eduAnxiety: {
    id: "edu-anxiety",
    title: "Comprendre l'anxiété",
    description: "L'anxiété anticipe le danger pour te protéger. Quand elle s'emballe, elle peut devenir envahissante. Elle se traite très bien avec les bonnes techniques.",
    duration: "2 min",
    brick: "psychoeducation",
  },
  chargeMentale: {
    id: "charge-mentale",
    title: "Bilan de charge mentale",
    description: "Liste tout ce qui occupe ton esprit. Entoure ce qui est urgent et important. Barre ce qui peut attendre. Identifie ce que tu peux déléguer ou supprimer.",
    duration: "10 min",
    brick: "psychoeducation",
  },
  microPause: {
    id: "micro-pause",
    title: "Micro-pause récupératrice",
    description: "Pose tout. 3 minutes sans écran, sans tâche. Assis confortablement. Juste être là. Le cerveau se réinitialise en quelques minutes de repos actif.",
    duration: "3 min",
    brick: "psychoeducation",
  },
  sleepHygiene: {
    id: "sleep-hygiene",
    title: "Hygiène de sommeil",
    description: "Ce soir : même heure de coucher, pas d'écran 30 min avant, chambre fraîche. Le sommeil est le premier outil de récupération psychique.",
    duration: "Ce soir",
    brick: "psychoeducation",
  },

  // — Soutien social —
  nameTrustedPerson: {
    id: "name-trusted-person",
    title: "Nommer une personne de confiance",
    description: "Pense à quelqu'un à qui tu pourrais parler. Pas besoin d'avoir une grande conversation — un simple 'je ne suis pas au mieux en ce moment' peut suffire.",
    duration: "Quand tu es prêt",
    brick: "social_support",
  },
  gratitude: {
    id: "gratitude",
    title: "3 gratitudes",
    description: "Note 3 choses pour lesquelles tu es reconnaissant aujourd'hui. Même petites. La gratitude renforce les circuits du bien-être.",
    duration: "3 min",
    brick: "cbt_activation",
  },
  positiveAnchor: {
    id: "positive-anchor",
    title: "Ancrage positif",
    description: "Ferme les yeux. Rappelle un moment de bonheur récent. Revivez-le en détail — sons, odeurs, sensations. Gardez cette image quelques secondes.",
    duration: "3 min",
    brick: "mindfulness",
  },
};

// ─── MESSAGES PAR PROFIL ET MODE ─────────────────────────────────────────────
// Structure : message empathique + lecture émotionnelle (combinés)
// Source : scripts UX/cliniques v2 — Kids & Adult × 3 niveaux

type ModeMessages = { kids: string; adult: string };

export const PROFILE_MESSAGES: Record<string, Record<number, ModeMessages>> = {
  joy: {
    0: {
      kids: "C'est chouette de sentir autant de joie. Quand on se sent joyeux, c'est souvent que quelque chose nous fait du bien dans notre cœur, dans notre tête ou dans notre journée.",
      adult: "C'est précieux de ressentir cet élan positif. La joie est souvent le signe que quelque chose vous nourrit, vous stimule ou vous fait du bien — une ressource réelle.",
    },
  },
  calm: {
    0: {
      kids: "C'est une belle sensation d'être calme. Ton corps et ton cœur ont l'air tranquilles. Quand on se sent zen, on se sent souvent en sécurité, posé et bien.",
      adult: "C'est un état précieux qui mérite d'être remarqué. Le calme indique souvent un meilleur alignement entre votre corps, vos pensées et votre environnement.",
    },
  },
  pride: {
    0: {
      kids: "Bravo — c'est important de voir ce que tu as réussi. Être fier, c'est sentir qu'on a fait quelque chose de bien ou qu'on a avancé.",
      adult: "C'est important de reconnaître ce que vous avez accompli. La fierté saine renforce l'estime de soi et le sentiment d'avancer dans la bonne direction.",
    },
  },
  sadness: {
    1: {
      kids: "Je comprends. Parfois, on a le cœur un peu lourd, et c'est normal. La tristesse peut arriver quand quelque chose nous a déçu, manqué ou blessé.",
      adult: "Je comprends. Avoir le cœur lourd peut être difficile à porter. La tristesse apparaît souvent quand quelque chose a manqué, déçu, blessé ou épuisé.",
    },
    2: {
      kids: "On dirait que quelque chose t'a vraiment touché. Cette tristesse prend beaucoup de place, et tu n'as pas à la porter seul.",
      adult: "Cette tristesse semble peser davantage. Ce que vous traversez mérite attention et soutien. Parler à quelqu'un peut vraiment alléger ce poids.",
    },
    3: {
      kids: "Ce que tu ressens est important. Il y a des personnes formées pour t'aider. Tu n'es pas seul — même si ça peut sembler comme ça en ce moment.",
      adult: "La persistance de cette tristesse mérite un accompagnement professionnel. Vous traversez quelque chose de difficile, et vous méritez un vrai soutien.",
    },
    4: {
      kids: "Ce que tu ressens est très important. Tu dois le dire tout de suite à un adulte. Tu n'es pas seul — des personnes sont là pour toi maintenant.",
      adult: "Ce que vous exprimez semble très difficile à porter. Vous méritez une aide humaine maintenant. Vous n'êtes pas seul(e) — des personnes sont disponibles immédiatement.",
    },
  },
  anger: {
    1: {
      kids: "Je vois que quelque chose t'a vraiment énervé. La colère, c'est une émotion forte — elle dit souvent que quelque chose nous a semblé injuste ou blessant.",
      adult: "Je vois que quelque chose vous a touché fortement. La colère est souvent liée à une limite franchie, une injustice ou un sentiment d'impuissance.",
    },
    2: {
      kids: "Quand on est très fâché, ça peut être dur de réfléchir. Prends d'abord le temps de souffler — la colère a besoin d'espace pour se calmer.",
      adult: "La colère récurrente cache souvent de l'anxiété ou de l'épuisement sous-jacent. La colère contient souvent une information importante — l'objectif n'est pas de l'étouffer, mais de la canaliser.",
    },
    3: {
      kids: "Si tu as l'impression de ne plus contrôler ta colère, parler à un adulte de confiance peut vraiment t'aider à aller mieux.",
      adult: "Quand la colère devient envahissante, un professionnel peut vous donner des outils concrets pour la réguler et comprendre ce qu'elle exprime.",
    },
    4: {
      kids: "Tu as besoin d'aide maintenant. Un adulte de confiance peut t'aider à traverser ce moment.",
      adult: "Vous avez besoin d'un espace sécurisé maintenant. Il est important de ne pas rester seul(e) avec cette intensité.",
    },
  },
  fear: {
    1: {
      kids: "C'est courageux de le dire. La peur peut être très forte parfois. Ton corps essaie de te protéger — on peut apprendre à calmer cette alarme ensemble.",
      adult: "L'inquiétude peut prendre beaucoup de place. L'anxiété est une réponse de protection — quelques techniques simples permettent de la réguler efficacement.",
    },
    2: {
      kids: "Ta peur prend beaucoup de place. Quand la peur revient souvent, c'est important d'en parler. Tu n'as pas à la gérer seul.",
      adult: "Votre esprit anticipe beaucoup en ce moment. Une anxiété persistante peut s'aggraver sans accompagnement — des thérapies courtes sont très efficaces.",
    },
    3: {
      kids: "Ta peur est réelle et mérite d'être entendue par quelqu'un de formé pour ça. Tu n'as pas besoin d'affronter ça tout seul.",
      adult: "L'anxiété semble envahissante en ce moment. Ce niveau nécessite un soutien professionnel — un médecin ou psychologue peut vous aider rapidement.",
    },
    4: {
      kids: "Tu dois être aidé tout de suite. Aller vers un adulte de confiance maintenant, c'est la chose la plus courageuse à faire.",
      adult: "Ce que vous ressentez nécessite une aide humaine immédiate. Vous n'avez pas à affronter ça seul(e).",
    },
  },
  stress: {
    1: {
      kids: "Je comprends. Quand on est nerveux, le corps peut se sentir tout tendu. Parfois, le corps réagit avant même qu'on comprenne exactement pourquoi.",
      adult: "Je comprends. Être sous pression fatigue à la fois le corps et l'esprit. Un stress ponctuel est adaptif — quelques techniques permettent de relâcher rapidement.",
    },
    2: {
      kids: "Tu sembles vraiment tendu en ce moment. Si tu te sens souvent comme ça, c'est important d'en parler à quelqu'un de confiance.",
      adult: "La pression est importante. Un stress chronique use les ressources physiques et psychiques. Il est temps de faire un bilan honnête de votre charge.",
    },
    3: {
      kids: "Quand le stress devient trop fort, un adulte de confiance ou un professionnel peut vraiment t'aider.",
      adult: "La surcharge devient critique. Ce niveau de stress dépasse les capacités d'auto-gestion. Une consultation médicale est recommandée.",
    },
    4: {
      kids: "On doit t'aider à te calmer maintenant. Dis-le à un adulte de confiance tout de suite.",
      adult: "Vous avez besoin de soutien maintenant. Ne restez pas seul(e) avec cette pression.",
    },
  },
  tiredness: {
    1: {
      kids: "Je comprends. Quand on est très fatigué, tout peut sembler plus difficile. Ton corps te dit qu'il a besoin de repos — l'écouter, c'est important.",
      adult: "La fatigue peut rendre chaque chose plus lourde. Elle peut être physique, mentale ou émotionnelle. Votre corps demande une récupération — ne l'ignorez pas.",
    },
    2: {
      kids: "Ton corps est vraiment très fatigué en ce moment. Ça mérite attention — il y a sûrement quelque chose qu'on peut alléger ensemble.",
      adult: "La fatigue s'installe profondément. Un épuisement de ce niveau peut signaler un burn-out débutant. Un regard professionnel est utile.",
    },
    3: {
      kids: "Cette fatigue intense mérite d'être partagée avec un adulte ou un médecin. Tu as besoin d'aide pour récupérer.",
      adult: "L'épuisement que vous décrivez dépasse la fatigue normale. Le repos n'est pas un échec — et une consultation médicale est nécessaire.",
    },
    4: {
      kids: "Tu as besoin d'aide maintenant. Dis à un adulte de confiance ce que tu ressens.",
      adult: "Cet épuisement est un signal d'alarme. Il est important de consulter un professionnel dès maintenant.",
    },
  },
};

// ─── CLÔTURES PAR ÉMOTION ET NIVEAU ──────────────────────────────────────────
// Phrase de clôture rassurante affichée en bas de l'écran Solutions

export const PROFILE_CLOSINGS: Record<string, Record<number, ModeMessages>> = {
  joy: {
    0: {
      kids: "Ta joie compte. On peut la garder en souvenir pour plus tard.",
      adult: "Cet état est une ressource. Il peut devenir un point d'appui pour les jours plus lourds.",
    },
  },
  calm: {
    0: {
      kids: "Ce calme est une force. Tu pourras y revenir quand tu en as besoin.",
      adult: "Ce calme peut devenir une ressource à laquelle revenir plus tard.",
    },
  },
  pride: {
    0: {
      kids: "Tu peux être fier de toi. Tes efforts comptent vraiment.",
      adult: "Reconnaître ce qui va bien fait aussi partie de l'équilibre psychique.",
    },
  },
  sadness: {
    1: {
      kids: "Tu as le droit d'être triste. Tu n'as pas à garder ça tout seul.",
      adult: "Vous n'avez pas à tout porter seul. Ce ressenti mérite attention et douceur.",
    },
    2: {
      kids: "Tu n'es pas seul. On peut y aller ensemble, doucement.",
      adult: "Prendre soin de vous n'est pas un luxe, c'est une nécessité.",
    },
    3: {
      kids: "Des personnes formées sont là pour t'aider à aller mieux.",
      adult: "Vous méritez un accompagnement. Faire le premier pas, c'est déjà beaucoup.",
    },
    4: {
      kids: "Des personnes sont là pour toi maintenant. Tu n'es pas seul.",
      adult: "Ne restez pas seul(e). Des personnes sont disponibles immédiatement.",
    },
  },
  anger: {
    1: {
      kids: "Ta colère veut dire quelque chose. On peut l'écouter sans qu'elle prenne toute la place.",
      adult: "La colère contient une information. L'objectif n'est pas de l'étouffer, mais de la canaliser.",
    },
    2: {
      kids: "Souffler d'abord, comprendre ensuite — dans cet ordre.",
      adult: "Avant d'agir, baisser l'intensité. C'est une force, pas une faiblesse.",
    },
    3: {
      kids: "Un adulte de confiance peut t'aider à traverser ça.",
      adult: "Un professionnel peut vous donner des outils durables pour mieux réguler ces émotions.",
    },
    4: {
      kids: "Tu as besoin d'aide maintenant. Un adulte peut t'aider.",
      adult: "Vous avez besoin d'un espace sécurisé. Ne restez pas seul(e).",
    },
  },
  fear: {
    1: {
      kids: "Tu n'as pas besoin d'affronter ta peur tout seul.",
      adult: "Quand l'inquiétude est nommée, elle devient souvent un peu plus gérable.",
    },
    2: {
      kids: "Ta peur peut s'apaiser. On peut y aller doucement.",
      adult: "Avec les bons outils, l'anxiété se régule très bien. Vous n'êtes pas seul(e).",
    },
    3: {
      kids: "Un adulte ou un professionnel peut vraiment t'aider à aller mieux.",
      adult: "Un accompagnement professionnel peut transformer ce vécu. Faites le premier pas.",
    },
    4: {
      kids: "Tu n'es pas seul. Des personnes sont là pour toi maintenant.",
      adult: "Vous n'êtes pas seul(e). Des personnes sont disponibles immédiatement.",
    },
  },
  stress: {
    1: {
      kids: "Ce que tu ressens peut s'apaiser. On peut y aller doucement.",
      adult: "Vous n'avez pas besoin de résoudre toute la journée d'un coup. Une étape suffit.",
    },
    2: {
      kids: "Un pas à la fois. Tu n'as pas à tout gérer seul.",
      adult: "Réduire la pression est possible. Une décision à la fois.",
    },
    3: {
      kids: "Un adulte de confiance peut vraiment t'aider.",
      adult: "Parler de cette surcharge à un professionnel est une décision intelligente.",
    },
    4: {
      kids: "Tu n'es pas seul. Des personnes sont là pour toi.",
      adult: "Vous méritez du soutien maintenant. Ne restez pas seul(e) avec ça.",
    },
  },
  tiredness: {
    1: {
      kids: "Ton corps te parle. L'écouter, c'est important.",
      adult: "Le repos n'est pas un échec. C'est parfois la première étape utile.",
    },
    2: {
      kids: "Prendre soin de toi, c'est aussi te reposer quand ton corps le demande.",
      adult: "Récupérer n'est pas abandonner. C'est reprendre des forces intelligemment.",
    },
    3: {
      kids: "Un adulte ou un médecin peut t'aider à récupérer.",
      adult: "Consulter un professionnel est le geste le plus utile que vous puissiez faire maintenant.",
    },
    4: {
      kids: "Tu as besoin d'aide. Un adulte peut t'aider maintenant.",
      adult: "Ce signal d'alarme mérite une réponse immédiate. Vous n'êtes pas seul(e).",
    },
  },
};

// ─── MICRO-ACTIONS PAR PROFIL ET NIVEAU ─────────────────────────────────────

export const PROFILE_ACTIONS: Record<string, Record<number, { kids: MicroAction[]; adult: MicroAction[] }>> = {
  wellbeing: {
    0: {
      kids: [MICRO_ACTIONS.gratitude, MICRO_ACTIONS.journalingKids, MICRO_ACTIONS.positiveAnchor],
      adult: [MICRO_ACTIONS.gratitude, MICRO_ACTIONS.positiveAnchor, MICRO_ACTIONS.successJournal],
    },
  },
  adjustment: {
    1: {
      kids: [MICRO_ACTIONS.breathingCoherent, MICRO_ACTIONS.journalingKids, MICRO_ACTIONS.pleasantActivity],
      adult: [MICRO_ACTIONS.breathingCoherent, MICRO_ACTIONS.journaling, MICRO_ACTIONS.pleasantActivity],
    },
  },
  anxiety: {
    1: {
      kids: [MICRO_ACTIONS.breathing478, MICRO_ACTIONS.grounding54321, MICRO_ACTIONS.actObserve],
      adult: [MICRO_ACTIONS.breathing478, MICRO_ACTIONS.grounding54321, MICRO_ACTIONS.eduAnxiety],
    },
    2: {
      kids: [MICRO_ACTIONS.breathingCoherent, MICRO_ACTIONS.grounding54321, MICRO_ACTIONS.nameTrustedPerson],
      adult: [MICRO_ACTIONS.breathing478, MICRO_ACTIONS.eduAnxiety, MICRO_ACTIONS.thoughtChallenge],
    },
    3: {
      kids: [MICRO_ACTIONS.breathingCoherent, MICRO_ACTIONS.nameTrustedPerson],
      adult: [MICRO_ACTIONS.breathing478, MICRO_ACTIONS.eduAnxiety, MICRO_ACTIONS.nameTrustedPerson],
    },
  },
  depression: {
    1: {
      kids: [MICRO_ACTIONS.journalingKids, MICRO_ACTIONS.pleasantActivity, MICRO_ACTIONS.breathingCoherent],
      adult: [MICRO_ACTIONS.journaling, MICRO_ACTIONS.pleasantActivity, MICRO_ACTIONS.actObserve],
    },
    2: {
      kids: [MICRO_ACTIONS.thoughtChallengeKids, MICRO_ACTIONS.pleasantActivity, MICRO_ACTIONS.nameTrustedPerson],
      adult: [MICRO_ACTIONS.thoughtChallenge, MICRO_ACTIONS.journaling, MICRO_ACTIONS.nameTrustedPerson],
    },
    3: {
      kids: [MICRO_ACTIONS.breathingCoherent, MICRO_ACTIONS.nameTrustedPerson],
      adult: [MICRO_ACTIONS.actValues, MICRO_ACTIONS.nameTrustedPerson],
    },
  },
  burnout: {
    2: {
      kids: [MICRO_ACTIONS.microPause, MICRO_ACTIONS.sleepHygiene, MICRO_ACTIONS.breathingCoherent],
      adult: [MICRO_ACTIONS.chargeMentale, MICRO_ACTIONS.microPause, MICRO_ACTIONS.eduBurnout],
    },
    3: {
      kids: [MICRO_ACTIONS.microPause, MICRO_ACTIONS.nameTrustedPerson],
      adult: [MICRO_ACTIONS.microPause, MICRO_ACTIONS.eduBurnout, MICRO_ACTIONS.nameTrustedPerson],
    },
  },
  crisis: {
    4: {
      kids: [MICRO_ACTIONS.breathingCoherent],
      adult: [MICRO_ACTIONS.breathingCoherent],
    },
  },
};

// ─── RESSOURCES PAR NIVEAU ───────────────────────────────────────────────────

export const RESOURCES_BY_LEVEL: Record<number, { kids: Resource[]; adult: Resource[] }> = {
  0: { kids: [], adult: [] },
  1: { kids: [], adult: [] },
  2: {
    kids: [RESOURCES.procheEnfant, RESOURCES.antiHarcelement, RESOURCES.harcelementScolaire, RESOURCES.filSanteJeunes],
    adult: [RESOURCES.proche, RESOURCES.monSoutienPsy, RESOURCES.psycom],
  },
  3: {
    // 3114 + 119 inclus au niveau 3 — prévention active, danger potentiel
    kids: [RESOURCES.line3114, RESOURCES.enfanceEnDanger, RESOURCES.antiHarcelement, RESOURCES.harcelementScolaire, RESOURCES.procheEnfant],
    adult: [RESOURCES.medecinTraitant, RESOURCES.line3114, RESOURCES.violencesFemmes, RESOURCES.monSoutienPsy],
  },
  4: {
    kids: [RESOURCES.line3114, RESOURCES.enfanceEnDanger, RESOURCES.filSanteJeunes, RESOURCES.procheEnfant],
    adult: [RESOURCES.line3114, RESOURCES.violencesFemmes, RESOURCES.samu, RESOURCES.proche],
  },
};
