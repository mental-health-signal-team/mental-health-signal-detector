/**
 * scoringEngine — Fonctions pures d'analyse clinique
 *
 * Correctifs sécurité (v2) :
 *   Fix 1 — Fallback sans ML : utilise selfScore + DISTRESS_TEXT_SIGNALS
 *   Fix 2 — Règle du maximum : seuil de masquage abaissé à > 0.25 (bonus +0.20)
 *            le texte ne peut jamais abaisser le triage sous son propre signal
 *   Fix 3 — Dimensions cliniques vérifiées avant le null-guard ML
 *            → s'appliquent même si l'API est indisponible
 */

import type { ClinicalDimension, ClinicalProfile, DiagnosticProfile } from "../types/diagnostic";

export type DistressLevel = "light" | "elevated" | "critical";

// ─── Filet de sécurité absolu ────────────────────────────────────────────────
// Idéation suicidaire explicite + variantes indirectes (fardeau, disparition,
// absence de raison de vivre). Toute correspondance → "critical" immédiat.
export const CRITICAL_KEYWORDS = [
  // Idéation directe
  "suicide", "suicider", "me tuer", "mourir", "plus envie de vivre",
  "disparaître", "en finir", "me supprimer", "j'ai envie de mourir",
  "pensées suicidaires", "je veux mourir",
  "je n'en peux plus", "je veux disparaître", "je ne veux plus être là",
  "personne ne m'aime", "personne ne peut m'aider",
  "je ne veux plus vivre", "j'en ai marre de tout", "tout seul au monde",
  // Idéation indirecte — fardeau / absence perçue bénéfique (rec. clinicien)
  "ça serait mieux sans moi", "tout serait plus simple si je n'étais plus là",
  "plus de raison de vivre", "à quoi bon vivre", "je suis un fardeau",
  "personne ne remarquerait si je disparaissais",
  // EN — indirect suicidal ideation
  "better off without me", "no reason to live", "can't go on anymore",
];

// ─── Seuils de triage clinique ───────────────────────────────────────────────
export const SCORE_CRITICAL = 0.65;
export const SCORE_ELEVATED = 0.35;

// ─── Planchers cliniques recalibrés ─────────────────────────────────────────
export const EMOTION_FLOOR: Record<string, number> = {
  sadness: 0.35, fear: 0.35,
  anger: 0.30, stress: 0.25, tiredness: 0.30,
  joy: 0.0, calm: 0.0, pride: 0.0,
};

// ─── Signaux texte de détresse générale (Fix 1 & 3) ─────────────────────────
// Utilisés dans le fallback (API ML indisponible) pour détecter une incohérence
// entre l'émotion sélectionnée (positive) et le texte saisi (négatif).
export const DISTRESS_TEXT_SIGNALS = [
  "je me sens mal", "ça ne va pas", "je souffre", "j'ai du mal",
  "tout va mal", "je suis épuisé", "je pleure", "je suis triste",
  "j'en ai marre", "je suis perdu", "je suis seul", "je me sens seul",
  "j'ai peur", "je suis anxieux", "je suis stressé", "ça fait mal",
  "je ne suis pas bien", "je vais pas bien", "j'ai besoin d'aide",
  "i feel bad", "i'm struggling", "i'm suffering", "i feel terrible",
  "i'm not okay", "i'm not ok",
];

// ─── Détection de dimensions cliniques dans le texte ────────────────────────
export const DIMENSION_KEYWORDS: Record<ClinicalDimension, string[]> = {
  // Axe : épuisement motivationnel / énergétique / professionnel
  // Tri-dimensionnel : épuisement + cynisme/désengagement + inefficacité
  burnout: [
    // Épuisement (existant)
    "j'en peux plus", "à bout", "plus la force", "épuisé depuis",
    "plus d'énergie", "je tiens plus", "i can't go on", "burned out",
    "exhausted for weeks",
    // Épuisement complémentaire
    "je suis à plat", "plus aucune motivation", "running on empty",
    "no motivation left", "je suis vidé",
    // Cynisme / désengagement (rec. clinicien — discriminant burnout vs dépression)
    "je m'en fiche de tout", "plus de sens au travail", "je suis désengagé",
    "rien ne sert à rien", "i don't care anymore",
    // Inefficacité (rec. clinicien — 3e dimension Maslach)
    "je suis dépassé", "complètement dépassé", "je suis submergé", "je n'y arrive plus au travail",
    "overwhelmed", "je suis incompétent",
  ],
  // Axe : activation anxieuse — projection future + activation physiologique
  anxiety: [
    // Contrôle / fréquence (existant)
    "je ne contrôle plus", "tout le temps peur", "ça ne s'arrête pas",
    "j'angoisse", "attaque de panique", "je panique", "peur de tout",
    "anxiété", "rumination", "can't stop worrying", "panic attack",
    // Anticipation catastrophiste (rec. clinicien — projection future)
    "je m'inquiète tout le temps", "je pense au pire", "et si ça tourne mal",
    "j'anticipe le pire",
    // Hypervigilance / tension (rec. clinicien)
    "je suis tendu", "je n'arrive pas à me détendre", "always on edge",
    "je suis sur les nerfs",
    // Somatique (rec. clinicien — activation physiologique)
    "je tremble", "boule au ventre", "je n'arrive pas à dormir",
    "i can't sleep", "heart racing", "j'ai du mal à respirer",
  ],
  // Axe : registre cognitif / existentiel — dépression masquée
  // Triade classique : humeur ↓ + énergie ↓ + plaisir ↓
  depression_masked: [
    // Existentiel / sens (existant)
    "à quoi ça sert", "plus de plaisir", "je suis nul", "rien ne va",
    "plus rien ne m'intéresse", "je me sens vide", "sans espoir",
    // Anhédonie (rec. clinicien — plaisir ↓)
    "plus envie de rien", "rien ne me fait plaisir", "plus rien ne m'amuse",
    // Fatigue morale / charge (rec. clinicien — énergie ↓)
    "tout me coûte", "c'est trop lourd", "je suis épuisé moralement",
    // Ralentissement psychomoteur (rec. clinicien)
    "je n'avance pas", "je suis ralenti", "je n'arrive pas à me lever",
    // Isolement / invisibilité (rec. clinicien)
    "je me sens seul", "personne ne comprend", "je suis invisible",
    "je ne sers à rien", "je suis inutile",
    // EN (existant + nouveau)
    "worthless", "hopeless", "nothing matters", "no point",
    "i feel empty", "i feel alone",
  ],
  // Axe : impulsivité / perte de contrôle comportemental
  dysregulation: [
    // Passage à l'acte / auto-agression (existant)
    "j'explose", "je casse tout", "je me suis blessé", "je perds le contrôle",
    "je me fais du mal", "self-harm", "je me coupe", "envie de frapper",
    // Impulsivité / fuite (rec. clinicien)
    "j'ai envie de tout lâcher", "je veux tout casser", "je n'en peux plus de moi",
    "out of control",
  ],
};

// ─── Sanitisation du score ML ────────────────────────────────────────────────
export function sanitizeMlScore(raw: unknown): number | null {
  if (typeof raw !== "number" || !isFinite(raw)) return null;
  return Math.min(1, Math.max(0, raw));
}

// ─── Détection de dimensions cliniques ──────────────────────────────────────
export function detectClinicalDimensions(text: string): ClinicalDimension[] {
  const lower = text.toLowerCase();
  return (Object.keys(DIMENSION_KEYWORDS) as ClinicalDimension[]).filter(
    (dim) => DIMENSION_KEYWORDS[dim].some((kw) => lower.includes(kw))
  );
}

// ─── Score final fusionné (Fix 2 — règle du maximum) ────────────────────────
// Seuil de masquage abaissé : mlScore > 0.25 (vs 0.50 avant) pour les émotions
// positives (floor < 0.2 : joy/calm/pride). Bonus porté à +0.20 (vs +0.15).
// Garantit que "joy + texte modérément négatif" n'est jamais sous-estimé.
export function computeFinalScore(
  mlScore: number,
  emotionId: string,
  selfScore: number | null
): number {
  const floor = EMOTION_FLOOR[emotionId] ?? 0.0;

  // Détection de masquage émotion/texte : émotion positive + texte distressant
  const isPositiveEmotion = floor < 0.2;
  const isMasking = isPositiveEmotion && mlScore > 0.25;
  const mlAdjusted = Math.min(1.0, mlScore + (isMasking ? 0.20 : 0));

  const blended = selfScore !== null
    ? selfScore * 0.45 + mlAdjusted * 0.55
    : mlAdjusted;

  // Le texte ne peut pas abaisser le triage en dessous de son propre signal
  return Math.min(1.0, Math.max(blended, floor, mlAdjusted));
}

// ─── Niveau de détresse ──────────────────────────────────────────────────────
export function getDistressLevel(
  mlScore: number | null,
  userText: string,
  emotionId: string,
  dimensions: ClinicalDimension[],
  selfScore: number | null
): DistressLevel {
  const lowerText = userText.toLowerCase();

  // 1. Sécurité absolue : keywords critiques
  if (CRITICAL_KEYWORDS.some((kw) => lowerText.includes(kw))) return "critical";

  // 2. Dysrégulation → toujours au moins elevated
  if (dimensions.includes("dysregulation")) return "elevated";

  // 3. Fix 3 — Dimensions cliniques : signal indépendant du score ML
  //    S'applique même sans API (slim deployment) — évite l'incohérence
  //    émotion positive + texte angoissant sans modèle disponible.
  if (dimensions.length > 0) return "elevated";

  // 4. Fusion ML + self-report + émotion (Fix 2)
  if (mlScore !== null) {
    const finalScore = computeFinalScore(mlScore, emotionId, selfScore);
    if (finalScore >= SCORE_CRITICAL) return "critical";
    if (finalScore >= SCORE_ELEVATED) return "elevated";
    return "light";
  }

  // 5. Fix 1 — Fallback si API indisponible
  //    Ordre de priorité : selfScore > plancher émotionnel > signaux texte
  const floor = EMOTION_FLOOR[emotionId] ?? 0.0;
  const fallbackScore = selfScore !== null ? Math.max(floor, selfScore) : floor;
  if (fallbackScore >= SCORE_CRITICAL) return "critical";
  if (fallbackScore >= SCORE_ELEVATED) return "elevated";
  // Signaux texte généraux : détecte "joy + je me sens mal" sans ML
  if (DISTRESS_TEXT_SIGNALS.some((kw) => lowerText.includes(kw))) return "elevated";
  return "light";
}

// ─── Profil clinique synthétique ────────────────────────────────────────────
export function deriveClinicalProfile(
  distressLevel: DistressLevel,
  emotionId: string,
  dimensions: ClinicalDimension[]
): ClinicalProfile {
  if (distressLevel === "critical") return "crisis";
  if (dimensions.includes("dysregulation")) return "crisis";
  if (
    dimensions.includes("burnout") ||
    (emotionId === "tiredness" && dimensions.includes("depression_masked"))
  ) return "burnout";
  if (dimensions.includes("anxiety") || emotionId === "fear") return "anxiety";
  if (dimensions.includes("depression_masked") || emotionId === "sadness") return "depression";
  if (distressLevel === "elevated") return "adjustment";
  return "wellbeing";
}

// ─── Pipeline complet ────────────────────────────────────────────────────────
export function buildDiagnosticProfile(params: {
  emotionId: string;
  mode: "kids" | "adult";
  userText: string;
  mlScore: number | null;
  selfScore: number | null;
  selfReportAnswers: number[] | null;
}): DiagnosticProfile {
  const { emotionId, mode, userText, mlScore, selfScore, selfReportAnswers } = params;
  const clinicalDimensions = detectClinicalDimensions(userText);
  const distressLevel = getDistressLevel(mlScore, userText, emotionId, clinicalDimensions, selfScore);
  const finalScore = mlScore !== null ? computeFinalScore(mlScore, emotionId, selfScore) : null;
  const clinicalProfile = deriveClinicalProfile(distressLevel, emotionId, clinicalDimensions);

  return {
    emotionId,
    mode,
    userText,
    selfScore,
    selfReportAnswers,
    mlScore,
    finalScore,
    distressLevel,
    clinicalDimensions,
    clinicalProfile,
  };
}
