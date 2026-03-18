/**
 * Moteur de recommandation clinique — Phase 3
 *
 * Entrée  : DiagnosticProfile (Phase 2)
 * Sortie  : SolutionResponse (Phase 3)
 *
 * Logique : triage en 5 niveaux (0–4) → protocole adapté
 * Référence : Stepped-care model (NICE) + CBT + ACT
 */

import type { DiagnosticProfile, ClinicalProfile } from "../types/diagnostic";
import type { SolutionResponse, TriageLevel, TherapeuticBrick } from "../types/solutions";
import {
  PROFILE_MESSAGES,
  PROFILE_CLOSINGS,
  PROFILE_ACTIONS,
  RESOURCES,
  RESOURCES_BY_LEVEL,
  MICRO_ACTIONS,
} from "../data/solutions";

// ─── Mapping ClinicalProfile → TriageLevel ───────────────────────────────────

function mapToTriageLevel(profile: DiagnosticProfile): TriageLevel {
  const { clinicalProfile, distressLevel, clinicalDimensions } = profile;

  // Niveau 4 — urgence critique
  if (clinicalProfile === "crisis") return 4;

  // Niveau 3 — alerte clinique
  if (distressLevel === "critical") return 3;
  if (
    clinicalProfile === "burnout" &&
    clinicalDimensions.includes("burnout") &&
    (profile.emotionId === "tiredness" || profile.emotionId === "sadness")
  )
    return 3;

  // Niveau 2 — détresse modérée
  if (distressLevel === "elevated") return 2;
  if (clinicalProfile === "anxiety" || clinicalProfile === "depression" || clinicalProfile === "burnout")
    return 2;

  // Niveau 1 — inconfort léger
  if (clinicalProfile === "adjustment") return 1;

  // Niveau 0 — bien-être
  return 0;
}

// ─── Sélection de la brique thérapeutique dominante ─────────────────────────

function selectBrick(profile: ClinicalProfile, level: TriageLevel): TherapeuticBrick {
  if (level >= 4) return "crisis";
  if (level >= 3) return "professional";
  switch (profile) {
    case "anxiety":    return "mindfulness";
    case "burnout":    return "psychoeducation";
    case "depression": return level >= 2 ? "cbt_restructuring" : "cbt_activation";
    case "adjustment": return "cbt_activation";
    case "wellbeing":  return "cbt_activation";
    default:           return "mindfulness";
  }
}

// ─── Sélection du message empathique ────────────────────────────────────────

function selectMessage(profile: DiagnosticProfile, level: TriageLevel): string {
  const mode = profile.mode;
  const emotionId = profile.emotionId;

  // Cherche d'abord un message spécifique à l'émotion + niveau
  const emotionMessages = PROFILE_MESSAGES[emotionId];
  if (emotionMessages?.[level]) {
    return emotionMessages[level][mode];
  }

  // Remonte au niveau le plus proche disponible pour cette émotion
  for (let l = level - 1; l >= 0; l--) {
    if (emotionMessages?.[l]) return emotionMessages[l][mode];
  }

  // Fallback générique par niveau
  const fallbacks: Record<TriageLevel, ModeMessages> = {
    0: {
      kids: "Tu vas bien et c'est précieux. Continue à prendre soin de toi !",
      adult: "Vous êtes dans un bon équilibre. Prenez le temps de l'apprécier.",
    },
    1: {
      kids: "Ce que tu ressens est normal. Voici quelque chose qui peut t'aider.",
      adult: "Ce que vous ressentez est compréhensible. Quelques petites actions peuvent faire la différence.",
    },
    2: {
      kids: "Je sens que c'est difficile en ce moment. Tu n'es pas seul.",
      adult: "Ce que vous traversez mérite attention et soutien.",
    },
    3: {
      kids: "Ce que tu ressens est important. Des personnes formées peuvent t'aider.",
      adult: "Vous traversez quelque chose de difficile. Un professionnel peut vous accompagner.",
    },
    4: {
      kids: "Tu traverses un moment très difficile. Tu n'es pas seul — des personnes sont là pour toi maintenant.",
      adult: "Vous traversez une situation très difficile. Vous n'êtes pas seul(e). Des personnes sont disponibles immédiatement.",
    },
  };

  return fallbacks[level][mode];
}

type ModeMessages = { kids: string; adult: string };

// ─── Sélection de la phrase de clôture ──────────────────────────────────────

function selectClosing(profile: DiagnosticProfile, level: TriageLevel): string {
  const { emotionId, mode } = profile;

  const emotionClosings = PROFILE_CLOSINGS[emotionId];
  if (emotionClosings?.[level]) return emotionClosings[level][mode];

  // Remonte au niveau le plus proche
  for (let l = level - 1; l >= 0; l--) {
    if (emotionClosings?.[l]) return emotionClosings[l][mode];
  }

  // Fallback génériques
  const fallbacks: Record<TriageLevel, ModeMessages> = {
    0: { kids: "Prends soin de toi.", adult: "Prenez soin de vous." },
    1: { kids: "Tu n'es pas seul.", adult: "Vous n'êtes pas seul(e)." },
    2: { kids: "Un pas à la fois.", adult: "Un pas à la fois." },
    3: { kids: "Des personnes peuvent t'aider.", adult: "Des professionnels peuvent vous accompagner." },
    4: { kids: "Des personnes sont là pour toi maintenant.", adult: "Ne restez pas seul(e). Des personnes sont disponibles immédiatement." },
  };

  return fallbacks[level][mode];
}

// ─── Sélection des micro-actions ─────────────────────────────────────────────

function selectActions(profile: DiagnosticProfile, level: TriageLevel) {
  const { clinicalProfile, mode } = profile;

  // Cherche les actions pour ce profil + niveau exact
  const profileActions = PROFILE_ACTIONS[clinicalProfile];
  const levelActions = profileActions?.[level];
  if (levelActions) return levelActions[mode];

  // Remonte au niveau le plus proche disponible
  for (let l = level - 1; l >= 0; l--) {
    const fallback = profileActions?.[l];
    if (fallback) return fallback[mode];
  }

  // Fallback universel : respiration + ancrage
  return [MICRO_ACTIONS.breathingCoherent, MICRO_ACTIONS.grounding54321];
}

// ─── Sélection contextuelle des ressources ───────────────────────────────────
//
// Enrichit la liste de base (niveau × mode) avec des ressources pertinentes
// selon l'émotion choisie et les dimensions cliniques détectées.

function selectResources(profile: DiagnosticProfile, level: TriageLevel) {
  const { mode, emotionId, clinicalDimensions } = profile;
  const base = [...RESOURCES_BY_LEVEL[level][mode]];

  const ids = new Set(base.map((r) => r.id));
  const add = (r: typeof RESOURCES[string]) => { if (!ids.has(r.id)) { base.push(r); ids.add(r.id); } };

  if (mode === "adult" && level >= 2) {
    // 3919 — contexte violence/danger : anger, fear, dysrégulation
    if (
      emotionId === "anger" ||
      emotionId === "fear" ||
      clinicalDimensions.includes("dysregulation")
    ) {
      add(RESOURCES.violencesFemmes);
    }
  }

  if (mode === "kids" && level >= 2) {
    // 119 — contexte danger physique : anger ou fear
    if (emotionId === "anger" || emotionId === "fear") {
      add(RESOURCES.enfanceEnDanger);
    }
    // 3018/3020 — contexte harcèlement : stress ou anger
    if (emotionId === "stress" || emotionId === "anger") {
      add(RESOURCES.antiHarcelement);
      add(RESOURCES.harcelementScolaire);
    }
  }

  return base;
}

// ─── Moteur principal ────────────────────────────────────────────────────────

export function computeSolution(profile: DiagnosticProfile): SolutionResponse {
  const level = mapToTriageLevel(profile);
  const brick = selectBrick(profile.clinicalProfile, level);
  const message = selectMessage(profile, level);
  const closing = selectClosing(profile, level);
  const microActions = selectActions(profile, level);
  const resources = selectResources(profile, level);

  return {
    level,
    clinicalProfile: profile.clinicalProfile,
    message,
    closing,
    microActions,
    therapeuticBrick: brick,
    resources,
    escalationRequired: level >= 4,
  };
}
