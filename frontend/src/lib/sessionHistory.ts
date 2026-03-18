/**
 * sessionHistory — Historique local des sessions émotionnelles
 *
 * Stockage : localStorage (on-device uniquement, aucune donnée serveur)
 * Rétention : 30 jours glissants, 10 sessions max
 * Déduplique automatiquement si la même session est soumise dans les 5 minutes
 *
 * Utilisé pour afficher la tendance (amélioration / stable / dégradation)
 * sur l'écran Solutions, sans aucune transmission de données personnelles.
 */

import type { ClinicalProfile } from "../types/diagnostic";
import type { TriageLevel } from "../types/solutions";

export interface SessionRecord {
  date: string;             // ISO string
  level: TriageLevel;
  emotionId: string;
  finalScore: number | null;
  clinicalProfile: ClinicalProfile;
}

export type Trend = "improving" | "stable" | "worsening";

const STORAGE_KEY  = "mh_session_history";
const MAX_SESSIONS = 10;
const MAX_AGE_MS   = 30 * 24 * 60 * 60 * 1000; // 30 jours
const DEDUP_MS     = 5  * 60 * 1000;            // 5 minutes

export function getSessions(): SessionRecord[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const all: SessionRecord[] = JSON.parse(raw);
    const cutoff = Date.now() - MAX_AGE_MS;
    return all.filter((s) => new Date(s.date).getTime() > cutoff);
  } catch {
    return [];
  }
}

export function saveSession(record: SessionRecord): void {
  try {
    const sessions = getSessions();

    // Déduplique : ne pas sauvegarder si une session existe déjà dans les 5 dernières minutes
    const dedupCutoff = Date.now() - DEDUP_MS;
    if (sessions.some((s) => new Date(s.date).getTime() > dedupCutoff)) return;

    sessions.push(record);
    // Garder les MAX_SESSIONS les plus récentes
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions.slice(-MAX_SESSIONS)));
  } catch {
    // localStorage indisponible (mode privé, quota dépassé) → silencieux
  }
}

/**
 * Calcule la tendance sur les 2 dernières sessions.
 * Retourne null si moins de 2 sessions disponibles.
 *
 * Seuil : delta de ≥ 1 niveau pour considérer une évolution significative.
 */
export function getTrend(sessions: SessionRecord[]): Trend | null {
  if (sessions.length < 2) return null;
  const delta = sessions[sessions.length - 1].level - sessions[sessions.length - 2].level;
  if (delta <= -1) return "improving";
  if (delta >= 1)  return "worsening";
  return "stable";
}
