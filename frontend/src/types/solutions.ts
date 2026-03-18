/**
 * Types Phase 3 — Moteur de recommandation clinique
 * Entrée : DiagnosticProfile  →  Sortie : SolutionResponse
 */

import type { ClinicalProfile } from "./diagnostic";

// ─── Niveau de triage (0–4) ──────────────────────────────────────────────────

export type TriageLevel = 0 | 1 | 2 | 3 | 4;

// ─── Briques thérapeutiques ──────────────────────────────────────────────────

export type TherapeuticBrick =
  | "cbt_activation"        // Activation comportementale
  | "cbt_restructuring"     // Restructuration cognitive
  | "act"                   // Acceptation & Engagement
  | "mindfulness"           // Pleine conscience / respiration
  | "psychoeducation"       // Comprendre son état
  | "social_support"        // Inciter le lien social
  | "professional"          // Orientation professionnelle
  | "crisis";               // Protocole de crise

// ─── Micro-action concrète ───────────────────────────────────────────────────

export interface MicroAction {
  id: string;
  title: string;                    // Titre court affiché
  description: string;              // Instruction claire
  duration: string;                 // "2 min", "5 min", etc.
  brick: TherapeuticBrick;
}

// ─── Ressource externe ──────────────────────────────────────────────────────

export type ResourceType = "phone" | "website" | "app" | "person";

export interface Resource {
  id: string;
  label: string;
  detail: string;
  type: ResourceType;
  href?: string;            // tel:3114, https://...
  website?: string;         // Lien secondaire "Voir le site →" (pour ressources téléphoniques)
  urgent: boolean;          // true = afficher en premier, style alerte
}

// ─── Réponse complète du moteur ──────────────────────────────────────────────

export interface SolutionResponse {
  level: TriageLevel;
  clinicalProfile: ClinicalProfile;
  message: string;                  // Message empathique + lecture émotionnelle
  closing: string;                  // Phrase de clôture rassurante
  microActions: MicroAction[];      // 2–3 actions concrètes
  therapeuticBrick: TherapeuticBrick;
  resources: Resource[];            // Vides si niveau 0–1
  escalationRequired: boolean;      // true si niveau ≥ 4
}
