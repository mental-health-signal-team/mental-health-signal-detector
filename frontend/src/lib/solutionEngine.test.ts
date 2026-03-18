/**
 * Tests unitaires — solutionEngine.ts (computeSolution)
 *
 * Couverture :
 *   - Triage 0-4 depuis le profil clinique
 *   - escalationRequired selon le niveau
 *   - Brique thérapeutique par profil
 *   - 3114 présent aux niveaux 3 et 4
 *   - Ressources vides au niveau 0
 *   - Ressources contextuelles (harcèlement, violences)
 *   - Adaptation kids/adult (messages non vides)
 *   - Sécurité absolue : profil crisis → niveau 4
 */

import { describe, it, expect } from "vitest";
import { computeSolution } from "./solutionEngine";
import type { DiagnosticProfile } from "../types/diagnostic";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function makeProfile(overrides: Partial<DiagnosticProfile> = {}): DiagnosticProfile {
  return {
    emotionId: "stress",
    mode: "adult",
    userText: "je suis stressé",
    selfScore: null,
    selfReportAnswers: null,
    mlScore: 0.4,
    finalScore: 0.4,
    distressLevel: "elevated",
    clinicalDimensions: [],
    clinicalProfile: "adjustment",
    ...overrides,
  };
}

// ─── Triage ───────────────────────────────────────────────────────────────────

describe("computeSolution — triage", () => {
  it("crisis → niveau 4 + escalation", () => {
    const sol = computeSolution(makeProfile({ clinicalProfile: "crisis", distressLevel: "critical" }));
    expect(sol.level).toBe(4);
    expect(sol.escalationRequired).toBe(true);
  });

  it("distressLevel critical (non-crisis) → niveau 3", () => {
    const sol = computeSolution(makeProfile({ clinicalProfile: "depression", distressLevel: "critical" }));
    expect(sol.level).toBe(3);
    expect(sol.escalationRequired).toBe(false);
  });

  it("burnout + tiredness → niveau 3", () => {
    const sol = computeSolution(makeProfile({
      emotionId: "tiredness",
      clinicalProfile: "burnout",
      distressLevel: "elevated",
      clinicalDimensions: ["burnout"],
    }));
    expect(sol.level).toBe(3);
  });

  it("distressLevel elevated → niveau 2 minimum", () => {
    const sol = computeSolution(makeProfile({ distressLevel: "elevated", clinicalProfile: "adjustment" }));
    expect(sol.level).toBe(2);
  });

  it("clinicalProfile adjustment + light → niveau 1", () => {
    const sol = computeSolution(makeProfile({ distressLevel: "light", clinicalProfile: "adjustment" }));
    expect(sol.level).toBe(1);
  });

  it("wellbeing → niveau 0", () => {
    const sol = computeSolution(makeProfile({
      emotionId: "joy",
      clinicalProfile: "wellbeing",
      distressLevel: "light",
      mlScore: 0.05,
      finalScore: 0.05,
    }));
    expect(sol.level).toBe(0);
    expect(sol.escalationRequired).toBe(false);
  });
});

// ─── Brique thérapeutique ────────────────────────────────────────────────────

describe("computeSolution — brique thérapeutique", () => {
  it("crisis → brique 'crisis'", () => {
    const sol = computeSolution(makeProfile({ clinicalProfile: "crisis", distressLevel: "critical" }));
    expect(sol.therapeuticBrick).toBe("crisis");
  });

  it("niveau 3 → brique 'professional'", () => {
    const sol = computeSolution(makeProfile({ clinicalProfile: "depression", distressLevel: "critical" }));
    expect(sol.therapeuticBrick).toBe("professional");
  });

  it("anxiety → brique 'mindfulness'", () => {
    const sol = computeSolution(makeProfile({ clinicalProfile: "anxiety", distressLevel: "light" }));
    expect(sol.therapeuticBrick).toBe("mindfulness");
  });

  it("burnout → brique 'psychoeducation'", () => {
    const sol = computeSolution(makeProfile({ clinicalProfile: "burnout", distressLevel: "light" }));
    expect(sol.therapeuticBrick).toBe("psychoeducation");
  });

  it("adjustment → brique 'cbt_activation'", () => {
    const sol = computeSolution(makeProfile({ clinicalProfile: "adjustment", distressLevel: "light" }));
    expect(sol.therapeuticBrick).toBe("cbt_activation");
  });
});

// ─── Ressources de sécurité ──────────────────────────────────────────────────

describe("computeSolution — ressources de sécurité", () => {
  it("niveau 4 adult — 3114 présent", () => {
    const sol = computeSolution(makeProfile({
      mode: "adult",
      clinicalProfile: "crisis",
      distressLevel: "critical",
    }));
    const has3114 = sol.resources.some((r) => r.id === "3114" || (r.href ?? "").includes("3114"));
    expect(has3114).toBe(true);
  });

  it("niveau 4 kids — 3114 présent", () => {
    const sol = computeSolution(makeProfile({
      mode: "kids",
      clinicalProfile: "crisis",
      distressLevel: "critical",
    }));
    const has3114 = sol.resources.some((r) => r.id === "3114" || (r.href ?? "").includes("3114"));
    expect(has3114).toBe(true);
  });

  it("niveau 3 adult — 3114 présent", () => {
    const sol = computeSolution(makeProfile({
      mode: "adult",
      clinicalProfile: "depression",
      distressLevel: "critical",
    }));
    const has3114 = sol.resources.some((r) => r.id === "3114" || (r.href ?? "").includes("3114"));
    expect(has3114).toBe(true);
  });

  it("niveau 0 — aucune ressource", () => {
    const sol = computeSolution(makeProfile({
      emotionId: "joy",
      clinicalProfile: "wellbeing",
      distressLevel: "light",
      mlScore: 0.05,
      finalScore: 0.05,
    }));
    expect(sol.resources).toHaveLength(0);
  });

  it("niveau 2 — au moins une ressource", () => {
    const sol = computeSolution(makeProfile({
      clinicalProfile: "depression",
      distressLevel: "elevated",
    }));
    expect(sol.resources.length).toBeGreaterThan(0);
  });
});

// ─── Ressources contextuelles ────────────────────────────────────────────────

describe("computeSolution — ressources contextuelles", () => {
  it("kids + anger + niveau >= 2 → Enfance en Danger (119)", () => {
    const sol = computeSolution(makeProfile({
      mode: "kids",
      emotionId: "anger",
      distressLevel: "elevated",
      clinicalProfile: "adjustment",
    }));
    const has119 = sol.resources.some((r) => r.href?.includes("119") || r.id === "enfanceEnDanger");
    expect(has119).toBe(true);
  });

  it("kids + stress + niveau >= 2 → harcèlement scolaire", () => {
    const sol = computeSolution(makeProfile({
      mode: "kids",
      emotionId: "stress",
      distressLevel: "elevated",
      clinicalProfile: "adjustment",
    }));
    const hasHarcelement = sol.resources.some(
      (r) => r.id === "antiHarcelement" || r.id === "harcelementScolaire" || (r.href ?? "").includes("3018") || (r.href ?? "").includes("3020")
    );
    expect(hasHarcelement).toBe(true);
  });
});

// ─── Messages et clôture ──────────────────────────────────────────────────────

describe("computeSolution — messages", () => {
  it("message non vide en mode kids", () => {
    const sol = computeSolution(makeProfile({ mode: "kids" }));
    expect(sol.message).toBeTruthy();
    expect(sol.message.length).toBeGreaterThan(0);
  });

  it("message non vide en mode adult", () => {
    const sol = computeSolution(makeProfile({ mode: "adult" }));
    expect(sol.message).toBeTruthy();
  });

  it("closing non vide", () => {
    const sol = computeSolution(makeProfile());
    expect(sol.closing).toBeTruthy();
    expect(sol.closing.length).toBeGreaterThan(0);
  });

  it("kids et adult ont des messages différents au niveau 4", () => {
    const kidsMsg = computeSolution(makeProfile({ mode: "kids", clinicalProfile: "crisis", distressLevel: "critical" })).message;
    const adultMsg = computeSolution(makeProfile({ mode: "adult", clinicalProfile: "crisis", distressLevel: "critical" })).message;
    expect(kidsMsg).not.toBe(adultMsg);
  });
});

// ─── Micro-actions ────────────────────────────────────────────────────────────

describe("computeSolution — microActions", () => {
  it("retourne au moins une action pour tous les niveaux 0-3", () => {
    const profiles: Array<Partial<DiagnosticProfile>> = [
      { clinicalProfile: "wellbeing", distressLevel: "light", finalScore: 0.1 },
      { clinicalProfile: "adjustment", distressLevel: "light" },
      { clinicalProfile: "depression", distressLevel: "elevated" },
      { clinicalProfile: "depression", distressLevel: "critical" },
    ];
    for (const p of profiles) {
      const sol = computeSolution(makeProfile(p));
      expect(sol.microActions.length).toBeGreaterThan(0);
    }
  });

  it("chaque action a un titre et une description", () => {
    const sol = computeSolution(makeProfile());
    for (const action of sol.microActions) {
      expect(action.title).toBeTruthy();
      expect(action.description).toBeTruthy();
    }
  });
});
