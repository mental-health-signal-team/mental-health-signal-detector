/**
 * Tests unitaires — scoringEngine.ts
 *
 * Couverture :
 *   - sanitizeMlScore
 *   - computeSelfScore (DSM-5 weighting)
 *   - detectClinicalDimensions (keyword matching)
 *   - detectDimensionsFromSelfReport (per-emotion logic)
 *   - computeFinalScore (masking, floor, blending)
 *   - getDistressLevel (safety net, ML path, fallback)
 *   - deriveClinicalProfile
 */

import { describe, it, expect } from "vitest";
import {
  sanitizeMlScore,
  computeSelfScore,
  detectClinicalDimensions,
  detectDimensionsFromSelfReport,
  computeFinalScore,
  getDistressLevel,
  deriveClinicalProfile,
  EMOTION_FLOOR,
  SCORE_CRITICAL,
  SCORE_ELEVATED,
} from "./scoringEngine";

// ─── sanitizeMlScore ─────────────────────────────────────────────────────────

describe("sanitizeMlScore", () => {
  it("retourne null pour undefined", () => expect(sanitizeMlScore(undefined)).toBeNull());
  it("retourne null pour null", () => expect(sanitizeMlScore(null)).toBeNull());
  it("retourne null pour une chaîne", () => expect(sanitizeMlScore("0.5")).toBeNull());
  it("retourne null pour NaN", () => expect(sanitizeMlScore(NaN)).toBeNull());
  it("retourne null pour Infinity", () => expect(sanitizeMlScore(Infinity)).toBeNull());
  it("clamp à 0 si négatif", () => expect(sanitizeMlScore(-0.5)).toBe(0));
  it("clamp à 1 si > 1", () => expect(sanitizeMlScore(1.5)).toBe(1));
  it("retourne la valeur telle quelle dans [0,1]", () => expect(sanitizeMlScore(0.42)).toBe(0.42));
  it("accepte 0", () => expect(sanitizeMlScore(0)).toBe(0));
  it("accepte 1", () => expect(sanitizeMlScore(1)).toBe(1));
});

// ─── computeSelfScore ────────────────────────────────────────────────────────

describe("computeSelfScore", () => {
  it("retourne 0 pour tableau vide", () => expect(computeSelfScore([])).toBe(0));

  it("1 réponse → intensité / 3", () => {
    expect(computeSelfScore([3])).toBeCloseTo(1.0);
    expect(computeSelfScore([0])).toBeCloseTo(0.0);
  });

  it("3 réponses minimales → score 0", () => {
    expect(computeSelfScore([0, 0, 0])).toBe(0);
  });

  it("3 réponses maximales → score 1", () => {
    expect(computeSelfScore([3, 3, 3])).toBeCloseTo(1.0);
  });

  it("durée et impact pondérés 1.5× (DSM-5)", () => {
    // answers[0]=intensité×1.0, answers[1]=durée×1.5, answers[2]=impact×1.5
    // [0, 3, 0] → (0×1.0 + 3×1.5 + 0×1.5) / (3×1.0 + 3×1.5 + 3×1.5) = 4.5 / 12 = 0.375
    expect(computeSelfScore([0, 3, 0])).toBeCloseTo(0.375);
  });

  it("intensity seule ne suffit pas pour score élevé", () => {
    // [3, 0, 0] → (3×1.0 + 0 + 0) / 12 = 3/12 = 0.25
    expect(computeSelfScore([3, 0, 0])).toBeCloseTo(0.25);
  });

  it("ne dépasse jamais 1.0", () => {
    expect(computeSelfScore([3, 3, 3, 3])).toBeLessThanOrEqual(1.0);
  });
});

// ─── detectClinicalDimensions ────────────────────────────────────────────────

describe("detectClinicalDimensions", () => {
  it("retourne [] pour un texte neutre", () => {
    expect(detectClinicalDimensions("je suis bien aujourd'hui")).toEqual([]);
  });

  it("détecte burnout sur 'épuisé depuis'", () => {
    const dims = detectClinicalDimensions("je suis épuisé depuis des semaines");
    expect(dims).toContain("burnout");
  });

  it("détecte anxiety sur 'attaque de panique'", () => {
    const dims = detectClinicalDimensions("j'ai eu une attaque de panique");
    expect(dims).toContain("anxiety");
  });

  it("détecte depression_masked sur 'plus de plaisir'", () => {
    const dims = detectClinicalDimensions("je n'ai plus de plaisir à rien");
    expect(dims).toContain("depression_masked");
  });

  it("détecte dysregulation sur 'je perds le contrôle'", () => {
    const dims = detectClinicalDimensions("je perds le contrôle et j'explose");
    expect(dims).toContain("dysregulation");
  });

  it("est insensible à la casse", () => {
    const dims = detectClinicalDimensions("J'EN PEUX PLUS");
    expect(dims).toContain("burnout");
  });

  it("peut retourner plusieurs dimensions simultanément", () => {
    const dims = detectClinicalDimensions("j'angoisse et je me sens vide");
    expect(dims).toContain("anxiety");
    expect(dims).toContain("depression_masked");
  });

  it("détecte les signaux en anglais", () => {
    expect(detectClinicalDimensions("i feel empty")).toContain("depression_masked");
    expect(detectClinicalDimensions("burned out")).toContain("burnout");
    expect(detectClinicalDimensions("panic attack")).toContain("anxiety");
  });
});

// ─── detectDimensionsFromSelfReport ──────────────────────────────────────────

describe("detectDimensionsFromSelfReport", () => {
  it("retourne [] si moins de 2 réponses", () => {
    expect(detectDimensionsFromSelfReport([], "stress")).toEqual([]);
    expect(detectDimensionsFromSelfReport([2], "stress")).toEqual([]);
  });

  describe("stress", () => {
    it("stress persistant → burnout", () => {
      // duration=2 (>1 sem) → isPersistent
      const dims = detectDimensionsFromSelfReport([1, 2, 0], "stress");
      expect(dims).toContain("burnout");
    });

    it("stress persistant + intense → burnout + anxiety", () => {
      const dims = detectDimensionsFromSelfReport([2, 2, 0], "stress");
      expect(dims).toContain("burnout");
      expect(dims).toContain("anxiety");
    });

    it("stress court et léger → aucune dimension", () => {
      const dims = detectDimensionsFromSelfReport([1, 0, 0], "stress");
      expect(dims).toEqual([]);
    });
  });

  describe("tiredness", () => {
    it("fatigue persistante → burnout", () => {
      const dims = detectDimensionsFromSelfReport([1, 2, 0], "tiredness");
      expect(dims).toContain("burnout");
    });

    it("fatigue longue durée + énergie nulle → burnout + depression_masked", () => {
      const dims = detectDimensionsFromSelfReport([1, 3, 2], "tiredness");
      expect(dims).toContain("burnout");
      expect(dims).toContain("depression_masked");
    });
  });

  describe("sadness", () => {
    it("tristesse persistante → depression_masked", () => {
      const dims = detectDimensionsFromSelfReport([1, 2, 0], "sadness");
      expect(dims).toContain("depression_masked");
    });

    it("tristesse avec impact → depression_masked", () => {
      const dims = detectDimensionsFromSelfReport([1, 0, 2], "sadness");
      expect(dims).toContain("depression_masked");
    });

    it("tristesse courte sans impact → aucune dimension", () => {
      const dims = detectDimensionsFromSelfReport([1, 0, 1], "sadness");
      expect(dims).toEqual([]);
    });
  });

  describe("fear", () => {
    it("peur persistante → anxiety", () => {
      const dims = detectDimensionsFromSelfReport([1, 2, 0], "fear");
      expect(dims).toContain("anxiety");
    });

    it("peur avec impact quotidien → anxiety", () => {
      const dims = detectDimensionsFromSelfReport([1, 0, 2], "fear");
      expect(dims).toContain("anxiety");
    });
  });

  describe("anger", () => {
    it("colère intense + persistante → dysregulation", () => {
      const dims = detectDimensionsFromSelfReport([2, 2, 0], "anger");
      expect(dims).toContain("dysregulation");
    });

    it("colère légère et courte → aucune dimension", () => {
      const dims = detectDimensionsFromSelfReport([1, 1, 0], "anger");
      expect(dims).toEqual([]);
    });
  });
});

// ─── computeFinalScore ───────────────────────────────────────────────────────

describe("computeFinalScore", () => {
  it("plancher émotionnel appliqué (sadness floor 0.35)", () => {
    // mlScore 0.1 < floor 0.35 → score = 0.35
    const score = computeFinalScore(0.1, "sadness", null);
    expect(score).toBeGreaterThanOrEqual(EMOTION_FLOOR["sadness"]);
  });

  it("émotion positive sans masquage", () => {
    // joy, mlScore 0.1 → pas de masquage (mlScore ≤ 0.25 threshold)
    const score = computeFinalScore(0.1, "joy", null);
    expect(score).toBeCloseTo(0.1);
  });

  it("masquage détecté (joy + mlScore > 0.25) → bonus +0.20", () => {
    const score = computeFinalScore(0.4, "joy", null);
    // mlAdjusted = min(1, 0.4 + 0.20) = 0.60
    expect(score).toBeCloseTo(0.60);
  });

  it("selfScore fusionne avec mlScore (55/45)", () => {
    // selfScore=0.8, mlScore=0.2, émotion neutre (calm, floor=0), pas de masquage
    const score = computeFinalScore(0.2, "calm", 0.8);
    // blended = 0.8×0.45 + 0.2×0.55 = 0.36 + 0.11 = 0.47
    // max(0.47, 0, 0.2) = 0.47
    expect(score).toBeCloseTo(0.47);
  });

  it("ne dépasse jamais 1.0", () => {
    expect(computeFinalScore(1.0, "sadness", 1.0)).toBeLessThanOrEqual(1.0);
  });

  it("score jamais inférieur au mlScore ajusté (règle du maximum)", () => {
    const ml = 0.5;
    const score = computeFinalScore(ml, "calm", 0.1);
    expect(score).toBeGreaterThanOrEqual(ml);
  });
});

// ─── getDistressLevel ────────────────────────────────────────────────────────

describe("getDistressLevel", () => {
  it("keyword critique → critical immédiat", () => {
    expect(getDistressLevel(0.1, "je veux mourir", "joy", [], null)).toBe("critical");
  });

  it("keyword critique ignore le score ML faible", () => {
    expect(getDistressLevel(0.05, "j'ai pensées suicidaires", "calm", [], null)).toBe("critical");
  });

  it("dysregulation dans les dimensions → elevated minimum", () => {
    expect(getDistressLevel(0.1, "texte neutre", "anger", ["dysregulation"], null)).toBe("elevated");
  });

  it("toute dimension → elevated minimum", () => {
    expect(getDistressLevel(0.05, "texte neutre", "joy", ["burnout"], null)).toBe("elevated");
  });

  it("finalScore >= SCORE_CRITICAL → critical", () => {
    // mlScore=0.8 > SCORE_CRITICAL=0.65, émotion sadness
    expect(getDistressLevel(0.8, "texte", "sadness", [], null)).toBe("critical");
  });

  it("finalScore entre SCORE_ELEVATED et SCORE_CRITICAL → elevated", () => {
    expect(getDistressLevel(0.5, "texte", "sadness", [], null)).toBe("elevated");
  });

  it("finalScore < SCORE_ELEVATED → light", () => {
    // joy, mlScore=0.1, pas de masquage → score~0.1 < 0.35
    expect(getDistressLevel(0.1, "texte neutre", "joy", [], null)).toBe("light");
  });

  describe("fallback sans ML", () => {
    it("plancher sadness (0.35) → elevated", () => {
      expect(getDistressLevel(null, "texte neutre", "sadness", [], null)).toBe("elevated");
    });

    it("plancher joy (0.0) + selfScore faible → light", () => {
      expect(getDistressLevel(null, "texte neutre", "joy", [], 0.1)).toBe("light");
    });

    it("signaux texte détresse sans ML → elevated", () => {
      expect(getDistressLevel(null, "je me sens mal", "joy", [], null)).toBe("elevated");
    });
  });
});

// ─── deriveClinicalProfile ───────────────────────────────────────────────────

describe("deriveClinicalProfile", () => {
  it("distressLevel critical → crisis", () => {
    expect(deriveClinicalProfile("critical", "sadness", [])).toBe("crisis");
  });

  it("dysregulation → crisis", () => {
    expect(deriveClinicalProfile("elevated", "anger", ["dysregulation"])).toBe("crisis");
  });

  it("burnout + tiredness → burnout", () => {
    expect(deriveClinicalProfile("elevated", "tiredness", ["burnout"])).toBe("burnout");
  });

  it("anxiety ou fear → anxiety", () => {
    expect(deriveClinicalProfile("elevated", "fear", [])).toBe("anxiety");
    expect(deriveClinicalProfile("light", "joy", ["anxiety"])).toBe("anxiety");
  });

  it("depression_masked ou sadness → depression", () => {
    expect(deriveClinicalProfile("elevated", "sadness", [])).toBe("depression");
    expect(deriveClinicalProfile("light", "joy", ["depression_masked"])).toBe("depression");
  });

  it("elevated sans dimension spécifique → adjustment", () => {
    expect(deriveClinicalProfile("elevated", "stress", [])).toBe("adjustment");
  });

  it("light sans dimension → wellbeing", () => {
    expect(deriveClinicalProfile("light", "joy", [])).toBe("wellbeing");
  });
});
