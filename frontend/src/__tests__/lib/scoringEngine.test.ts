import { describe, it, expect } from "vitest";
import {
  sanitizeMlScore,
  computeFinalScore,
  getDistressLevel,
  detectClinicalDimensions,
  deriveClinicalProfile,
  buildDiagnosticProfile,
  SCORE_CRITICAL,
  SCORE_ELEVATED,
  EMOTION_FLOOR,
  CRITICAL_KEYWORDS,
  DISTRESS_TEXT_SIGNALS,
} from "../../lib/scoringEngine";

// ─── sanitizeMlScore ─────────────────────────────────────────────────────────

describe("sanitizeMlScore", () => {
  it("retourne null pour une valeur non numérique", () => {
    expect(sanitizeMlScore("0.5")).toBeNull();
    expect(sanitizeMlScore(null)).toBeNull();
    expect(sanitizeMlScore(undefined)).toBeNull();
    expect(sanitizeMlScore({})).toBeNull();
  });

  it("retourne null pour NaN et Infinity", () => {
    expect(sanitizeMlScore(NaN)).toBeNull();
    expect(sanitizeMlScore(Infinity)).toBeNull();
    expect(sanitizeMlScore(-Infinity)).toBeNull();
  });

  it("clamp entre 0 et 1", () => {
    expect(sanitizeMlScore(1.5)).toBe(1);
    expect(sanitizeMlScore(-0.2)).toBe(0);
    expect(sanitizeMlScore(0)).toBe(0);
    expect(sanitizeMlScore(1)).toBe(1);
  });

  it("conserve une valeur valide dans [0, 1]", () => {
    expect(sanitizeMlScore(0.72)).toBe(0.72);
    expect(sanitizeMlScore(0.35)).toBe(0.35);
  });
});

// ─── computeFinalScore ───────────────────────────────────────────────────────

describe("computeFinalScore", () => {
  it("retourne le score ML sans self-report", () => {
    const score = computeFinalScore(0.5, "stress", null);
    expect(score).toBeGreaterThanOrEqual(0.25); // plancher stress
    expect(score).toBeLessThanOrEqual(1.0);
  });

  it("applique le plancher émotionnel sadness (0.35)", () => {
    const score = computeFinalScore(0.1, "sadness", null);
    expect(score).toBe(EMOTION_FLOOR.sadness);
  });

  it("applique le plancher émotionnel fear (0.35)", () => {
    const score = computeFinalScore(0.05, "fear", null);
    expect(score).toBe(EMOTION_FLOOR.fear);
  });

  it("aucun plancher pour joy/calm/pride avec score très bas (≤ 0.25)", () => {
    // Score ≤ 0.25 : pas de masquage → résultat = mlScore
    expect(computeFinalScore(0.05, "joy", null)).toBe(0.05);
    expect(computeFinalScore(0.0, "calm", null)).toBe(0.0);
    expect(computeFinalScore(0.1, "pride", null)).toBe(0.1);
  });

  it("blend pondéré avec self-report", () => {
    const ml = 0.6;
    const self = 0.8;
    const expected = self * 0.45 + (ml + 0.20) * 0.55; // masking car joy → +0.20
    // Utilisons stress (pas d'émotion positive) pour tester le blend pur
    const expectedStress = self * 0.45 + ml * 0.55; // 0.36 + 0.33 = 0.69
    expect(computeFinalScore(ml, "stress", self)).toBeCloseTo(expectedStress, 5);
  });

  // Fix 2 — seuil masquage abaissé à > 0.25, bonus porté à +0.20
  it("Fix 2 — masking : joy + ML > 0.25 ajoute +0.20", () => {
    // joy floor=0.0 < 0.2, mlScore=0.3 > 0.25 → isMasking = true
    expect(computeFinalScore(0.3, "joy", null)).toBeCloseTo(0.5, 5); // 0.3 + 0.20
  });

  it("Fix 2 — masking : joy + ML = 0.6 → 0.80", () => {
    expect(computeFinalScore(0.6, "joy", null)).toBeCloseTo(0.80, 5); // 0.6 + 0.20
  });

  it("Fix 2 — pas de masking pour joy avec ML ≤ 0.25", () => {
    // Score authentiquement bas → pas de bonus
    expect(computeFinalScore(0.1, "joy", null)).toBe(0.1);
    expect(computeFinalScore(0.25, "joy", null)).toBe(0.25); // exactement 0.25 → pas > 0.25
  });

  it("Fix 2 — pas de masking pour sadness (floor=0.35 ≥ 0.2)", () => {
    const scoreStressTexte = computeFinalScore(0.6, "sadness", null);
    const scoreJoieTexte = computeFinalScore(0.6, "joy", null);
    expect(scoreStressTexte).not.toBeCloseTo(scoreJoieTexte, 5);
    expect(scoreStressTexte).toBe(0.6);
  });

  it("ne dépasse jamais 1.0", () => {
    expect(computeFinalScore(1.0, "joy", 1.0)).toBe(1.0);
    expect(computeFinalScore(0.95, "joy", null)).toBe(1.0); // 0.95 + 0.20 → clamp
  });

  it("règle du maximum : mlAdjusted ne peut pas abaisser le score final", () => {
    // Stress floor=0.25, mlScore=0.4 → final ≥ 0.4
    expect(computeFinalScore(0.4, "stress", null)).toBeGreaterThanOrEqual(0.4);
  });
});

// ─── detectClinicalDimensions ────────────────────────────────────────────────

describe("detectClinicalDimensions", () => {
  it("retourne tableau vide pour texte neutre", () => {
    expect(detectClinicalDimensions("Je me sens bien aujourd'hui")).toEqual([]);
  });

  it("détecte burnout", () => {
    expect(detectClinicalDimensions("je n'arrive plus, je suis à bout")).toContain("burnout");
  });

  it("détecte anxiety", () => {
    expect(detectClinicalDimensions("j'angoisse tout le temps et je panique")).toContain("anxiety");
  });

  it("détecte depression_masked", () => {
    expect(detectClinicalDimensions("à quoi ça sert, je me sens vide")).toContain("depression_masked");
  });

  it("détecte dysregulation", () => {
    expect(detectClinicalDimensions("j'explose et je perds le contrôle")).toContain("dysregulation");
  });

  it("détecte plusieurs dimensions simultanément", () => {
    const dims = detectClinicalDimensions("j'angoisse et à quoi ça sert de continuer");
    expect(dims).toContain("anxiety");
    expect(dims).toContain("depression_masked");
  });

  it("détection insensible à la casse", () => {
    expect(detectClinicalDimensions("BURNOUT : plus la force")).toContain("burnout");
  });

  it("détecte mots-clés anglais (burnout)", () => {
    expect(detectClinicalDimensions("I'm burned out and exhausted for weeks")).toContain("burnout");
  });

  // Nouveaux mots-clés cliniques (rec. clinicien)
  it("burnout — cynisme/désengagement détecté", () => {
    expect(detectClinicalDimensions("je m'en fiche de tout, plus de sens au travail")).toContain("burnout");
    expect(detectClinicalDimensions("i don't care anymore about anything")).toContain("burnout");
  });

  it("burnout — inefficacité détectée", () => {
    expect(detectClinicalDimensions("je suis complètement dépassé et submergé")).toContain("burnout");
    expect(detectClinicalDimensions("overwhelmed by everything")).toContain("burnout");
  });

  it("anxiety — anticipation catastrophiste détectée", () => {
    expect(detectClinicalDimensions("je m'inquiète tout le temps, je pense au pire")).toContain("anxiety");
    expect(detectClinicalDimensions("always on edge and can't relax")).toContain("anxiety");
  });

  it("anxiety — hypervigilance/somatique détectée", () => {
    expect(detectClinicalDimensions("je suis tendu, je n'arrive pas à me détendre")).toContain("anxiety");
    expect(detectClinicalDimensions("je tremble et j'ai une boule au ventre")).toContain("anxiety");
    expect(detectClinicalDimensions("je n'arrive pas à dormir tellement j'angoisse")).toContain("anxiety");
  });

  it("depression_masked — anhédonie détectée", () => {
    expect(detectClinicalDimensions("plus envie de rien, rien ne me fait plaisir")).toContain("depression_masked");
  });

  it("depression_masked — fatigue morale/ralentissement détecté", () => {
    expect(detectClinicalDimensions("tout me coûte, je suis épuisé moralement")).toContain("depression_masked");
    expect(detectClinicalDimensions("je n'avance pas, je suis ralenti dans tout")).toContain("depression_masked");
  });

  it("depression_masked — isolement/inutilité détecté", () => {
    expect(detectClinicalDimensions("je me sens seul et je ne sers à rien")).toContain("depression_masked");
    expect(detectClinicalDimensions("personne ne comprend, je suis invisible")).toContain("depression_masked");
    expect(detectClinicalDimensions("i feel empty and i feel alone")).toContain("depression_masked");
  });

  it("dysregulation — impulsivité/fuite détectée", () => {
    expect(detectClinicalDimensions("j'ai envie de tout lâcher et je veux tout casser")).toContain("dysregulation");
  });

  it("CRITICAL_KEYWORDS contient au moins 25 entrées (directes + indirectes)", () => {
    expect(CRITICAL_KEYWORDS.length).toBeGreaterThanOrEqual(25);
  });
});

// ─── getDistressLevel ────────────────────────────────────────────────────────

describe("getDistressLevel", () => {
  it("retourne critical sur keyword critique — indépendamment du score", () => {
    expect(getDistressLevel(0.1, "je veux mourir", "joy", [], null)).toBe("critical");
    expect(getDistressLevel(null, "pensées suicidaires", "calm", [], null)).toBe("critical");
    expect(getDistressLevel(0.0, "je n'en peux plus", "stress", [], null)).toBe("critical");
  });

  it("retourne elevated sur dysregulation quelle que soit l'intensité", () => {
    expect(getDistressLevel(0.1, "texte normal", "joy", ["dysregulation"], null)).toBe("elevated");
  });

  it("retourne critical quand score final ≥ SCORE_CRITICAL", () => {
    expect(getDistressLevel(0.9, "texte normal", "stress", [], null)).toBe("critical");
  });

  it("retourne elevated quand score final ≥ SCORE_ELEVATED", () => {
    expect(getDistressLevel(0.4, "texte normal", "stress", [], null)).toBe("elevated");
  });

  it("retourne light pour score bas authentiquement positif", () => {
    // mlScore=0.1 ≤ 0.25 → pas de masking → score=0.1 < 0.35, texte positif
    expect(getDistressLevel(0.1, "je suis content", "joy", [], null)).toBe("light");
  });

  // Fix 3 — dimensions vérifiées avant le null-guard ML
  it("Fix 3 — retourne elevated si dimensions présentes (avec ou sans ML)", () => {
    expect(getDistressLevel(0.1, "je n'arrive plus", "anger", ["burnout"], null)).toBe("elevated");
    expect(getDistressLevel(null, "je n'arrive plus", "joy", ["burnout"], null)).toBe("elevated");
  });

  it("Fix 3 — joy + texte anxieux (dimensions) → elevated même sans ML", () => {
    expect(getDistressLevel(null, "j'angoisse tout le temps", "joy", ["anxiety"], null)).toBe("elevated");
  });

  // Fix 1 — fallback utilise selfScore
  it("Fix 1 — fallback sans ML : sadness → elevated via plancher", () => {
    expect(getDistressLevel(null, "je suis triste", "sadness", [], null)).toBe("elevated");
  });

  it("Fix 1 — fallback sans ML : joy + selfScore modéré → elevated", () => {
    // selfScore=0.5 ∈ [0.35, 0.65[ → elevated
    expect(getDistressLevel(null, "je suis content", "joy", [], 0.5)).toBe("elevated");
  });

  it("Fix 1 — fallback sans ML : joy + selfScore critique → critical", () => {
    // selfScore=0.8 ≥ SCORE_CRITICAL=0.65 → critical
    expect(getDistressLevel(null, "je suis content", "joy", [], 0.8)).toBe("critical");
  });

  it("Fix 1 — fallback sans ML : joy → light si selfScore bas", () => {
    expect(getDistressLevel(null, "je suis content", "joy", [], null)).toBe("light");
    expect(getDistressLevel(null, "je suis content", "joy", [], 0.1)).toBe("light");
  });

  // Fix 1 + 3 — DISTRESS_TEXT_SIGNALS dans le fallback
  it("Fix 1+3 — joy + texte négatif explicite sans ML → elevated (incohérence émotion/texte)", () => {
    expect(getDistressLevel(null, "je me sens mal", "joy", [], null)).toBe("elevated");
    expect(getDistressLevel(null, "ça ne va pas du tout", "calm", [], null)).toBe("elevated");
    expect(getDistressLevel(null, "j'ai besoin d'aide", "pride", [], null)).toBe("elevated");
  });

  it("Fix 1+3 — texte positif avec émotion positive sans ML → light (pas de faux positif)", () => {
    expect(getDistressLevel(null, "je suis vraiment content", "joy", [], null)).toBe("light");
    expect(getDistressLevel(null, "tout va bien", "calm", [], null)).toBe("light");
  });

  it("self-report élevé avec ML augmente le niveau", () => {
    const level = getDistressLevel(0.3, "stressé", "stress", [], 0.9);
    expect(level).toBe("elevated");
  });

  it("tous les keywords critiques déclenchent critical (idéation directe + indirecte)", () => {
    for (const kw of CRITICAL_KEYWORDS) {
      const level = getDistressLevel(0, `context: ${kw}`, "joy", [], null);
      expect(level).toBe("critical");
    }
  });

  it("idéation indirecte — fardeau → critical", () => {
    expect(getDistressLevel(null, "je suis un fardeau pour tout le monde", "joy", [], null)).toBe("critical");
    expect(getDistressLevel(null, "ça serait mieux sans moi", "calm", [], null)).toBe("critical");
    expect(getDistressLevel(null, "plus de raison de vivre", "stress", [], null)).toBe("critical");
  });

  it("idéation indirecte EN → critical", () => {
    expect(getDistressLevel(null, "i feel better off without me", "joy", [], null)).toBe("critical");
    expect(getDistressLevel(null, "no reason to live anymore", "calm", [], null)).toBe("critical");
  });

  // Fix 2 — masquage émotion/texte avec ML modéré
  it("Fix 2 — joy + ML modéré (0.3) → elevated via masquage", () => {
    // mlScore=0.3 > 0.25 → masking → mlAdjusted=0.50 ≥ 0.35 → elevated
    expect(getDistressLevel(0.3, "je vais bien", "joy", [], null)).toBe("elevated");
  });
});

// ─── deriveClinicalProfile ───────────────────────────────────────────────────

describe("deriveClinicalProfile", () => {
  it("critical → crisis toujours", () => {
    expect(deriveClinicalProfile("critical", "joy", [])).toBe("crisis");
    expect(deriveClinicalProfile("critical", "sadness", ["anxiety"])).toBe("crisis");
  });

  it("dysregulation → crisis même en light", () => {
    expect(deriveClinicalProfile("light", "anger", ["dysregulation"])).toBe("crisis");
  });

  it("burnout dimensions → burnout", () => {
    expect(deriveClinicalProfile("elevated", "stress", ["burnout"])).toBe("burnout");
  });

  it("tiredness + depression_masked → burnout", () => {
    expect(deriveClinicalProfile("elevated", "tiredness", ["depression_masked"])).toBe("burnout");
  });

  it("anxiety dimension ou émotion fear → anxiety", () => {
    expect(deriveClinicalProfile("elevated", "stress", ["anxiety"])).toBe("anxiety");
    expect(deriveClinicalProfile("elevated", "fear", [])).toBe("anxiety");
  });

  it("depression_masked ou émotion sadness → depression", () => {
    expect(deriveClinicalProfile("elevated", "anger", ["depression_masked"])).toBe("depression");
    expect(deriveClinicalProfile("elevated", "sadness", [])).toBe("depression");
  });

  it("elevated sans dimension → adjustment", () => {
    expect(deriveClinicalProfile("elevated", "stress", [])).toBe("adjustment");
  });

  it("light sans dimension → wellbeing", () => {
    expect(deriveClinicalProfile("light", "joy", [])).toBe("wellbeing");
    expect(deriveClinicalProfile("light", "calm", [])).toBe("wellbeing");
  });
});

// ─── buildDiagnosticProfile (pipeline complet) ──────────────────────────────

describe("buildDiagnosticProfile", () => {
  it("construit un profil complet pour une entrée normale", () => {
    const profile = buildDiagnosticProfile({
      emotionId: "sadness",
      mode: "adult",
      userText: "je suis un peu triste sans raison",
      mlScore: 0.4,
      selfScore: 0.5,
      selfReportAnswers: [1, 1, 2],
    });

    expect(profile.emotionId).toBe("sadness");
    expect(profile.mode).toBe("adult");
    expect(profile.distressLevel).toBe("elevated");
    expect(profile.clinicalProfile).toBe("depression");
    expect(profile.finalScore).not.toBeNull();
    expect(profile.selfScore).toBe(0.5);
    expect(profile.selfReportAnswers).toEqual([1, 1, 2]);
  });

  it("détecte la crise sur keyword", () => {
    const profile = buildDiagnosticProfile({
      emotionId: "sadness",
      mode: "kids",
      userText: "je veux mourir tellement je suis triste",
      mlScore: 0.2,
      selfScore: null,
      selfReportAnswers: null,
    });

    expect(profile.distressLevel).toBe("critical");
    expect(profile.clinicalProfile).toBe("crisis");
  });

  it("gère mlScore null (API indisponible) — émotion négative → elevated", () => {
    const profile = buildDiagnosticProfile({
      emotionId: "fear",
      mode: "adult",
      userText: "j'ai un peu peur",
      mlScore: null,
      selfScore: null,
      selfReportAnswers: null,
    });

    expect(profile.finalScore).toBeNull();
    expect(profile.distressLevel).toBe("elevated"); // plancher fear=0.35
  });

  it("produit wellbeing pour une émotion positive avec faible score", () => {
    const profile = buildDiagnosticProfile({
      emotionId: "joy",
      mode: "adult",
      userText: "je suis vraiment content aujourd'hui",
      mlScore: 0.05,
      selfScore: null,
      selfReportAnswers: null,
    });

    expect(profile.distressLevel).toBe("light");
    expect(profile.clinicalProfile).toBe("wellbeing");
  });

  // Fix 1+3 — cas critique : incohérence émotion/texte sans ML
  it("Fix 1+3 — joy + texte négatif sans ML → elevated (biais corrigé)", () => {
    const profile = buildDiagnosticProfile({
      emotionId: "joy",
      mode: "adult",
      userText: "je me sens mal, ça ne va pas",
      mlScore: null,
      selfScore: null,
      selfReportAnswers: null,
    });

    expect(profile.distressLevel).toBe("elevated");
    expect(profile.clinicalProfile).toBe("adjustment");
  });

  // Fix 2 — masquage avec ML modéré
  it("Fix 2 — joy + ML modéré → elevated via masquage émotion/texte", () => {
    const profile = buildDiagnosticProfile({
      emotionId: "joy",
      mode: "adult",
      userText: "bof",
      mlScore: 0.3,
      selfScore: null,
      selfReportAnswers: null,
    });

    expect(profile.distressLevel).toBe("elevated");
  });

  // Fix 1 — selfScore dans le fallback
  it("Fix 1 — joy + selfScore modéré sans ML → elevated", () => {
    // selfScore=0.5 ∈ [0.35, 0.65[ → elevated
    const profile = buildDiagnosticProfile({
      emotionId: "joy",
      mode: "adult",
      userText: "tout va bien en fait",
      mlScore: null,
      selfScore: 0.5,
      selfReportAnswers: [1, 2, 1],
    });

    expect(profile.distressLevel).toBe("elevated");
  });
});

// ─── Constantes exportées ───────────────────────────────────────────────────

describe("constantes", () => {
  it("SCORE_CRITICAL = 0.65", () => expect(SCORE_CRITICAL).toBe(0.65));
  it("SCORE_ELEVATED = 0.35", () => expect(SCORE_ELEVATED).toBe(0.35));
  it("EMOTION_FLOOR sadness = 0.35", () => expect(EMOTION_FLOOR.sadness).toBe(0.35));
  it("CRITICAL_KEYWORDS contient au moins 25 entrées (directes + indirectes)", () => {
    expect(CRITICAL_KEYWORDS.length).toBeGreaterThanOrEqual(25);
  });
  it("DISTRESS_TEXT_SIGNALS contient au moins 10 signaux", () => {
    expect(DISTRESS_TEXT_SIGNALS.length).toBeGreaterThanOrEqual(10);
  });
});
