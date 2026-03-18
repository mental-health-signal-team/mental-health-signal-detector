# Changelog

All notable changes to **Mental Health Signal Detector** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [0.3.0] - 2026-03-18

### Added
- Support du modèle `mental_bert_v3` dans l'API (endpoint `/analyze` + schéma Pydantic)
- Notebooks de fine-tuning v3 avec rapport de comparaison des modèles (export PNG)
- 28 nouveaux tests de sécurité (117 tests au total)

### Fixed
- `model_path_v3` corrigé pour pointer vers le sous-dossier réel `mental_bert_v3/`
- Export du tableau de comparaison des modèles corrigé pour l'environnement notebook VS Code
- Dépendance `anthropic>=0.40.0` ajoutée dans `requirements.slim.txt` (déploiement)
- Build TypeScript Vercel : exclusion des fichiers de test

### Security
- **B1** : rate limit ajouté sur `/checkin/reminder` (10 req/min)
- **B2** : client Anthropic en singleton (évite la création d'un pool par requête)
- **B3** : `_get_client_ip()` lit correctement l'en-tête `X-Forwarded-For`
- **B4** : `_MODELS_DIR` utilise `__file__` (indépendant du répertoire de travail)
- **B5** : garde-fou dans `run_explain` pour les textes hors vocabulaire
- **B6** : accès direct à `_DIM_LABELS` (suppression du fallback inutile)
- **S1** : `emotion_id` et `distress_level` non persistés en mémoire (RGPD Art. 9)
- **S2** : vérification de l'environnement pour la configuration CORS en production
- **S4** : `_build_user_prompt` reçoit des scalaires explicites (prévention injection `userText`)
- **S6** : `logger.exception()` remplace `logger.error(f"...{e}")` pour la capture de stack trace

### Changed
- Image Docker optimisée : modèles lourds exclus du build context via `.dockerignore`
- PyTorch remplacé par la variante CPU-only dans l'image Docker

---

## [0.2.0] - 2026-03-18

### Added
- Endpoint `POST /analyze` avec messages LLM personnalisés via l'API Claude (Anthropic)
- Accessibilité complète ARIA — passe Priority 2F (tous les composants React)
- Feedback utilisateur sur les micro-actions (Priorité 2E)
- Transition narrative QuickCheck avant les questions (Priorité 2D)
- Tests E2E Playwright : happy path + flow de crise
- Tests unitaires Vitest : `scoringEngine` et `solutionEngine`

---

## [0.1.0] - 2026-03-18

### Added
- Pipeline NLP Phase 1 : prétraitement, extraction de features, classification
- Modèle baseline (90,9 % d'accuracy) et modèle DistilBERT fine-tuné (96,8 % d'accuracy)
- API FastAPI avec endpoints de détection de signaux de santé mentale
- Application web React (Phase 2) — 5 écrans : QuickCheck, résultats, ressources, historique, profil
- Intégration ML côté front : moteur de scoring fusionnant score ML, émotion et masking
- CI GitHub Actions opérationnelle (tests verts, PR #2)

---

[Unreleased]: https://github.com/fmoncaut/mental-health-signal-detector/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/fmoncaut/mental-health-signal-detector/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/fmoncaut/mental-health-signal-detector/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/fmoncaut/mental-health-signal-detector/releases/tag/v0.1.0
