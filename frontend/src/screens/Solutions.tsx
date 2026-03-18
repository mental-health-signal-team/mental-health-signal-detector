import { useEffect, useState, useRef } from "react";
import { useNavigate, useLocation } from "react-router";
import { motion, AnimatePresence } from "motion/react";
import {
  Heart, ArrowRight, Phone, ExternalLink, ChevronDown,
  Sparkles, Shield, BookOpen, Wind, Users, HeartHandshake,
  Wind as BreathIcon, PenLine, MessageCircle, RotateCcw, Activity,
} from "lucide-react";
import { computeSolution } from "../lib/solutionEngine";
import { API_BASE } from "../lib/api";
import { saveSession, getSessions, getTrend } from "../lib/sessionHistory";
import type { SessionRecord, Trend } from "../lib/sessionHistory";
import type { DiagnosticProfile, ClinicalDimension } from "../types/diagnostic";
import type { SolutionResponse, MicroAction, Resource, TherapeuticBrick } from "../types/solutions";

// ─── Helpers visuels ─────────────────────────────────────────────────────────

const LEVEL_CONFIG = {
  0: { bg: "from-emerald-50 to-teal-50", badge: "bg-emerald-100 text-emerald-700", label: "Bien-être" },
  1: { bg: "from-sky-50 to-cyan-50",     badge: "bg-sky-100 text-sky-700",         label: "Léger" },
  2: { bg: "from-amber-50 to-orange-50", badge: "bg-amber-100 text-amber-700",     label: "Modéré" },
  3: { bg: "from-orange-50 to-red-50",   badge: "bg-orange-100 text-orange-700",   label: "Élevé" },
  4: { bg: "from-red-50 to-pink-50",     badge: "bg-red-100 text-red-700",         label: "Urgent" },
} as const;

const BRICK_ICON: Record<TherapeuticBrick, React.ComponentType<any>> = {
  cbt_activation:    BookOpen,
  cbt_restructuring: BookOpen,
  act:               Heart,
  mindfulness:       Wind,
  psychoeducation:   BookOpen,
  social_support:    Users,
  professional:      Users,
  crisis:            Shield,
};

const BRICK_LABEL: Record<TherapeuticBrick, string> = {
  cbt_activation:    "Activation comportementale",
  cbt_restructuring: "Restructuration cognitive",
  act:               "Acceptation & Engagement",
  mindfulness:       "Pleine conscience",
  psychoeducation:   "Psychoéducation",
  social_support:    "Soutien social",
  professional:      "Orientation professionnelle",
  crisis:            "Protocole de sécurité",
};

const DIM_LABEL: Record<ClinicalDimension, string> = {
  burnout:           "Épuisement",
  anxiety:           "Anxiété",
  depression_masked: "Humeur dépressive",
  dysregulation:     "Dysrégulation",
};

// ─── Sous-composants ─────────────────────────────────────────────────────────

// ─── Feedback local actions ───────────────────────────────────────────────────

type FeedbackValue = "helpful" | "not_helpful";

function saveActionFeedback(actionId: string, helpful: boolean): void {
  try {
    const raw = localStorage.getItem("mh_action_feedback");
    const all: { actionId: string; helpful: boolean; date: string }[] = raw ? JSON.parse(raw) : [];
    all.push({ actionId, helpful, date: new Date().toISOString() });
    localStorage.setItem("mh_action_feedback", JSON.stringify(all.slice(-50)));
  } catch {
    // localStorage indisponible → silencieux
  }
}

function ActionCard({ action, index, mode }: { action: MicroAction; index: number; mode: "kids" | "adult" }) {
  const [expanded, setExpanded]   = useState(false);
  const [feedback, setFeedback]   = useState<FeedbackValue | null>(null);
  const Icon = BRICK_ICON[action.brick];

  const handleFeedback = (value: FeedbackValue) => {
    setFeedback(value);
    saveActionFeedback(action.id, value === "helpful");
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 + index * 0.1 }}
      className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-md overflow-hidden"
    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 p-4 text-left"
      >
        <div className="w-10 h-10 bg-teal-100 rounded-xl flex items-center justify-center flex-shrink-0">
          <Icon className="w-5 h-5 text-teal-600" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="font-medium text-gray-800 text-sm">{action.title}</div>
          <div className="text-xs text-gray-400 mt-0.5">{action.duration}</div>
        </div>
        <motion.div animate={{ rotate: expanded ? 180 : 0 }} transition={{ duration: 0.2 }}>
          <ChevronDown className="w-4 h-4 text-gray-400" />
        </motion.div>
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <p className="px-4 pb-3 text-sm text-gray-600 leading-relaxed border-t border-gray-100 pt-3">
              {action.description}
            </p>

            {/* Feedback discret */}
            <div className="px-4 pb-4">
              {feedback === null ? (
                <div className="flex items-center gap-3 pt-1">
                  <span className="text-xs text-gray-400">
                    {mode === "kids" ? "Ça t'a aidé ?" : "Est-ce que ça vous a aidé ?"}
                  </span>
                  <button
                    onClick={() => handleFeedback("helpful")}
                    className="text-xs px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-600 hover:bg-emerald-100 transition-colors"
                  >
                    {mode === "kids" ? "👍 Oui !" : "Oui"}
                  </button>
                  <button
                    onClick={() => handleFeedback("not_helpful")}
                    className="text-xs px-2.5 py-1 rounded-full bg-gray-50 text-gray-500 hover:bg-gray-100 transition-colors"
                  >
                    {mode === "kids" ? "😐 Pas vraiment" : "Pas vraiment"}
                  </button>
                </div>
              ) : (
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="text-xs text-gray-400 pt-1"
                >
                  {feedback === "helpful"
                    ? (mode === "kids" ? "Super, content que ça t'ait aidé 💙" : "Merci pour votre retour.")
                    : (mode === "kids" ? "Noté. On essaiera autre chose la prochaine fois." : "Noté. D'autres pistes vous attendent.")}
                </motion.p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function ResourceCard({ resource, index }: { resource: Resource; index: number }) {
  const isPhone = resource.type === "phone";
  const isPerson = resource.type === "person";

  const content = (
    <div
      className={`w-full rounded-2xl p-4 flex items-center gap-3 shadow-md transition-all ${
        resource.urgent
          ? "bg-gradient-to-r from-pink-400 to-rose-400 text-white border-2 border-pink-300"
          : "bg-white/80 backdrop-blur-sm"
      }`}
    >
      <div
        className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 ${
          resource.urgent ? "bg-white/25" : "bg-teal-100"
        }`}
      >
        {isPhone || resource.urgent ? (
          <Phone className={`w-5 h-5 ${resource.urgent ? "text-white" : "text-teal-600"}`} />
        ) : isPerson ? (
          <Users className="w-5 h-5 text-teal-600" />
        ) : (
          <ExternalLink className="w-5 h-5 text-teal-600" />
        )}
      </div>
      <div className="flex-1 min-w-0">
        <div className={`font-medium text-sm ${resource.urgent ? "text-white" : "text-gray-800"}`}>
          {resource.label}
        </div>
        <div className={`text-xs mt-0.5 ${resource.urgent ? "text-white/80" : "text-gray-500"}`}>
          {resource.detail}
        </div>
        {resource.website && (
          <a
            href={resource.website}
            target="_blank"
            rel="noopener noreferrer"
            onClick={(e) => e.stopPropagation()}
            className={`inline-flex items-center gap-1 text-xs mt-1 underline underline-offset-2 ${
              resource.urgent ? "text-white/70 hover:text-white" : "text-teal-500 hover:text-teal-700"
            }`}
          >
            Voir le site
            <ExternalLink className="w-3 h-3" />
          </a>
        )}
      </div>
    </div>
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.5 + index * 0.08 }}
    >
      {resource.href ? (
        <a href={resource.href} target={isPhone ? "_self" : "_blank"} rel="noopener noreferrer">
          {content}
        </a>
      ) : (
        content
      )}
    </motion.div>
  );
}

// ─── Écran principal ─────────────────────────────────────────────────────────

export default function Solutions() {
  const navigate = useNavigate();
  const location = useLocation();

  const diagnosticProfile = location.state?.diagnosticProfile as DiagnosticProfile | undefined;
  const mode = diagnosticProfile?.mode ?? "kids";

  // Initialisation immédiate avec le moteur local (pas de flash de chargement)
  const [solution, setSolution] = useState<SolutionResponse | null>(() =>
    diagnosticProfile ? computeSolution(diagnosticProfile) : null
  );
  const abortRef = useRef<AbortController | null>(null);
  const isMountedRef = useRef(true);

  // Quick actions state
  const [activeQuick, setActiveQuick] = useState<string | null>(null);
  const [journalText, setJournalText] = useState("");
  const [breathPhase, setBreathPhase] = useState<"inhale" | "exhale">("inhale");

  // Historique de sessions — chargé une fois au montage
  const [trend, setTrend] = useState<Trend | null>(null);
  const [historyLength, setHistoryLength] = useState(0);
  const sessionSavedRef = useRef(false);

  // Guard : accès direct sans flow → redirection
  useEffect(() => {
    if (!diagnosticProfile) navigate("/", { replace: true });
  }, [diagnosticProfile, navigate]);

  // Sauvegarde de la session + calcul de tendance (une seule fois au montage)
  useEffect(() => {
    if (!diagnosticProfile || !solution || sessionSavedRef.current) return;
    sessionSavedRef.current = true;

    const record: SessionRecord = {
      date:           new Date().toISOString(),
      level:          solution.level,
      emotionId:      diagnosticProfile.emotionId,
      finalScore:     diagnosticProfile.finalScore,
      clinicalProfile: diagnosticProfile.clinicalProfile,
    };
    saveSession(record);

    const sessions = getSessions();
    setHistoryLength(sessions.length);
    setTrend(getTrend(sessions));
  }, [diagnosticProfile, solution]);

  // Cleanup au démontage
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      abortRef.current?.abort();
    };
  }, []);

  // Appel API en background — met à jour silencieusement (fondation LLM futur)
  useEffect(() => {
    if (!diagnosticProfile) return;
    abortRef.current = new AbortController();

    fetch(`${API_BASE}/solutions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(diagnosticProfile),
      signal: abortRef.current.signal,
    })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (!isMountedRef.current) return;
        // Map therapeuticBrick vers chaque action (l'API n'expose pas brick par action)
        setSolution({
          ...data,
          microActions: data.microActions.map((a: MicroAction) => ({
            ...a,
            brick: data.therapeuticBrick,
          })),
        });
      })
      .catch((err) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        // Le moteur local est déjà affiché — rien à faire
      });
  }, [diagnosticProfile]);

  // Breathing timer — actif uniquement quand le panneau respiration est ouvert
  useEffect(() => {
    if (activeQuick !== "breath") return;
    setBreathPhase("inhale");
    const id = setInterval(() => setBreathPhase((p) => (p === "inhale" ? "exhale" : "inhale")), 5000);
    return () => clearInterval(id);
  }, [activeQuick]);

  const handleQuickAction = (key: string) => {
    if (key === "talk") {
      document.getElementById("resources-section")?.scrollIntoView({ behavior: "smooth" });
      return;
    }
    if (key === "later") {
      navigate("/checkin", { state: { mode } });
      return;
    }
    setActiveQuick((prev) => (prev === key ? null : key));
  };

  if (!diagnosticProfile || !solution) return null;

  const levelConfig = LEVEL_CONFIG[solution.level];
  const BrickIcon = BRICK_ICON[solution.therapeuticBrick];

  return (
    <div className={`min-h-screen bg-gradient-to-b ${levelConfig.bg} flex flex-col`}>

      {/* Header */}
      <div className="px-6 pt-12 pb-4">
        <button
          onClick={() => navigate(-1)}
          className="text-gray-500 mb-4 hover:text-gray-700 transition-colors"
        >
          ← Retour
        </button>

        <motion.div
          initial={{ opacity: 0, y: -16 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          {/* Badge niveau — masqué en mode enfant */}
          {mode === "adult" && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", delay: 0.1 }}
              className="flex justify-center mb-4"
            >
              <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium ${levelConfig.badge}`}>
                <BrickIcon className="w-3.5 h-3.5" />
                {BRICK_LABEL[solution.therapeuticBrick]}
              </span>
            </motion.div>
          )}

          {/* Icône selon urgence */}
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", delay: 0.15 }}
            className="mb-5 inline-block"
          >
            <div className={`w-16 h-16 rounded-full flex items-center justify-center shadow-lg ${
              solution.level >= 4 ? "bg-gradient-to-br from-pink-400 to-rose-500"
              : solution.level >= 3 ? "bg-gradient-to-br from-orange-300 to-amber-400"
              : solution.level >= 2 ? "bg-gradient-to-br from-amber-300 to-yellow-400"
              : "bg-gradient-to-br from-teal-300 to-emerald-400"
            }`}>
              {solution.level >= 4 ? (
                mode === "kids"
                  ? <HeartHandshake className="w-8 h-8 text-white" />
                  : <Heart className="w-8 h-8 text-white" fill="white" />
              ) : solution.level >= 3 ? (
                <Shield className="w-8 h-8 text-white" />
              ) : solution.level >= 2 ? (
                <Heart className="w-8 h-8 text-white" fill="white" />
              ) : (
                <Sparkles className="w-8 h-8 text-white" />
              )}
            </div>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-xl font-medium px-4"
            style={{ color: "#2A5F7D" }}
          >
            {mode === "kids"
              ? solution.level >= 4 ? "Tu n'es pas seul"
                : solution.level >= 3 ? "Je suis là pour t'aider"
                : solution.level >= 2 ? "Voilà ce qui peut t'aider"
                : "Bravo, tu prends soin de toi !"
              : solution.level >= 4 ? "Vous n'êtes pas seul(e)"
                : solution.level >= 3 ? "Des solutions concrètes pour vous"
                : solution.level >= 2 ? "Vos pistes d'action"
                : "Cultivez votre bien-être"}
          </motion.h1>
        </motion.div>
      </div>

      {/* Contenu */}
      <div className="flex-1 px-6 pb-10 overflow-y-auto space-y-5">

        {/* Message empathique */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.25 }}
          className={`rounded-3xl p-5 shadow-lg ${
            solution.level >= 4
              ? "bg-gradient-to-br from-orange-100 to-pink-100 border-2 border-pink-200"
              : "bg-white/80 backdrop-blur-sm"
          }`}
        >
          <div className="flex items-start gap-3">
            <Heart className="w-5 h-5 text-pink-400 flex-shrink-0 mt-0.5" fill="currentColor" />
            <p className="text-gray-700 leading-relaxed text-sm">{solution.message}</p>
          </div>
        </motion.div>

        {/* Score discret — adulte, niveaux 1-3 uniquement */}
        {mode === "adult" && solution.level >= 1 && solution.level <= 3 && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.35 }}
            className="bg-white/60 backdrop-blur-sm rounded-2xl p-4 shadow-sm"
          >
            <div className="flex items-center gap-2 mb-3">
              <Activity className="w-3.5 h-3.5 text-gray-400" />
              <span className="text-xs text-gray-400 font-medium uppercase tracking-wide">Analyse</span>
            </div>

            {/* Niveau de triage — points */}
            <div className="flex items-center gap-3 mb-2">
              <span className="text-xs text-gray-500 w-16 shrink-0">Niveau</span>
              <div className="flex gap-1.5">
                {[0, 1, 2, 3, 4].map((l) => (
                  <div
                    key={l}
                    className={`w-2 h-2 rounded-full ${
                      l <= solution.level
                        ? solution.level >= 3 ? "bg-orange-400"
                          : solution.level >= 2 ? "bg-amber-400"
                          : "bg-sky-400"
                        : "bg-gray-200"
                    }`}
                  />
                ))}
              </div>
              <span className={`text-xs font-medium ${levelConfig.badge} px-2 py-0.5 rounded-full`}>
                {levelConfig.label}
              </span>
            </div>

            {/* Score de détresse */}
            {diagnosticProfile.finalScore !== null && (
              <div className="flex items-center gap-3 mb-2">
                <span className="text-xs text-gray-500 w-16 shrink-0">Score</span>
                <div className="flex-1 bg-gray-100 rounded-full h-1.5">
                  <div
                    className={`h-1.5 rounded-full ${
                      solution.level >= 3 ? "bg-orange-400"
                        : solution.level >= 2 ? "bg-amber-400"
                        : "bg-sky-400"
                    }`}
                    style={{ width: `${Math.round(diagnosticProfile.finalScore * 100)}%` }}
                  />
                </div>
                <span className="text-xs text-gray-400 w-8 text-right">
                  {Math.round(diagnosticProfile.finalScore * 100)}%
                </span>
              </div>
            )}

            {/* Dimensions cliniques détectées */}
            {diagnosticProfile.clinicalDimensions.length > 0 && (
              <div className="flex items-start gap-3">
                <span className="text-xs text-gray-500 w-16 shrink-0 pt-0.5">Thèmes</span>
                <div className="flex flex-wrap gap-1.5">
                  {diagnosticProfile.clinicalDimensions.map((dim) => (
                    <span key={dim} className="text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded-full">
                      {DIM_LABEL[dim]}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Tendance longitudinale — visible dès la 2e session */}
            {trend !== null && historyLength >= 2 && (
              <div className="flex items-center gap-3 mt-1 pt-2 border-t border-gray-100">
                <span className="text-xs text-gray-500 w-16 shrink-0">Évolution</span>
                <div className="flex items-center gap-2">
                  <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${
                    trend === "improving" ? "bg-emerald-100 text-emerald-700"
                    : trend === "worsening" ? "bg-orange-100 text-orange-700"
                    : "bg-gray-100 text-gray-600"
                  }`}>
                    {trend === "improving" ? "↓ en amélioration"
                     : trend === "worsening" ? "↑ dégradation"
                     : "→ stable"}
                  </span>
                  <span className="text-xs text-gray-400">vs session précédente</span>
                </div>
              </div>
            )}
          </motion.div>
        )}

        {/* Tendance simplifiée — mode enfant, dès 2e session, niveaux 0-3 */}
        {mode === "kids" && trend !== null && historyLength >= 2 && solution.level < 4 && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.38 }}
            className={`rounded-2xl px-4 py-3 text-sm text-center ${
              trend === "improving" ? "bg-emerald-50 text-emerald-700"
              : trend === "worsening" ? "bg-orange-50 text-orange-700"
              : "bg-sky-50 text-sky-700"
            }`}
          >
            {trend === "improving"
              ? "Tu progresses — c'est vraiment bien !"
              : trend === "worsening"
              ? "Ça semble plus difficile aujourd'hui. Tu n'es pas seul."
              : "Tu continues à prendre soin de toi."}
          </motion.div>
        )}

        {/* Ressources urgentes — niveau 4 en tête */}
        {solution.escalationRequired && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
            className="space-y-2"
          >
            <p className="text-sm font-medium text-gray-600 px-1">
              {mode === "kids" ? "Des personnes sont là pour toi maintenant" : "Aide disponible immédiatement"}
            </p>
            {solution.resources.filter((r) => r.urgent).map((r, i) => (
              <ResourceCard key={r.id} resource={r} index={i} />
            ))}
          </motion.div>
        )}

        {/* Micro-actions */}
        {solution.microActions.length > 0 && (
          <div className="space-y-2">
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.28 }}
              className="text-sm font-medium text-gray-600 px-1"
            >
              {mode === "kids" ? "Ce que tu peux faire maintenant" : "Actions concrètes"}
            </motion.p>
            {solution.microActions.map((action, i) => (
              <ActionCard key={action.id} action={action} index={i} mode={mode} />
            ))}
          </div>
        )}

        {/* Ressources non-urgentes */}
        {solution.resources.filter((r) => !r.urgent).length > 0 && (
          <div id="resources-section" className="space-y-2">
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
              className="text-sm font-medium text-gray-600 px-1"
            >
              {mode === "kids" ? "Si tu veux aller plus loin" : "Pour aller plus loin"}
            </motion.p>
            {solution.resources.filter((r) => !r.urgent).map((r, i) => (
              <ResourceCard key={r.id} resource={r} index={i} />
            ))}
          </div>
        )}

        {/* Et maintenant ? — 4 actions interactives (masqué niveau 4) */}
        {solution.level < 4 && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.65 }}
            className="space-y-3"
          >
            <p className="text-sm font-medium text-gray-600 px-1">
              {mode === "kids" ? "Et maintenant, qu'est-ce qui t'aiderait le plus ?" : "Et maintenant, qu'est-ce qui vous aiderait le plus ?"}
            </p>

            <div className="grid grid-cols-2 gap-2">
              {[
                { key: "breath", Icon: BreathIcon,    label: mode === "kids" ? "Respirer 2 min"           : "Respirer 2 minutes" },
                { key: "write",  Icon: PenLine,        label: mode === "kids" ? "Écrire ce que je ressens" : "Écrire ce que je ressens" },
                { key: "talk",   Icon: MessageCircle,  label: mode === "kids" ? "Parler à quelqu'un"       : "Parler à quelqu'un" },
                { key: "later",  Icon: RotateCcw,      label: mode === "kids" ? "Refaire un point plus tard" : "Refaire un point plus tard" },
              ].map(({ key, Icon, label }) => (
                <button
                  key={key}
                  onClick={() => handleQuickAction(key)}
                  className={`bg-white/70 backdrop-blur-sm rounded-2xl p-3 flex items-center gap-2 shadow-sm text-left transition-all hover:bg-white/90 ${
                    activeQuick === key ? "ring-2 ring-teal-300 bg-white/90" : ""
                  }`}
                >
                  <Icon className="w-4 h-4 text-teal-500 flex-shrink-0" />
                  <span className="text-xs text-gray-600 leading-tight">{label}</span>
                </button>
              ))}
            </div>

            {/* Contenu expansible */}
            <AnimatePresence>
              {activeQuick === "breath" && (
                <motion.div
                  key="breath-panel"
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.25 }}
                  className="overflow-hidden"
                >
                  <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-5 flex flex-col items-center gap-4">
                    <p className="text-xs text-gray-500 text-center">
                      {mode === "kids" ? "Suis le cercle avec ta respiration" : "Suivez le cercle avec votre respiration"}
                    </p>
                    <motion.div
                      animate={{ scale: breathPhase === "inhale" ? 1.5 : 1, opacity: breathPhase === "inhale" ? 1 : 0.55 }}
                      transition={{ duration: 5, ease: "easeInOut" }}
                      className="w-20 h-20 rounded-full bg-gradient-to-br from-teal-300 to-cyan-400 shadow-lg"
                    />
                    <p className="text-sm font-medium text-teal-600">
                      {breathPhase === "inhale" ? "Inspirez..." : "Expirez..."}
                    </p>
                    <p className="text-xs text-gray-400">5 secondes · répétez 3 minutes</p>
                  </div>
                </motion.div>
              )}

              {activeQuick === "write" && (
                <motion.div
                  key="write-panel"
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.25 }}
                  className="overflow-hidden"
                >
                  <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-4 space-y-3">
                    <p className="text-xs text-gray-500 italic">
                      {mode === "kids"
                        ? "En quelques mots, qu'est-ce qui pèse en ce moment ?"
                        : "En quelques mots, qu'est-ce qui pèse en ce moment ?"}
                    </p>
                    <textarea
                      value={journalText}
                      onChange={(e) => setJournalText(e.target.value)}
                      placeholder={mode === "kids" ? "Écris ici, juste pour toi..." : "Écrivez ici, pour vous seulement..."}
                      className="w-full min-h-[80px] bg-gray-50 rounded-xl p-3 text-sm text-gray-700 resize-none focus:outline-none focus:ring-2 focus:ring-teal-200 placeholder:text-gray-400"
                    />
                    <p className="text-xs text-gray-400 text-right">
                      🔒 {mode === "kids" ? "Ces mots restent sur ton appareil" : "Ce texte reste sur votre appareil"}
                    </p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}

        {/* Clôture */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.72 }}
          className="text-center px-4"
        >
          <p className="text-sm text-gray-500 italic leading-relaxed">
            {solution.closing}
          </p>
        </motion.div>

        {/* CTA → CheckIn */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
        >
          <button
            onClick={() =>
              navigate("/checkin", {
                state: { mode, diagnosticProfile },
              })
            }
            className="w-full bg-gradient-to-r from-teal-400 via-cyan-400 to-blue-400 text-white rounded-3xl py-5 shadow-lg hover:shadow-xl transition-all flex items-center justify-center gap-3 font-medium"
          >
            <span>
              {mode === "kids" ? "Programmer un prochain point" : "Planifier un suivi"}
            </span>
            <ArrowRight className="w-5 h-5" />
          </button>
        </motion.div>

      </div>
    </div>
  );
}
