import numpy as np

from src.training.predict import predict
from src.common.language import prepare_text
from src.training.preprocess import clean_text
from src.api.schemas import (
    PredictRequest,
    PredictResponse,
    ExplainRequest,
    ExplainResponse,
    FeatureContribution,
)


def run_prediction(request: PredictRequest, model) -> PredictResponse:
    result = predict(request.text, model=model, model_type=request.model_type)
    return PredictResponse(
        label=result["label"],
        score_distress=result["score_distress"],
        model=result["model"],
        text_preview=request.text[:100],
        detected_lang=result.get("detected_lang", "en"),
    )


def run_explain(request: ExplainRequest, model) -> ExplainResponse:
    """Calcule les contributions SHAP pour le baseline TF-IDF + LR."""
    text_en, detected_lang = prepare_text(request.text)
    text_clean = clean_text(text_en)

    vectorizer = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    X = vectorizer.transform([text_clean])

    # Contribution linéaire de chaque terme : tfidf(w) × coef_LR(w).
    # C'est une approximation des valeurs de Shapley pour les modèles linéaires
    # (valide pour la partie "features", sans tenir compte de l'intercept ni de
    # l'hypothèse de distribution de SHAP). Pour une valeur SHAP stricte,
    # utiliser shap.LinearExplainer — cf. evaluate.py::explain_with_shap().
    tfidf_values = X.toarray()[0]           # (n_features,)
    coefs = clf.coef_[0]                    # (n_features,) — classe 1 (détresse)
    contributions = tfidf_values * coefs    # contribution de chaque mot

    feature_names = vectorizer.get_feature_names_out()

    # Garder uniquement les features présentes dans le texte
    nonzero_idx = np.where(tfidf_values > 0)[0]
    top_idx = nonzero_idx[np.argsort(np.abs(contributions[nonzero_idx]))[::-1]][: request.n_features]

    features = [
        FeatureContribution(word=feature_names[i], shap_value=float(contributions[i]))
        for i in top_idx
    ]

    proba = model.predict_proba([text_clean])[0]
    return ExplainResponse(
        label=int(proba.argmax()),
        score_distress=float(proba[1]),
        features=features,
        detected_lang=detected_lang,
    )
