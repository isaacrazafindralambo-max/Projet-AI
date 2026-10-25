# AI tools and prompts — GetReview AI

## Tools used
- Google Colab: original notebook and model experimentation.
- Hugging Face models: sentiment classification, multilingual sentence embeddings, FLAN-T5 generation (see README).
- ChatGPT: adaptation of existing notebook to Streamlit, preservation of date data, month-over-month satisfaction, top-three pain points, UI improvements, and deployment instructions.

## User prompts supplied to ChatGPT (translated only where necessary)
1. "J'ai un fichier .ipynb d'un code python fait sur google collab. J'aimerais en faire une app ... en faisant le moins de modifications possible."
2. "Je souhaite une application accessible par un lien web. Prépare les fichiers et le déploiement en conservant au maximum mon code."
3. "fais moi un pas-a-pas détaillé"
4. "L'overview devrait présenter également un score de satisfaction du mois avec comparaison avec le mois précédent, ainsi qu'un graphique permettant de suivre la satisfaction au cours des mois ... le rapport IA ... top 3 des pain points ... interface ... plus ergonomique et esthétique ... modifiant principalement le notebook"

## Disclosure
ChatGPT-generated adaptations must be reviewed and tested. The model-based analysis logic and operational action library originate from the supplied notebook; no claim is made that ChatGPT trained the Hugging Face models. Add any other tools and prompts actually used.
