# AI-Powered Investor Intelligence Platform

Application locale pour indexer des rapports financiers, extraire leurs indicateurs et poser des questions sur leur contenu. Les PDF, les passages, les indicateurs et les questions sont traités sur ton ordinateur. Internet sert au téléchargement initial des modèles et des dépendances.

| Besoin | Composant local |
| --- | --- |
| Convertir les PDF en Markdown | PyMuPDF4LLM |
| Découper les rapports | LangChain et le modèle d'embeddings `embeddinggemma` |
| Stocker les passages et les indicateurs | SQLite, dans `data/investor.sqlite3` |
| Retrouver les passages pertinents | Similarité cosinus avec NumPy |
| Extraire les indicateurs et répondre aux questions | Ollama et `llama3.2:3b` |
| Afficher le tableau de bord | FastAPI sur `127.0.0.1:8000` |

## 1. Démarrer Ollama

Si Ollama tourne déjà, quitte l'application depuis la zone de notification Windows. Dans une fenêtre PowerShell, lance le serveur et garde la fenêtre ouverte.

```powershell
$env:OLLAMA_NO_CLOUD = '1'
ollama serve
```

Dans un deuxième PowerShell, vérifie les modèles. Télécharge seulement ceux qui manquent.

```powershell
ollama list
ollama pull embeddinggemma
ollama pull llama3.2:3b
```

## 2. Préparer Python

Dans le dossier du projet, utilise Python 3.13 et `uv`. Si le dossier `.venv` existe déjà, commence à la deuxième commande.

```powershell
uv venv --python 3.13
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

Les paramètres par défaut suffisent. Pour les changer, copie `.env.example` vers `.env` et renseigne le nom de tes modèles **locaux** ou le chemin du fichier SQLite.

## 3. Tester le découpage seul

```powershell
.\.venv\Scripts\python.exe -X utf8 -m ingestion.semantic_chunker
```

La commande lit `data/markdown/2024_Apple.md`, affiche le nombre de passages et montre un extrait des trois premiers. Passe le chemin d'un autre Markdown en argument pour changer de rapport.

## 4. Lancer le tableau de bord et importer un PDF

```powershell
.\.venv\Scripts\python.exe app.py
```

Ouvre [http://127.0.0.1:8000](http://127.0.0.1:8000). Dépose `data/raw_pdfs/2024_Apple.pdf` dans **Ingest Report**, attends la fin de l'indexation, puis sélectionne **Apple (2024)** dans le chatbot. Essaie la question « Quel est le chiffre d'affaires total d'Apple en 2024 ? ». Compare le chiffre et son unité au tableau du rapport.

Tu peux aussi importer le PDF depuis PowerShell. Pour traiter plusieurs rapports, passe le chemin d'un dossier.

```powershell
.\.venv\Scripts\python.exe -m ingestion.ingest_documents data/raw_pdfs/2024_Apple.pdf
```

Les noms `2024_Apple.pdf` et `Apple_2024.pdf` permettent de retrouver l'entreprise et l'année. Réactualise le tableau de bord après un import en ligne de commande. Si tu changes de modèle d'embeddings, réimporte les rapports pour recalculer les vecteurs avec le nouveau modèle.

## 5. Inspecter les passages du RAG

```powershell
.\.venv\Scripts\python.exe -X utf8 -m rag.retrieval_debug --company Apple --year 2024 "total net sales 2024"
```

Le script affiche les passages les mieux classés et leur fichier source. L'index et les indicateurs restent dans SQLite entre deux lancements. Les rapports convertis sont disponibles dans `data/markdown`.
