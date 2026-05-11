# Git Setup — Connecting Your Local Folder to GitHub

Follow these steps **once** from the Anaconda Prompt to connect your local folder to the GitHub repo.

---

## Step 1 — Navigate to Your Project Folder

```bash
cd D:\AI_Engineer-RAG
```

---

## Step 2 — Initialize Git

```bash
git init
```

This turns the folder into a Git repository. You'll see a `.git` folder appear (hidden by default).

---

## Step 3 — Connect to GitHub

```bash
git remote add origin https://github.com/AbdelRahman-Madboly/AI_Engineer-RAG.git
```

---

## Step 4 — Stage All Files

```bash
git add .
```

The `.gitignore` file will automatically exclude your conda environment, API keys, and other files that should not be committed.

---

## Step 5 — Make Your First Commit

```bash
git commit -m "Initial commit: project structure and Module 1 knowledge base"
```

---

## Step 6 — Push to GitHub

```bash
git branch -M main
git push -u origin main
```

If prompted for credentials, enter your GitHub username and a **Personal Access Token** (not your password).  
To create a token: GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic) → Generate new token. Give it `repo` scope.

---

## Setting Up the Conda Environment

```bash
# Create the environment from the file
conda env create -f environment.yml

# Activate it
conda activate rag-course

# Verify
python --version
# Should show Python 3.11.x

# Register it as a Jupyter kernel (so VS Code and Jupyter can use it)
python -m ipykernel install --user --name rag-course --display-name "RAG Course (Python 3.11)"
```

---

## Using VS Code with This Project

1. Open VS Code
2. File → Open Folder → select `D:\AI_Engineer-RAG`
3. In any `.ipynb` file, click the kernel selector (top right) → select **RAG Course (Python 3.11)**
4. The Git panel (source control icon on left sidebar) will show your changes

---

## Day-to-Day Git Workflow

After working on files, push your changes:

```bash
# See what changed
git status

# Stage everything
git add .

# Commit with a meaningful message
git commit -m "Add Module 1 quiz and LLM notes"

# Push to GitHub
git push
```

---

## API Key Setup (Never Commit This)

Create a file called `.env` in the project root:

```
TOGETHER_API_KEY=your_key_here
```

This file is in `.gitignore` — it will **never** be pushed to GitHub.

In your Python code or notebooks:

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
```
