# SkillNova — AI Career Mentor System
## Complete Deployment Guide ($0 Forever)

---

## 🏗️ Architecture
```
Student → React Frontend (Vercel)
              ↓
         FastAPI Backend (Render.com)
              ↓
    Groq LLaMA 3.3 + ChromaDB RAG
```

---

## PART 1 — BACKEND DEPLOYMENT (Render.com)

### Step 1: Prepare Files
Upload to a GitHub repo called `skillnova-backend`:
- `main.py`
- `requirements.txt`

### Step 2: Get Free Groq API Key
1. Go to https://console.groq.com
2. Sign up free → Create API Key
3. Copy: `gsk_xxxxxxxxxxxxxxxxxxxx`

### Step 3: Deploy on Render.com
1. Go to https://render.com → Sign up free
2. Click **New → Web Service**
3. Connect your GitHub repo `skillnova-backend`
4. Configure:
   - **Name**: skillnova-api
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port 10000`
5. Add Environment Variable:
   - Key: `GROQ_API_KEY`
   - Value: `gsk_xxxxxxxxxxxxxxxxxxxx`
6. Click **Create Web Service**
7. Wait ~5 mins for deploy
8. Your URL: `https://skillnova-api.onrender.com`

### Test Backend
Visit: `https://skillnova-api.onrender.com/health`
Should return: `{"status":"healthy","model":"llama-3.3-70b-versatile",...}`

---

## PART 2 — FRONTEND DEPLOYMENT (Vercel)

### Step 1: Update API URL
In `src/App.jsx` line 3, replace:
```javascript
const API_URL = "https://your-render-url.onrender.com";
// → change to your actual Render URL:
const API_URL = "https://skillnova-api.onrender.com";
```

### Step 2: Push to GitHub
Create repo `skillnova-frontend`, push all files.

### Step 3: Deploy on Vercel
1. Go to https://vercel.com → Sign up free
2. Click **New Project**
3. Import your GitHub repo `skillnova-frontend`
4. Framework: **Vite**
5. Click **Deploy**
6. Your URL: `https://skillnova-frontend.vercel.app`

---

## 📁 File Structure
```
skillnova-backend/
├── main.py              ← FastAPI app (all endpoints)
└── requirements.txt     ← Python dependencies

skillnova-frontend/
├── index.html           ← HTML entry
├── package.json         ← Node dependencies
├── vite.config.js       ← Vite config
└── src/
    ├── main.jsx         ← React entry
    └── App.jsx          ← Complete app (6 tabs)
```

---

## 🌐 API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/health` | Status + model info |
| GET | `/languages` | Supported languages |
| GET | `/careers` | Career list |
| POST | `/ask` | RAG + LLM mentor chat |
| POST | `/skill-gap` | Missing skills + readiness % |
| POST | `/update-skill` | Continual learning |
| GET | `/projects/{goal}` | Project suggestions |
| GET | `/roadmap/{goal}` | Learning roadmap |
| GET | `/interview-prep/{goal}` | Interview guide |

---

## 🌍 Supported Languages
`en` `ta` `hi` `te` `ml` `fr` `de` `zh` `ar` `es` `ja` `ko`

---

## 💡 Features
- **6 Tabs**: Profile, Skill Gap, Projects, Roadmap, AI Mentor, Placement
- **Animated Readiness Ring**: Live % with color coding
- **Continual Learning**: Mark skill → ChromaDB updates → readiness recalculates
- **Multi-lingual**: 12 languages including RTL Arabic
- **RAG Architecture**: Vector search over 10 career role knowledge bases
- **Offline Fallback**: Skill gap computed locally if API unreachable

---

## 🆓 Tools Used (All Free)
| Tool | Purpose |
|------|---------|
| Groq API | Free LLaMA 3.3 70B LLM |
| ChromaDB | Persistent vector store |
| sentence-transformers | Local embeddings |
| FastAPI | Python API framework |
| Render.com | Free API hosting |
| Vercel | Free React hosting |
| GitHub | Code + auto-deploy trigger |

**Total Cost: $0 forever ✅**
