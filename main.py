import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ─── App Init ─────────────────────────────────────────────────────────────────
app = FastAPI(title="SkillNova API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"

# ─── Language Config ───────────────────────────────────────────────────────────
LANGUAGE_INSTRUCTIONS = {
    "en": "Respond in English.",
    "ta": "Respond in Tamil (தமிழ்). Use Tamil script throughout.",
    "hi": "Respond in Hindi (हिन्दी). Use Devanagari script throughout.",
    "te": "Respond in Telugu (తెలుగు). Use Telugu script throughout.",
    "ml": "Respond in Malayalam (മലയാളം). Use Malayalam script throughout.",
    "fr": "Respond in French (Français).",
    "de": "Respond in German (Deutsch).",
    "zh": "Respond in Simplified Chinese (中文).",
    "ar": "Respond in Arabic (العربية). Use Arabic script throughout.",
    "es": "Respond in Spanish (Español).",
    "ja": "Respond in Japanese (日本語). Use Japanese script throughout.",
    "ko": "Respond in Korean (한국어). Use Korean script throughout.",
}

SUPPORTED_LANGUAGES = list(LANGUAGE_INSTRUCTIONS.keys())

# ─── Career Knowledge Base ─────────────────────────────────────────────────────
CAREER_DATA = {
    "Data Scientist": {
        "skills": ["Python", "Statistics", "Machine Learning", "Deep Learning", "SQL",
                   "Data Visualization", "Feature Engineering", "Model Deployment",
                   "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "Communication"],
        "description": "Data Scientists extract insights from complex data using statistical analysis and machine learning. They build predictive models and communicate findings to stakeholders.",
        "avg_salary": "$95,000 - $150,000",
        "top_companies": ["Google", "Meta", "Amazon", "Netflix", "Airbnb"],
        "certifications": ["Google Data Analytics", "IBM Data Science", "Coursera ML Specialization"],
    },
    "ML Engineer": {
        "skills": ["Python", "Machine Learning", "Deep Learning", "MLOps", "Docker",
                   "Kubernetes", "Cloud (AWS/GCP/Azure)", "CI/CD", "Model Optimization",
                   "PyTorch", "TensorFlow", "API Development", "Data Pipelines"],
        "description": "ML Engineers build, deploy, and maintain machine learning systems at scale. They bridge the gap between research and production.",
        "avg_salary": "$110,000 - $175,000",
        "top_companies": ["OpenAI", "DeepMind", "NVIDIA", "Tesla", "Microsoft"],
        "certifications": ["AWS ML Specialty", "GCP ML Engineer", "MLflow Certification"],
    },
    "Software Engineer": {
        "skills": ["Data Structures", "Algorithms", "System Design", "Python/Java/C++",
                   "Git", "SQL", "REST APIs", "Testing", "CI/CD", "Cloud Basics",
                   "OOP", "Problem Solving", "Code Review"],
        "description": "Software Engineers design, develop, and maintain software systems. Strong fundamentals in algorithms and system design are essential.",
        "avg_salary": "$85,000 - $160,000",
        "top_companies": ["FAANG", "Microsoft", "Salesforce", "Stripe", "Shopify"],
        "certifications": ["AWS Developer", "Google Associate Cloud Engineer", "Oracle Java"],
    },
    "Web Developer": {
        "skills": ["HTML", "CSS", "JavaScript", "React", "Node.js", "SQL/NoSQL",
                   "REST APIs", "Git", "Responsive Design", "TypeScript",
                   "Web Performance", "Testing", "Deployment"],
        "description": "Web Developers build and maintain websites and web applications. Full-stack developers work on both frontend and backend.",
        "avg_salary": "$70,000 - $130,000",
        "top_companies": ["Shopify", "Squarespace", "HubSpot", "Atlassian", "Vercel"],
        "certifications": ["Meta Front-End Developer", "freeCodeCamp", "AWS Solutions Architect"],
    },
    "Data Analyst": {
        "skills": ["SQL", "Excel", "Python/R", "Data Visualization", "Statistics",
                   "Tableau/Power BI", "Business Intelligence", "ETL", "Communication",
                   "Dashboard Design", "A/B Testing", "Data Cleaning"],
        "description": "Data Analysts interpret data to help businesses make informed decisions. They create reports, dashboards, and visualizations.",
        "avg_salary": "$60,000 - $100,000",
        "top_companies": ["Deloitte", "McKinsey", "Accenture", "IBM", "Capgemini"],
        "certifications": ["Google Data Analytics", "Microsoft Power BI", "Tableau Desktop Specialist"],
    },
    "Cloud Engineer": {
        "skills": ["AWS/GCP/Azure", "Linux", "Networking", "Docker", "Kubernetes",
                   "Terraform", "CI/CD", "Security", "Monitoring", "Python/Bash",
                   "Cost Optimization", "Microservices", "Serverless"],
        "description": "Cloud Engineers design and manage cloud infrastructure. They ensure scalability, security, and cost efficiency.",
        "avg_salary": "$100,000 - $165,000",
        "top_companies": ["AWS", "Google Cloud", "Microsoft Azure", "Cloudflare", "HashiCorp"],
        "certifications": ["AWS Solutions Architect", "GCP Professional", "Azure Administrator"],
    },
    "Cybersecurity Analyst": {
        "skills": ["Networking", "Linux", "Python", "Security Protocols", "SIEM",
                   "Penetration Testing", "Cryptography", "Incident Response",
                   "Compliance", "Threat Intelligence", "Firewalls", "Risk Assessment"],
        "description": "Cybersecurity Analysts protect organizations from digital threats. They monitor systems, investigate incidents, and implement security measures.",
        "avg_salary": "$80,000 - $140,000",
        "top_companies": ["CrowdStrike", "Palo Alto Networks", "FireEye", "IBM Security", "CISA"],
        "certifications": ["CompTIA Security+", "CEH", "CISSP", "OSCP"],
    },
    "DevOps Engineer": {
        "skills": ["Linux", "Docker", "Kubernetes", "CI/CD", "Git", "Cloud Platforms",
                   "Scripting (Python/Bash)", "Monitoring", "Infrastructure as Code",
                   "Terraform", "Ansible", "Jenkins", "Agile/Scrum"],
        "description": "DevOps Engineers streamline software delivery by automating deployment pipelines and bridging development and operations.",
        "avg_salary": "$95,000 - $155,000",
        "top_companies": ["GitLab", "GitHub", "HashiCorp", "Red Hat", "Puppet"],
        "certifications": ["Docker Certified", "Kubernetes CKA", "AWS DevOps Engineer"],
    },
    "AI Research Scientist": {
        "skills": ["Deep Learning", "PyTorch", "Mathematics", "Research Methods",
                   "Python", "NLP", "Computer Vision", "Reinforcement Learning",
                   "Paper Reading", "Academic Writing", "Statistics", "CUDA", "Distributed Training"],
        "description": "AI Research Scientists push the boundaries of artificial intelligence. They publish papers, develop novel architectures, and advance the field.",
        "avg_salary": "$130,000 - $250,000",
        "top_companies": ["OpenAI", "DeepMind", "Anthropic", "Meta AI", "Google Brain"],
        "certifications": ["PhD recommended", "Stanford AI courses", "Fast.ai"],
    },
    "Product Manager (Tech)": {
        "skills": ["Product Strategy", "User Research", "SQL", "Analytics",
                   "Communication", "Agile", "Roadmapping", "A/B Testing",
                   "Wireframing", "Stakeholder Management", "Market Analysis", "Leadership"],
        "description": "Technical Product Managers define product vision and strategy. They work with engineers, designers, and business teams to deliver impactful products.",
        "avg_salary": "$100,000 - $170,000",
        "top_companies": ["Google", "Apple", "Spotify", "Airbnb", "LinkedIn"],
        "certifications": ["PMP", "Pragmatic Institute", "Product School"],
    },
}

GENERAL_DOCS = """
PLACEMENT TIPS:
1. Start applying 6 months before graduation.
2. Build a portfolio with 3-5 real projects on GitHub.
3. Practice LeetCode (150+ problems) for coding interviews.
4. Network on LinkedIn — connect with alumni and professionals.
5. Tailor your resume for each job application using keywords from JDs.
6. Prepare STAR-format answers for behavioral interviews.
7. Research company culture before every interview.

HIGHER STUDIES:
- MS in CS/Data Science: Apply to top 50 US universities. GRE optional at many schools.
- MBA Tech Track: Good for transitioning into Product Management.
- PhD: Best for AI Research roles. Focus on publications and research fit.
- Online Masters: Georgia Tech OMSCS ($7,000 total) — world-class value.
- Scholarships: Fulbright, DAAD, Chevening, Commonwealth scholarships.

FREE LEARNING RESOURCES:
- Coursera (audit free): ML Specialization, Deep Learning Specialization
- fast.ai: Practical Deep Learning for Coders
- CS50: Harvard's free intro to CS
- Kaggle: Free ML courses + competitions
- YouTube: Sentdex, 3Blue1Brown, StatQuest, Andrej Karpathy
- Books: "Hands-On ML" (free PDF), "Deep Learning" by Goodfellow
- GitHub: Awesome-* lists for every tech domain

RESUME TIPS:
- 1 page for < 3 years experience
- Use action verbs: Built, Designed, Optimized, Reduced, Improved
- Quantify everything: "Improved model accuracy by 15%"
- Include: Education, Skills, Projects, Experience, Certifications
- ATS-friendly: Plain text, standard headings, no tables

INTERVIEW PREPARATION:
- Technical: DS&A on LeetCode, System Design (Grokking), Domain knowledge
- HR/Behavioral: STAR method, research company values
- Mock interviews: Pramp.com, interviewing.io (free)
- Salary negotiation: Always negotiate, research Glassdoor/Levels.fyi
"""

# ─── ChromaDB Vector Store Setup ───────────────────────────────────────────────
def build_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = []

    for role, data in CAREER_DATA.items():
        content = f"""
Career Role: {role}
Description: {data['description']}
Required Skills: {', '.join(data['skills'])}
Average Salary: {data['avg_salary']}
Top Companies: {', '.join(data['top_companies'])}
Certifications: {', '.join(data['certifications'])}
        """.strip()
        chunks = splitter.split_documents([Document(page_content=content, metadata={"role": role})])
        docs.extend(chunks)

    general_chunks = splitter.split_documents([Document(page_content=GENERAL_DOCS, metadata={"role": "general"})])
    docs.extend(general_chunks)

    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
    return vectorstore

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
try:
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    if vectorstore._collection.count() == 0:
        raise Exception("Empty collection")
except Exception:
    vectorstore = build_vectorstore()

# ─── Request Models ────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str
    student_skills: List[str] = []
    goal: str = "Software Engineer"
    language: str = "en"

class SkillGapRequest(BaseModel):
    student_skills: List[str]
    goal: str
    cgpa: Optional[float] = None

class UpdateSkillRequest(BaseModel):
    skill: str
    student_name: str = "Student"
    goal: str = "Software Engineer"

# ─── Helpers ───────────────────────────────────────────────────────────────────
def get_skill_gap(student_skills: List[str], goal: str):
    required = CAREER_DATA.get(goal, {}).get("skills", [])
    student_lower = [s.lower() for s in student_skills]
    have = [s for s in required if s.lower() in student_lower]
    missing = [s for s in required if s.lower() not in student_lower]
    readiness = round(len(have) / len(required) * 100) if required else 0
    return required, have, missing, readiness

def rag_retrieve(query: str, k: int = 4) -> str:
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([d.page_content for d in docs])

def llm_call(system_prompt: str, user_message: str, language: str = "en") -> str:
    lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["en"])
    full_system = f"{system_prompt}\n\nLanguage instruction: {lang_instruction}"
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_message},
        ],
        max_tokens=1500,
        temperature=0.7,
    )
    return response.choices[0].message.content

# ─── API Endpoints ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "SkillNova API is running 🚀", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": GROQ_MODEL, "vectorstore_docs": vectorstore._collection.count()}

@app.get("/languages")
def languages():
    return {"supported": SUPPORTED_LANGUAGES, "default": "en"}

@app.get("/careers")
def careers():
    return {"careers": list(CAREER_DATA.keys())}

@app.post("/ask")
def ask(req: AskRequest):
    context = rag_retrieve(f"{req.query} {req.goal}")
    _, have, missing, readiness = get_skill_gap(req.student_skills, req.goal)

    system_prompt = f"""You are SkillNova — an expert AI career mentor for university students.
You have deep knowledge of tech careers, skill gaps, learning paths, and placement strategies.
Be encouraging, specific, and actionable. Use the context below to inform your answer.

Student Profile:
- Goal: {req.goal}
- Current Skills: {', '.join(req.student_skills) if req.student_skills else 'Not specified'}
- Skills Acquired: {', '.join(have) if have else 'None yet'}
- Missing Skills: {', '.join(missing[:5]) if missing else 'None'}
- Job Readiness: {readiness}%

Knowledge Base Context:
{context}
"""
    answer = llm_call(system_prompt, req.query, req.language)
    return {"answer": answer, "readiness": readiness, "language": req.language}

@app.post("/skill-gap")
def skill_gap(req: SkillGapRequest):
    required, have, missing, readiness = get_skill_gap(req.student_skills, req.goal)

    priority_map = {s: i + 1 for i, s in enumerate(required)}
    missing_prioritized = sorted(missing, key=lambda s: priority_map.get(s, 99))

    cgpa_note = ""
    if req.cgpa:
        if req.cgpa >= 8.5:
            cgpa_note = "Excellent CGPA — eligible for on-campus placements at top firms."
        elif req.cgpa >= 7.0:
            cgpa_note = "Good CGPA — eligible for most placements. Focus on skill development."
        else:
            cgpa_note = "Lower CGPA — compensate with strong projects, internships, and certifications."

    return {
        "goal": req.goal,
        "readiness_percent": readiness,
        "required_skills": required,
        "have_skills": have,
        "missing_skills": missing_prioritized,
        "cgpa_note": cgpa_note,
        "total_required": len(required),
        "total_have": len(have),
        "total_missing": len(missing),
    }

@app.post("/update-skill")
def update_skill(req: UpdateSkillRequest):
    new_doc = Document(
        page_content=f"Student {req.student_name} has learned {req.skill}. "
                     f"This is a new skill acquired towards the goal of {req.goal}. "
                     f"With {req.skill}, the student can now work on more advanced topics.",
        metadata={"type": "student_progress", "skill": req.skill, "goal": req.goal}
    )
    vectorstore.add_documents([new_doc])
    # Note: Chroma 0.4+ auto-persists — no .persist() needed

    next_advice = llm_call(
        f"You are SkillNova. A student just learned {req.skill} towards becoming a {req.goal}.",
        f"What should they learn next after mastering {req.skill}? Give 3 specific next steps.",
        "en"
    )
    return {"success": True, "skill_added": req.skill, "next_steps": next_advice}

@app.get("/projects/{goal}")
def get_projects(goal: str, language: str = "en"):
    context = rag_retrieve(f"projects portfolio {goal}")
    answer = llm_call(
        f"You are SkillNova. Generate practical project ideas for someone targeting {goal} roles.",
        f"List 5 portfolio projects for a {goal} candidate. For each: title, description, tech stack, difficulty (Beginner/Intermediate/Advanced), and GitHub tags. Format as a structured list.",
        language
    )
    return {"goal": goal, "projects": answer, "language": language}

@app.get("/roadmap/{goal}")
def get_roadmap(goal: str, language: str = "en"):
    context = rag_retrieve(f"learning roadmap {goal} career path")
    answer = llm_call(
        f"You are SkillNova. Create a detailed, month-by-month learning roadmap.",
        f"Create a 6-month learning roadmap for becoming a {goal}. Include: monthly goals, key skills to learn, resources, projects to build, and milestones. Be specific and actionable.",
        language
    )
    return {"goal": goal, "roadmap": answer, "language": language}

@app.get("/interview-prep/{goal}")
def interview_prep(goal: str, language: str = "en"):
    context = rag_retrieve(f"interview preparation {goal} tips")
    answer = llm_call(
        f"You are SkillNova. Provide comprehensive interview preparation guidance.",
        f"Provide complete interview preparation for {goal} roles. Include: technical topics, common questions, system design topics, behavioral questions (STAR format), and final tips.",
        language
    )
    return {"goal": goal, "interview_prep": answer, "language": language}
