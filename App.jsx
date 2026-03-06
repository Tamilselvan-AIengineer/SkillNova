import { useState, useEffect, useRef } from "react";

const API_URL = "https://your-render-url.onrender.com"; // ← Replace with your Render URL

// ─── Career List ────────────────────────────────────────────────────────────────
const CAREERS = [
  "Data Scientist", "ML Engineer", "Software Engineer", "Web Developer",
  "Data Analyst", "Cloud Engineer", "Cybersecurity Analyst", "DevOps Engineer",
  "AI Research Scientist", "Product Manager (Tech)"
];

const ALL_SKILLS = [
  "Python", "SQL", "JavaScript", "React", "Node.js", "Machine Learning",
  "Deep Learning", "Statistics", "Data Visualization", "Docker", "Kubernetes",
  "AWS", "Git", "HTML", "CSS", "TypeScript", "MongoDB", "PostgreSQL",
  "TensorFlow", "PyTorch", "Pandas", "NumPy", "Scikit-learn", "Linux",
  "REST APIs", "System Design", "Algorithms", "Data Structures", "Java", "C++",
];

// ─── i18n ────────────────────────────────────────────────────────────────────
const LANGS = [
  { code: "en", flag: "🇺🇸", name: "English" },
  { code: "ta", flag: "🇮🇳", name: "தமிழ்" },
  { code: "hi", flag: "🇮🇳", name: "हिन्दी" },
  { code: "te", flag: "🇮🇳", name: "తెలుగు" },
  { code: "ml", flag: "🇮🇳", name: "മലയാളം" },
  { code: "fr", flag: "🇫🇷", name: "Français" },
  { code: "de", flag: "🇩🇪", name: "Deutsch" },
  { code: "zh", flag: "🇨🇳", name: "中文" },
  { code: "ar", flag: "🇸🇦", name: "العربية" },
  { code: "es", flag: "🇪🇸", name: "Español" },
  { code: "ja", flag: "🇯🇵", name: "日本語" },
  { code: "ko", flag: "🇰🇷", name: "한국어" },
];

const UI_TEXT = {
  en: { profile: "Profile", skillgap: "Skill Gap", projects: "Projects", roadmap: "Roadmap", mentor: "AI Mentor", placement: "Placement", yourGoal: "Your Goal", yourSkills: "Your Skills", readiness: "Job Readiness", analyse: "Analyse My Profile", missing: "Missing Skills", have: "Skills You Have", ask: "Ask your mentor...", send: "Send", generate: "Generate", loading: "Thinking...", cgpa: "Your CGPA" },
  ta: { profile: "சுயவிவரம்", skillgap: "திறன் இடைவெளி", projects: "திட்டங்கள்", roadmap: "வழிமாப்பு", mentor: "AI வழிகாட்டி", placement: "வேலைவாய்ப்பு", yourGoal: "உங்கள் இலக்கு", yourSkills: "உங்கள் திறன்கள்", readiness: "வேலை தயார்நிலை", analyse: "என் சுயவிவரத்தை பகுப்பாய்வு செய்", missing: "இல்லாத திறன்கள்", have: "உங்களிடம் உள்ள திறன்கள்", ask: "உங்கள் வழிகாட்டியிடம் கேளுங்கள்...", send: "அனுப்பு", generate: "உருவாக்கு", loading: "யோசிக்கிறேன்...", cgpa: "உங்கள் CGPA" },
  hi: { profile: "प्रोफ़ाइल", skillgap: "कौशल अंतर", projects: "प्रोजेक्ट", roadmap: "रोडमैप", mentor: "AI गुरु", placement: "प्लेसमेंट", yourGoal: "आपका लक्ष्य", yourSkills: "आपके कौशल", readiness: "नौकरी तत्परता", analyse: "मेरी प्रोफ़ाइल विश्लेषण करें", missing: "अनुपस्थित कौशल", have: "आपके पास कौशल", ask: "अपने गुरु से पूछें...", send: "भेजें", generate: "बनाएं", loading: "सोच रहा हूँ...", cgpa: "आपका CGPA" },
  es: { profile: "Perfil", skillgap: "Brecha de Habilidades", projects: "Proyectos", roadmap: "Hoja de Ruta", mentor: "Mentor IA", placement: "Empleo", yourGoal: "Tu Objetivo", yourSkills: "Tus Habilidades", readiness: "Preparación Laboral", analyse: "Analizar Mi Perfil", missing: "Habilidades Faltantes", have: "Habilidades Que Tienes", ask: "Pregunta a tu mentor...", send: "Enviar", generate: "Generar", loading: "Pensando...", cgpa: "Tu CGPA" },
  fr: { profile: "Profil", skillgap: "Écart de Compétences", projects: "Projets", roadmap: "Feuille de Route", mentor: "Mentor IA", placement: "Emploi", yourGoal: "Votre Objectif", yourSkills: "Vos Compétences", readiness: "Préparation à l'Emploi", analyse: "Analyser Mon Profil", missing: "Compétences Manquantes", have: "Compétences Acquises", ask: "Demandez à votre mentor...", send: "Envoyer", generate: "Générer", loading: "Réflexion...", cgpa: "Votre CGPA" },
};

const t = (lang, key) => (UI_TEXT[lang] || UI_TEXT["en"])[key] || UI_TEXT["en"][key];

const SUGGESTED_QUESTIONS = {
  en: ["How do I become a Data Scientist?", "What projects should I build?", "How do I prepare for FAANG?", "Which skills should I learn first?"],
  ta: ["Data Scientist ஆவது எப்படி?", "என்ன projects கட்டவேண்டும்?", "FAANG-க்கு எப்படி தயார் ஆவது?"],
  hi: ["Data Scientist कैसे बनें?", "कौन से प्रोजेक्ट बनाएं?", "FAANG की तैयारी कैसे करें?"],
};

// ─── Styles ────────────────────────────────────────────────────────────────────
const S = {
  app: { background: "#0a0a0f", minHeight: "100vh", fontFamily: "'DM Sans', sans-serif", color: "#e8e0d0" },
  header: { borderBottom: "1px solid #1e1a14", padding: "0 2rem", display: "flex", alignItems: "center", justifyContent: "space-between", height: "64px", background: "rgba(10,10,15,0.95)", backdropFilter: "blur(12px)", position: "sticky", top: 0, zIndex: 50 },
  logo: { fontFamily: "'Playfair Display', serif", fontSize: "1.5rem", fontWeight: 700, background: "linear-gradient(135deg, #f59e0b, #ef4444)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" },
  logoSub: { fontFamily: "'DM Mono', monospace", fontSize: "0.6rem", color: "#6b5e45", letterSpacing: "0.2em", textTransform: "uppercase", display: "block", marginTop: "-4px" },
  nav: { display: "flex", gap: "0.25rem", background: "#111118", borderRadius: "12px", padding: "4px", border: "1px solid #1e1a14" },
  navBtn: (active) => ({ padding: "6px 14px", borderRadius: "8px", border: "none", cursor: "pointer", fontSize: "0.8rem", fontFamily: "'DM Sans', sans-serif", fontWeight: 500, transition: "all 0.2s", background: active ? "linear-gradient(135deg, #f59e0b22, #ef444422)" : "transparent", color: active ? "#f59e0b" : "#6b5e45", borderColor: active ? "#f59e0b44" : "transparent", borderWidth: active ? "1px" : 0, borderStyle: "solid" }),
  langSelect: { background: "#111118", border: "1px solid #1e1a14", color: "#e8e0d0", padding: "6px 12px", borderRadius: "8px", fontSize: "0.85rem", cursor: "pointer", fontFamily: "'DM Sans', sans-serif" },
  main: { maxWidth: "1200px", margin: "0 auto", padding: "2rem" },
  card: { background: "#0f0f18", border: "1px solid #1e1a14", borderRadius: "16px", padding: "1.5rem", marginBottom: "1rem" },
  cardTitle: { fontFamily: "'Playfair Display', serif", fontSize: "1.1rem", color: "#f59e0b", marginBottom: "1rem", fontWeight: 600 },
  grid2: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" },
  grid3: { display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "1rem" },
  label: { fontSize: "0.75rem", color: "#6b5e45", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "0.5rem", display: "block" },
  select: { width: "100%", background: "#111118", border: "1px solid #2a2218", color: "#e8e0d0", padding: "10px 14px", borderRadius: "10px", fontSize: "0.9rem", fontFamily: "'DM Sans', sans-serif", cursor: "pointer" },
  input: { width: "100%", background: "#111118", border: "1px solid #2a2218", color: "#e8e0d0", padding: "10px 14px", borderRadius: "10px", fontSize: "0.9rem", fontFamily: "'DM Sans', sans-serif", boxSizing: "border-box" },
  btn: { background: "linear-gradient(135deg, #f59e0b, #ef4444)", border: "none", color: "#000", padding: "10px 24px", borderRadius: "10px", cursor: "pointer", fontWeight: 700, fontFamily: "'DM Sans', sans-serif", fontSize: "0.9rem", transition: "opacity 0.2s" },
  btnSm: { background: "#1a1a24", border: "1px solid #2a2218", color: "#e8e0d0", padding: "5px 12px", borderRadius: "7px", cursor: "pointer", fontSize: "0.78rem", fontFamily: "'DM Sans', sans-serif", transition: "all 0.2s" },
  tag: (active) => ({ display: "inline-block", padding: "4px 10px", borderRadius: "20px", fontSize: "0.75rem", cursor: "pointer", margin: "3px", transition: "all 0.2s", background: active ? "linear-gradient(135deg, #f59e0b22, #ef444422)" : "#111118", border: active ? "1px solid #f59e0b66" : "1px solid #1e1a14", color: active ? "#f59e0b" : "#6b5e45", fontWeight: active ? 600 : 400 }),
  prose: { fontSize: "0.9rem", lineHeight: "1.8", color: "#b8a890", whiteSpace: "pre-wrap" },
  ring: { position: "relative", width: "120px", height: "120px" },
  missingSkill: { display: "flex", alignItems: "center", gap: "8px", padding: "8px 12px", borderRadius: "8px", background: "#110b0b", border: "1px solid #2a1414", marginBottom: "6px" },
  haveSkill: { display: "flex", alignItems: "center", gap: "8px", padding: "8px 12px", borderRadius: "8px", background: "#0b110b", border: "1px solid #142a14", marginBottom: "6px" },
  chatBubble: (isUser) => ({ display: "flex", flexDirection: isUser ? "row-reverse" : "row", gap: "10px", marginBottom: "1rem", alignItems: "flex-start" }),
  chatMsg: (isUser) => ({ maxWidth: "75%", padding: "12px 16px", borderRadius: isUser ? "18px 18px 4px 18px" : "18px 18px 18px 4px", background: isUser ? "linear-gradient(135deg, #f59e0b22, #ef444422)" : "#111118", border: isUser ? "1px solid #f59e0b33" : "1px solid #1e1a14", fontSize: "0.88rem", lineHeight: "1.7", color: "#e8e0d0", whiteSpace: "pre-wrap" }),
  stat: { textAlign: "center", padding: "1rem" },
  statNum: { fontFamily: "'Playfair Display', serif", fontSize: "2.2rem", background: "linear-gradient(135deg, #f59e0b, #ef4444)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", fontWeight: 700 },
  statLabel: { fontSize: "0.72rem", color: "#6b5e45", textTransform: "uppercase", letterSpacing: "0.1em" },
};

// ─── Progress Ring SVG ─────────────────────────────────────────────────────────
function ReadinessRing({ percent }) {
  const r = 50, cx = 60, cy = 60;
  const circ = 2 * Math.PI * r;
  const dash = (percent / 100) * circ;
  const color = percent >= 70 ? "#22c55e" : percent >= 40 ? "#f59e0b" : "#ef4444";
  return (
    <div style={{ position: "relative", width: 120, height: 120, margin: "0 auto" }}>
      <svg width="120" height="120" style={{ transform: "rotate(-90deg)" }}>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="#1a1a24" strokeWidth="10" />
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={color} strokeWidth="10"
          strokeDasharray={`${dash} ${circ}`} strokeLinecap="round"
          style={{ transition: "stroke-dasharray 1s ease" }} />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
        <span style={{ fontFamily: "'Playfair Display', serif", fontSize: "1.6rem", fontWeight: 700, color }}>{percent}%</span>
        <span style={{ fontSize: "0.6rem", color: "#6b5e45", textTransform: "uppercase", letterSpacing: "0.1em" }}>Ready</span>
      </div>
    </div>
  );
}

// ─── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("profile");
  const [lang, setLang] = useState("en");
  const [goal, setGoal] = useState("Data Scientist");
  const [skills, setSkills] = useState(["Python", "SQL"]);
  const [cgpa, setCgpa] = useState("");
  const [skillGapData, setSkillGapData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [chatMessages, setChatMessages] = useState([{ role: "assistant", content: "👋 Hello! I'm SkillNova, your AI career mentor. Tell me your goal and I'll guide you to success. What would you like to know?" }]);
  const [chatInput, setChatInput] = useState("");
  const [projectsData, setProjectsData] = useState("");
  const [roadmapData, setRoadmapData] = useState("");
  const [interviewData, setInterviewData] = useState("");
  const [localReadiness, setLocalReadiness] = useState(0);
  const chatEndRef = useRef(null);
  const isRTL = lang === "ar";

  // Compute local readiness whenever skills/goal change
  useEffect(() => {
    if (skillGapData) setLocalReadiness(skillGapData.readiness_percent);
  }, [skillGapData]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  const toggleSkill = (skill) => {
    setSkills(prev => prev.includes(skill) ? prev.filter(s => s !== skill) : [...prev, skill]);
  };

  const analyseProfile = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/skill-gap`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ student_skills: skills, goal, cgpa: cgpa ? parseFloat(cgpa) : null }),
      });
      const data = await res.json();
      setSkillGapData(data);
      setLocalReadiness(data.readiness_percent);
      setTab("skillgap");
    } catch {
      // Fallback: compute locally
      const required = { "Data Scientist": ["Python", "Statistics", "Machine Learning", "Deep Learning", "SQL", "Data Visualization", "Feature Engineering", "Model Deployment", "Pandas", "NumPy", "Scikit-learn", "TensorFlow"], "Software Engineer": ["Data Structures", "Algorithms", "System Design", "Python", "Git", "SQL", "REST APIs", "Testing", "Cloud Basics", "OOP"] };
      const req = required[goal] || required["Software Engineer"];
      const sl = skills.map(s => s.toLowerCase());
      const have = req.filter(s => sl.includes(s.toLowerCase()));
      const missing = req.filter(s => !sl.includes(s.toLowerCase()));
      const readiness = Math.round(have.length / req.length * 100);
      setSkillGapData({ goal, readiness_percent: readiness, required_skills: req, have_skills: have, missing_skills: missing, total_required: req.length, total_have: have.length, total_missing: missing.length, cgpa_note: "" });
      setLocalReadiness(readiness);
      setTab("skillgap");
    }
    setLoading(false);
  };

  const sendChat = async (msg) => {
    const q = msg || chatInput;
    if (!q.trim()) return;
    setChatInput("");
    setChatMessages(prev => [...prev, { role: "user", content: q }]);
    setChatMessages(prev => [...prev, { role: "assistant", content: "⏳ " + t(lang, "loading") }]);
    try {
      const res = await fetch(`${API_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, student_skills: skills, goal, language: lang }),
      });
      const data = await res.json();
      setChatMessages(prev => [...prev.slice(0, -1), { role: "assistant", content: data.answer }]);
    } catch {
      setChatMessages(prev => [...prev.slice(0, -1), { role: "assistant", content: "⚠️ Could not reach API. Please check your connection or try again." }]);
    }
  };

  const fetchProjects = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/projects/${encodeURIComponent(goal)}?language=${lang}`);
      const data = await res.json();
      setProjectsData(data.projects);
    } catch { setProjectsData("⚠️ API unavailable. Please check your Render URL."); }
    setLoading(false);
  };

  const fetchRoadmap = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/roadmap/${encodeURIComponent(goal)}?language=${lang}`);
      const data = await res.json();
      setRoadmapData(data.roadmap);
    } catch { setRoadmapData("⚠️ API unavailable. Please check your Render URL."); }
    setLoading(false);
  };

  const fetchInterview = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/interview-prep/${encodeURIComponent(goal)}?language=${lang}`);
      const data = await res.json();
      setInterviewData(data.interview_prep);
    } catch { setInterviewData("⚠️ API unavailable. Please check your Render URL."); }
    setLoading(false);
  };

  const markSkillLearned = async (skill) => {
    if (!skills.includes(skill)) {
      setSkills(prev => [...prev, skill]);
      try {
        await fetch(`${API_URL}/update-skill`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ skill, student_name: "Student", goal }),
        });
        await analyseProfile();
      } catch {}
    }
  };

  const tabs = ["profile", "skillgap", "projects", "roadmap", "mentor", "placement"];
  const tabLabels = { profile: t(lang, "profile"), skillgap: t(lang, "skillgap"), projects: t(lang, "projects"), roadmap: t(lang, "roadmap"), mentor: t(lang, "mentor"), placement: t(lang, "placement") };

  return (
    <div style={{ ...S.app, direction: isRTL ? "rtl" : "ltr" }}>
      <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet" />

      {/* Header */}
      <header style={S.header}>
        <div>
          <span style={S.logo}>SkillNova</span>
          <span style={S.logoSub}>AI Career Mentor</span>
        </div>
        <nav style={S.nav}>
          {tabs.map(tb => (
            <button key={tb} style={S.navBtn(tab === tb)} onClick={() => setTab(tb)}>
              {tabLabels[tb]}
            </button>
          ))}
        </nav>
        <select style={S.langSelect} value={lang} onChange={e => setLang(e.target.value)}>
          {LANGS.map(l => <option key={l.code} value={l.code}>{l.flag} {l.name}</option>)}
        </select>
      </header>

      <main style={S.main}>

        {/* ── PROFILE TAB ─────────────────────────────────────────────────────── */}
        {tab === "profile" && (
          <div style={{ animation: "fadeUp 0.4s ease" }}>
            <div style={S.grid2}>
              <div style={S.card}>
                <p style={S.cardTitle}>🎯 {t(lang, "yourGoal")}</p>
                <label style={S.label}>Target Career</label>
                <select style={S.select} value={goal} onChange={e => setGoal(e.target.value)}>
                  {CAREERS.map(c => <option key={c}>{c}</option>)}
                </select>
                <div style={{ marginTop: "1rem" }}>
                  <label style={S.label}>{t(lang, "cgpa")} (optional)</label>
                  <input style={S.input} type="number" min="0" max="10" step="0.1"
                    placeholder="e.g. 8.5" value={cgpa} onChange={e => setCgpa(e.target.value)} />
                </div>
                <div style={{ marginTop: "1.5rem", textAlign: "center" }}>
                  <ReadinessRing percent={localReadiness} />
                </div>
              </div>

              <div style={S.card}>
                <p style={S.cardTitle}>⚡ {t(lang, "yourSkills")}</p>
                <p style={{ fontSize: "0.78rem", color: "#6b5e45", marginBottom: "0.75rem" }}>Click to toggle skills you know:</p>
                <div style={{ maxHeight: "240px", overflowY: "auto" }}>
                  {ALL_SKILLS.map(skill => (
                    <span key={skill} style={S.tag(skills.includes(skill))} onClick={() => toggleSkill(skill)}>
                      {skills.includes(skill) ? "✓ " : ""}{skill}
                    </span>
                  ))}
                </div>
                <div style={{ marginTop: "0.75rem", fontSize: "0.78rem", color: "#6b5e45" }}>
                  {skills.length} skills selected
                </div>
              </div>
            </div>

            <div style={{ textAlign: "center", marginTop: "0.5rem" }}>
              <button style={{ ...S.btn, padding: "14px 40px", fontSize: "1rem" }} onClick={analyseProfile} disabled={loading}>
                {loading ? "⏳ Analysing..." : `🔍 ${t(lang, "analyse")}`}
              </button>
            </div>

            <div style={{ ...S.card, marginTop: "1rem", background: "linear-gradient(135deg, #0f0f18, #111118)" }}>
              <p style={S.cardTitle}>✨ Quick Stats</p>
              <div style={S.grid3}>
                <div style={S.stat}><div style={S.statNum}>{skills.length}</div><div style={S.statLabel}>Skills</div></div>
                <div style={S.stat}><div style={S.statNum}>{localReadiness}%</div><div style={S.statLabel}>Readiness</div></div>
                <div style={S.stat}><div style={S.statNum}>{CAREERS.indexOf(goal) + 1}</div><div style={S.statLabel}>Career #{CAREERS.indexOf(goal) + 1}</div></div>
              </div>
            </div>
          </div>
        )}

        {/* ── SKILL GAP TAB ───────────────────────────────────────────────────── */}
        {tab === "skillgap" && (
          <div style={{ animation: "fadeUp 0.4s ease" }}>
            {!skillGapData ? (
              <div style={{ ...S.card, textAlign: "center", padding: "3rem" }}>
                <p style={{ color: "#6b5e45", marginBottom: "1rem" }}>Run profile analysis first</p>
                <button style={S.btn} onClick={() => setTab("profile")}>← Go to Profile</button>
              </div>
            ) : (
              <>
                <div style={{ ...S.card, textAlign: "center" }}>
                  <p style={S.cardTitle}>📊 {goal} — Skill Analysis</p>
                  <ReadinessRing percent={skillGapData.readiness_percent} />
                  <div style={S.grid3}>
                    <div style={S.stat}><div style={S.statNum}>{skillGapData.total_have}</div><div style={S.statLabel}>Have</div></div>
                    <div style={S.stat}><div style={S.statNum}>{skillGapData.total_missing}</div><div style={S.statLabel}>Missing</div></div>
                    <div style={S.stat}><div style={S.statNum}>{skillGapData.total_required}</div><div style={S.statLabel}>Required</div></div>
                  </div>
                  {skillGapData.cgpa_note && <p style={{ fontSize: "0.85rem", color: "#f59e0b", marginTop: "0.75rem", padding: "8px 16px", background: "#f59e0b11", borderRadius: "8px", display: "inline-block" }}>🎓 {skillGapData.cgpa_note}</p>}
                </div>

                <div style={S.grid2}>
                  <div style={S.card}>
                    <p style={{ ...S.cardTitle, color: "#ef4444" }}>🔴 {t(lang, "missing")} (Priority Order)</p>
                    {skillGapData.missing_skills.map((skill, i) => (
                      <div key={skill} style={S.missingSkill}>
                        <span style={{ fontFamily: "'DM Mono', monospace", fontSize: "0.7rem", color: "#ef444466", width: "20px" }}>#{i + 1}</span>
                        <span style={{ flex: 1, fontSize: "0.85rem" }}>{skill}</span>
                        <button style={{ ...S.btnSm, fontSize: "0.7rem", color: "#22c55e" }} onClick={() => markSkillLearned(skill)}>+ Mark Learned</button>
                      </div>
                    ))}
                  </div>
                  <div style={S.card}>
                    <p style={{ ...S.cardTitle, color: "#22c55e" }}>✅ {t(lang, "have")}</p>
                    {skillGapData.have_skills.map(skill => (
                      <div key={skill} style={S.haveSkill}>
                        <span style={{ color: "#22c55e" }}>✓</span>
                        <span style={{ fontSize: "0.85rem" }}>{skill}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* ── PROJECTS TAB ────────────────────────────────────────────────────── */}
        {tab === "projects" && (
          <div style={{ animation: "fadeUp 0.4s ease" }}>
            <div style={S.card}>
              <p style={S.cardTitle}>🛠️ Portfolio Projects — {goal}</p>
              <p style={{ fontSize: "0.85rem", color: "#6b5e45", marginBottom: "1rem" }}>AI-generated project ideas tailored for your target role</p>
              <button style={S.btn} onClick={fetchProjects} disabled={loading}>
                {loading ? "⏳ Generating..." : `⚡ ${t(lang, "generate")} Projects`}
              </button>
            </div>
            {projectsData && (
              <div style={S.card}>
                <pre style={S.prose}>{projectsData}</pre>
              </div>
            )}
          </div>
        )}

        {/* ── ROADMAP TAB ─────────────────────────────────────────────────────── */}
        {tab === "roadmap" && (
          <div style={{ animation: "fadeUp 0.4s ease" }}>
            <div style={S.card}>
              <p style={S.cardTitle}>🗺️ 6-Month Learning Roadmap — {goal}</p>
              <p style={{ fontSize: "0.85rem", color: "#6b5e45", marginBottom: "1rem" }}>Personalised month-by-month plan to reach your goal</p>
              <button style={S.btn} onClick={fetchRoadmap} disabled={loading}>
                {loading ? "⏳ Generating..." : `🗺️ ${t(lang, "generate")} Roadmap`}
              </button>
            </div>
            {roadmapData && (
              <div style={S.card}>
                <pre style={S.prose}>{roadmapData}</pre>
              </div>
            )}
          </div>
        )}

        {/* ── AI MENTOR TAB ───────────────────────────────────────────────────── */}
        {tab === "mentor" && (
          <div style={{ animation: "fadeUp 0.4s ease" }}>
            <div style={{ ...S.card, marginBottom: "0.5rem" }}>
              <p style={S.cardTitle}>💬 AI Career Mentor</p>
              <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                {(SUGGESTED_QUESTIONS[lang] || SUGGESTED_QUESTIONS["en"]).map(q => (
                  <button key={q} style={S.btnSm} onClick={() => sendChat(q)}>{q}</button>
                ))}
              </div>
            </div>

            <div style={{ ...S.card, height: "400px", overflowY: "auto", padding: "1rem" }}>
              {chatMessages.map((msg, i) => (
                <div key={i} style={S.chatBubble(msg.role === "user")}>
                  <div style={{ width: 32, height: 32, borderRadius: "50%", background: msg.role === "user" ? "linear-gradient(135deg, #f59e0b, #ef4444)" : "#1a1a24", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "0.9rem", flexShrink: 0 }}>
                    {msg.role === "user" ? "👤" : "🤖"}
                  </div>
                  <div style={S.chatMsg(msg.role === "user")}>{msg.content}</div>
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>

            <div style={{ display: "flex", gap: "8px", marginTop: "0.5rem" }}>
              <input style={{ ...S.input, flex: 1 }} placeholder={t(lang, "ask")} value={chatInput}
                onChange={e => setChatInput(e.target.value)}
                onKeyDown={e => e.key === "Enter" && sendChat()} />
              <button style={S.btn} onClick={() => sendChat()}>{t(lang, "send")}</button>
            </div>
          </div>
        )}

        {/* ── PLACEMENT TAB ───────────────────────────────────────────────────── */}
        {tab === "placement" && (
          <div style={{ animation: "fadeUp 0.4s ease" }}>
            <div style={S.card}>
              <p style={S.cardTitle}>🎯 Interview Preparation — {goal}</p>
              <button style={S.btn} onClick={fetchInterview} disabled={loading}>
                {loading ? "⏳ Generating..." : "🎯 Get Interview Guide"}</button>
            </div>
            {interviewData && (
              <div style={S.card}><pre style={S.prose}>{interviewData}</pre></div>
            )}

            <div style={S.grid2}>
              <div style={S.card}>
                <p style={S.cardTitle}>📋 Placement Checklist</p>
                {["Resume polished (1 page)", "LinkedIn profile optimized", "GitHub with 3+ projects", "LeetCode 100+ problems solved", "Mock interviews done (Pramp)", "References ready", "Offer negotiation prepared"].map(item => (
                  <CheckItem key={item} label={item} />
                ))}
              </div>
              <div style={S.card}>
                <p style={S.cardTitle}>🎓 Higher Studies Paths</p>
                <div style={{ fontSize: "0.85rem", color: "#b8a890", lineHeight: 2 }}>
                  <p>🇺🇸 <strong>MS CS/Data Science</strong> — Apply 6 months early, GRE optional at many schools</p>
                  <p>💻 <strong>Georgia Tech OMSCS</strong> — World-class online MS, ~$7,000 total</p>
                  <p>🏛️ <strong>PhD in AI/ML</strong> — Best for research roles. Focus on publications</p>
                  <p>💼 <strong>MBA Tech</strong> — Transition into Product Management</p>
                  <p>🌍 <strong>Scholarships</strong> — Fulbright, DAAD, Chevening, Commonwealth</p>
                </div>
              </div>
            </div>

            <div style={S.card}>
              <p style={S.cardTitle}>🆓 Free Learning Resources</p>
              <div style={S.grid3}>
                {[["Coursera", "Audit ML & DS courses free", "🎓"], ["fast.ai", "Practical Deep Learning", "🚀"], ["CS50", "Harvard's free intro to CS", "🏛️"], ["Kaggle", "Competitions + free courses", "📊"], ["YouTube", "3Blue1Brown, Sentdex, Karpathy", "▶️"], ["GitHub", "Awesome-* curated lists", "💻"]].map(([name, desc, icon]) => (
                  <div key={name} style={{ ...S.card, margin: 0 }}>
                    <div style={{ fontSize: "1.5rem", marginBottom: "4px" }}>{icon}</div>
                    <div style={{ fontWeight: 600, fontSize: "0.9rem", color: "#f59e0b" }}>{name}</div>
                    <div style={{ fontSize: "0.78rem", color: "#6b5e45", marginTop: "4px" }}>{desc}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

      </main>

      <style>{`
        @keyframes fadeUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
        ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: #0a0a0f; } ::-webkit-scrollbar-thumb { background: #2a2218; border-radius: 2px; }
        select option { background: #111118; }
        button:hover { opacity: 0.85; }
      `}</style>
    </div>
  );
}

function CheckItem({ label }) {
  const [checked, setChecked] = useState(false);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "10px", padding: "8px 0", borderBottom: "1px solid #1a1a1a", cursor: "pointer" }} onClick={() => setChecked(!checked)}>
      <div style={{ width: 18, height: 18, borderRadius: "4px", border: checked ? "none" : "1px solid #2a2218", background: checked ? "linear-gradient(135deg, #f59e0b, #ef4444)" : "transparent", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
        {checked && <span style={{ color: "#000", fontSize: "0.7rem", fontWeight: 700 }}>✓</span>}
      </div>
      <span style={{ fontSize: "0.85rem", color: checked ? "#6b5e45" : "#e8e0d0", textDecoration: checked ? "line-through" : "none" }}>{label}</span>
    </div>
  );
}
