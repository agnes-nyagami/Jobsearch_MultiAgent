from typing import List, Optional, Dict, Any
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import requests
import pandas as pd
import os
import json


load_dotenv()

# --- CrewAI persistent storage ---
os.environ["CREWAI_STORAGE_DIR"] = "./my_resume_storage"

# --- OpenAI client only ---
class OpenAIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def chat(self, prompt: str, temperature: float = 0.2) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1000
        }
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("No OpenAI API key found. Set OPENAI_API_KEY in your environment.")
LLM = OpenAIClient(OPENAI_KEY)


# --- Simple Indeed MCP client ---
class MCPIndeedClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def search_jobs(self, title: str, city: str, limit: int = 10) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/search"
        params = {"title": title, "city": city, "limit": limit}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("jobs", [])


# --- Simplified Local H-1B verifier (CSV only) ---
class MCPH1BClient:
    def __init__(self, local_csv: str):
        self.local_csv = local_csv

    def verify(self, company_name: str) -> Dict[str, Any]:
        if not os.path.exists(self.local_csv):
            return {"eligible": False, "records": 0, "note": "CSV not found"}
        df = pd.read_csv(self.local_csv)
        if "Employer" not in df.columns:
            return {"eligible": False, "records": 0, "note": "Missing 'Employer' column"}
        matches = df[df["Employer"].str.contains(company_name, case=False, na=False)]
        return {
            "eligible": not matches.empty,
            "records": len(matches),
            "note": "Employer found in USCIS H-1B dataset" if not matches.empty else "No record found"
        }


# --- Structured models ---
class JobListing(BaseModel):
    title: str
    company: str
    location: str
    description: str
    raw: Optional[Dict[str, Any]] = None


class VerificationResult(BaseModel):
    company: str
    eligible: bool
    records: int = 0
    note: Optional[str] = None


class TailoredResume(BaseModel):
    job_title: str
    company: str
    tailored_text: str
    change_summary: List[str] = Field(default_factory=list)


# --- Crew definition ---
@CrewBase
class JobSearchCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    indeed_mcp = MCPIndeedClient(os.getenv("MCP_INDEED_URL", "http://localhost:8001"))
    h1b_mcp = MCPH1BClient(local_csv=r"C:\Users\Aggy\AIJobtl\H-1B_DATA\H1B_2020_2023_summary.csv")

    @agent
    def job_search_agent(self) -> Agent:
        return Agent(config=self.agents_config.get("job_searcher"), tools=[], verbose=False)

    @agent
    def employer_verifier_agent(self) -> Agent:
        return Agent(config=self.agents_config.get("employer_verifier"), tools=[], verbose=False)

    @agent
    def resume_tailor_agent(self) -> Agent:
        return Agent(config=self.agents_config.get("resume_tailor"), tools=[], verbose=False)

    # --- Tasks ---
    @task
    def job_search_task(self) -> Task:
        def run(title: str, city: str, limit: int = 8) -> List[JobListing]:
            raw_jobs = self.indeed_mcp.search_jobs(title, city, limit)
            jobs = [JobListing(
                title=j.get("title") or j.get("job_title", ""),
                company=j.get("company") or j.get("employer", ""),
                location=j.get("location", city),
                description=j.get("description") or j.get("summary", ""),
                raw=j
            ) for j in raw_jobs]
            return jobs
        return Task(config=self.tasks_config.get("job_search_task"), func=run)

    @task
    def employer_verification_task(self) -> Task:
        def run(jobs: List[JobListing]) -> List[VerificationResult]:
            out = []
            for job in jobs:
                res = self.h1b_mcp.verify(job.company)
                out.append(VerificationResult(company=job.company, eligible=res["eligible"],
                                              records=res["records"], note=res["note"]))
            return out
        return Task(config=self.tasks_config.get("employer_verification_task"), func=run)

    @task
    def resume_tailoring_task(self) -> Task:
        def run(job: JobListing, resume_text: str) -> TailoredResume:
            prompt = f"""
You are a resume tailoring assistant.
Job Title: {job.title}
Company: {job.company}
Job Description:
{job.description}

User resume:
{resume_text}

Produce a tailored resume optimized for this job and a short bullet list of what was changed.
Return JSON with fields: 'tailored_text' and 'changes'.
"""
            raw_out = LLM.chat(prompt)
            try:
                parsed = json.loads(raw_out)
                tailored = parsed.get("tailored_text", raw_out)
                changes = parsed.get("changes", [])
            except Exception:
                tailored = raw_out
                changes = []
            return TailoredResume(job_title=job.title, company=job.company,
                                  tailored_text=tailored, change_summary=changes)
        return Task(config=self.tasks_config.get("resume_tailoring_task"), func=run)

    @task
    def orchestration_task(self) -> Task:
        def run(title: str, city: str, resume_text: str) -> Dict[str, Any]:
            jobs = self.job_search_task().run(title=title, city=city, limit=8)
            verifications = self.employer_verification_task().run(jobs=jobs)

            results = []
            for job, ver in zip(jobs, verifications):
                if not ver.eligible:
                    results.append({"job": job.dict(), "verified": ver.dict(), "tailored": None})
                    continue
                tailored = self.resume_tailoring_task().run(job=job, resume_text=resume_text)
                results.append({"job": job.dict(), "verified": ver.dict(), "tailored": tailored.dict()})
            return {"query": {"title": title, "city": city}, "results": results}
        return Task(config=self.tasks_config.get("workflow_orchestration_task"), func=run)

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, process=Process.sequential, verbose=2, memory=True)

