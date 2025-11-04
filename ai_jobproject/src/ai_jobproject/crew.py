from typing import List, Optional, Dict, Any
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool, MDXSearchTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import requests
import pandas as pd
import os
import json

load_dotenv()

# --- OpenAI client (unchanged) ---
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

    def search_jobs(self, title: str, city: str, limit: int = 3) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/search"
        params = {"title": title, "city": city, "limit": limit}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("jobs", [])


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

    # --- Tools ---
    read_resume = FileReadTool(file_path=r"C:\Users\Aggy\AIJobtl\Agnes_Nyagami.txt")
    semantic_search_resume = MDXSearchTool(mdx=r"C:\Users\Aggy\AIJobtl\Agnes_Nyagami.txt")
    read_h1b_csv = FileReadTool(file_path=r"C:\Users\Aggy\AIJobtl\H-1B_DATA\H1B_2020_2023_summary.csv")

    indeed_mcp = MCPIndeedClient(os.getenv("MCP_INDEED_URL", "http://localhost:8001"))

    # --- Agents ---
    @agent
    def job_search_agent(self) -> Agent:
        return Agent(config=self.agents_config.get("job_searcher"), tools=[], verbose=False)

    @agent
    def employer_verifier_agent(self) -> Agent:
        # Uses FileReadTool to access CSV directly
        return Agent(config=self.agents_config.get("employer_verifier"),
                     tools=[self.read_h1b_csv],
                     verbose=False)

    @agent
    def resume_reader_agent(self) -> Agent:
        # Uses resume reading and semantic search tools (RAG)
        return Agent(config=self.agents_config.get("resume_reader"),
                     tools=[self.read_resume, self.semantic_search_resume],
                     verbose=False)

    @agent
    def resume_tailor_agent(self) -> Agent:
        # Only focused on writing — not reading
        return Agent(config=self.agents_config.get("resume_tailor"), tools=[], verbose=False)

    # --- Tasks ---
    @task
    def job_search_task(self) -> Task:
        def run(title: str, city: str, limit: int = 8) -> List[JobListing]:
            raw_jobs = self.indeed_mcp.search_jobs(title, city, limit)
            return [
                JobListing(
                    title=j.get("title") or j.get("job_title", ""),
                    company=j.get("company") or j.get("employer", ""),
                    location=j.get("location", city),
                    description=j.get("description") or j.get("summary", ""),
                    raw=j
                )
                for j in raw_jobs
            ]
        return Task(config=self.tasks_config.get("job_search_task"), func=run)

    @task
    def employer_verification_task(self) -> Task:
        def run(jobs: List[JobListing]) -> List[VerificationResult]:
            csv_content = self.read_h1b_csv.run()
            df = pd.read_csv(pd.compat.StringIO(csv_content))
            results = []
            for job in jobs:
                matches = df[df["Employer"].str.contains(job.company, case=False, na=False)]
                eligible = not matches.empty
                results.append(VerificationResult(
                    company=job.company,
                    eligible=eligible,
                    records=len(matches),
                    note="Employer found in H-1B dataset" if eligible else "No record found"
                ))
            return results
        return Task(config=self.tasks_config.get("employer_verification_task"), func=run)

    @task
    def resume_reader_task(self) -> Task:
        def run() -> str:
            # Summarize and extract key insights from the resume
            resume_text = self.read_resume.run()
            search_summary = self.semantic_search_resume.run("Summarize key skills and experiences")
            combined_summary = f"Resume Overview:\n{resume_text[:1000]}\n\nKey Highlights:\n{search_summary}"
            return combined_summary
        return Task(config=self.tasks_config.get("resume_reader_task"), func=run)

    @task
    def resume_tailoring_task(self) -> Task:
        def run(job: JobListing, resume_summary: str) -> TailoredResume:
            prompt = f"""
You are a resume tailoring assistant.
Job Title: {job.title}
Company: {job.company}
Job Description:
{job.description}

Candidate profile summary:
{resume_summary}

Write a tailored resume section optimized for this job. 
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
        def run(title: str, city: str) -> Dict[str, Any]:
            jobs = self.job_search_task().run(title=title, city=city, limit=8)
            verifications = self.employer_verification_task().run(jobs=jobs)
            resume_summary = self.resume_reader_task().run()

            results = []
            for job, ver in zip(jobs, verifications):
                if not ver.eligible:
                    results.append({"job": job.dict(), "verified": ver.dict(), "tailored": None})
                    continue
                tailored = self.resume_tailoring_task().run(job=job, resume_summary=resume_summary)
                results.append({"job": job.dict(), "verified": ver.dict(), "tailored": tailored.dict()})
            return {"query": {"title": title, "city": city}, "results": results}
        return Task(config=self.tasks_config.get("workflow_orchestration_task"), func=run)

    # --- Crew ---
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=2,
            memory=True
        )
