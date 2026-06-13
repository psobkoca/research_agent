from pydantic import BaseModel, Field
from typing import List

class SearchQueries(BaseModel):
    """List of generated search queries."""
    queries: List[str] = Field(
        description="A list of 3 to 5 distinct, targeted search queries designed to find projects the company is working on."
    )

class RelevanceGrade(BaseModel):
    """Relevance grade of research results."""
    relevance_score: int = Field(
        description="A score from 1 to 10 representing how relevant the document is to finding the company's active projects, products, or services. 1 is completely irrelevant, 10 is extremely relevant."
    )
    rationale: str = Field(
        description="Brief explanation of why the search result received this relevance score."
    )

class HallucinationGrade(BaseModel):
    """Grade representing if the generated breakdown contains hallucinations."""
    has_hallucinations: bool = Field(
        description="True if the generated breakdown includes claims, projects, or details NOT supported by the source search results. False if all details are fully supported."
    )
    explanation: str = Field(
        description="Detailed explanation of what claims are unsupported (if any) and how to correct them, or why it is fully grounded."
    )

class Project(BaseModel):
    """An active project of the company."""
    name: str = Field(description="Name of the project or product.")
    description: str = Field(description="Description of the project's purpose, scope, and technologies used.")
    citations: List[str] = Field(description="List of exact URLs from the search results that support this project.")


class ServiceOffer(BaseModel):
    """Proposed service we could offer the company."""
    service_name: str = Field(description="Name/title of the proposed service or solution.")
    rationale: str = Field(description="Why this service would be valuable to the company, referencing specific projects.")
    target_projects: List[str] = Field(description="List of project names this service targets.")

class CompanyAnalysis(BaseModel):
    """Final output breakdown of the company analysis."""
    company_name: str = Field(description="Name of the company analyzed.")
    summary: str = Field(description="A high-level summary of the company's project landscape.")
    projects: List[Project] = Field(description="Active projects currently worked on by the company.")
    recommended_services: List[ServiceOffer] = Field(description="Tailored services we can offer to help them with their projects.")

class MockSearchResult(BaseModel):
    """A mock search result page."""
    title: str = Field(description="Title of the search result page.")
    url: str = Field(description="URL of the search result page.")
    content: str = Field(description="Snippet content of the search result page.")

class MockSearchResponse(BaseModel):
    """A list of mock search results."""
    results: List[MockSearchResult] = Field(description="List of search result items.")

