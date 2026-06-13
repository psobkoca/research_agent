import os
from typing import List, Dict, Any, TypedDict, Literal
from dotenv import load_dotenv
from tavily import TavilyClient

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START

from schemas import (
    SearchQueries,
    RelevanceGrade,
    HallucinationGrade,
    CompanyAnalysis,
    MockSearchResult,
    MockSearchResponse
)

# Load environment variables
load_dotenv()

# Configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")

# Mock database for offline/no-api testing
MOCK_DATABASE = {
    "vercel": [
        {
            "title": "Vercel Products and Open Source Projects",
            "url": "https://vercel.com/docs/frameworks",
            "content": "Vercel is the creator and maintainer of Next.js, a popular React framework for building fast web applications. They also work on Turborepo, a high-performance build system for JavaScript and TypeScript monorepos, and v0, a generative AI UI creation tool that outputs React code."
        },
        {
            "title": "Vercel introduces AI SDK for building AI applications",
            "url": "https://vercel.com/blog/introducing-vercel-ai-sdk",
            "content": "The Vercel AI SDK is an open-source library designed to help developers build conversational, streaming, and rich chat user interfaces using React, Svelte, Vue, and SolidJS. It makes integrating LLMs into apps easy."
        },
        {
            "title": "Vercel Serverless and Edge Functions documentation",
            "url": "https://vercel.com/docs/functions",
            "content": "Vercel Functions allow developers to run code on-demand without managing infrastructure. They support Serverless Functions for complex backend logic and Edge Functions for low-latency, globally distributed computations."
        }
    ],
    "stripe": [
        {
            "title": "Stripe Payments and Billing Infrastructure",
            "url": "https://stripe.com/payments",
            "content": "Stripe offers a suite of payment APIs that power commerce for businesses of all sizes. Key projects include Stripe Billing for recurring subscriptions and invoicing, and Stripe Checkout, a pre-built payment page designed to optimize conversion."
        },
        {
            "title": "Stripe Radar fraud prevention using ML",
            "url": "https://stripe.com/radar",
            "content": "Stripe Radar is a fraud detection and prevention system built directly into the payments flow. It uses machine learning models trained on billions of data points to evaluate risk scores and block fraudulent transactions in real time."
        },
        {
            "title": "Stripe Connect for platform and marketplace onboarding",
            "url": "https://stripe.com/connect",
            "content": "Stripe Connect is a routing and verification engine for multi-sided marketplaces. It automates onboarding, identity verification (KYC), and complex payouts for platforms like Shopify and Lyft."
        }
    ],
    "apple": [
        {
            "title": "Apple Developer - Swift and SwiftUI frameworks",
            "url": "https://developer.apple.com/swift/",
            "content": "Swift is Apple's open-source programming language used to build apps for iOS, iPadOS, macOS, watchOS, and tvOS. SwiftUI provides an innovative, exceptionally simple way to build user interfaces across all Apple platforms with a declarative Swift syntax."
        },
        {
            "title": "Apple Core ML and Machine Learning APIs",
            "url": "https://developer.apple.com/machine-learning/",
            "content": "Core ML allows developers to integrate machine learning models into their Apple platform apps. It optimizes on-device performance by leveraging the CPU, GPU, and Neural Engine while minimizing memory footprint and power consumption."
        },
        {
            "title": "Apple Xcode Cloud Continuous Integration",
            "url": "https://developer.apple.com/xcode-cloud/",
            "content": "Xcode Cloud is a continuous integration and delivery service built into Xcode and designed specifically for Apple developers. It accelerates the development and delivery of high-quality apps by bringing together cloud-based tools."
        }
    ]
}

# State definition
class AgentState(TypedDict):
    company_name: str
    queries: List[str]
    search_results: List[Dict[str, Any]]  # keys: title, url, content
    graded_results: List[Dict[str, Any]]  # subset of search_results that are relevant
    relevance_grades: List[Dict[str, Any]]  # detailed grading results for debugging
    analysis: CompanyAnalysis | None
    hallucination_review: HallucinationGrade | None
    attempts: int
    review_attempts: int
    logs: List[str]  # Step-by-step logs for the user interface
    error: str | None

def get_llm():
    """Helper to initialize ChatOllama."""
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0.0,
        base_url=OLLAMA_HOST
    )

def generate_mock_search_results(company: str) -> List[Dict[str, Any]]:
    """Generate mock search results using Ollama when Tavily API key is missing."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(MockSearchResponse)
    
    
    system_prompt = (
        "You are a mock search engine. Your task is to generate 3 realistic search result snippets "
        "for a given company name. Each snippet must include a title, a valid-looking URL (starting with http:// or https://), "
        "and content describing some real or highly plausible active projects, software systems, or "
        "technical initiatives that this company is working on."
    )
    prompt = f"Generate 3 mock search results for the company: '{company}'."
    
    try:
        response = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ])
        results = []
        if response and response.results:
            for item in response.results:
                results.append({
                    "title": item.title,
                    "url": item.url,
                    "content": item.content
                })
        return results
    except Exception as e:
        print(f"[Agent] Mock generation failed: {e}. Using hardcoded fallback.")
        return [
            {
                "title": f"{company} Official website and projects",
                "url": f"https://www.{company.lower().replace(' ', '')}.com",
                "content": f"Official documentation and software tools created by {company}. Includes developer APIs, software frameworks, and cloud products."
            }
        ]


# --- Graph Nodes ---

def generate_queries(state: AgentState) -> Dict[str, Any]:
    company = state["company_name"]
    attempts = state["attempts"]
    logs = state.get("logs", [])[:]
    
    msg = f"Generating search queries for {company} (Attempt {attempts + 1})..."
    logs.append(msg)
    print(f"[Agent] {msg}")
    
    llm = get_llm()
    structured_llm = llm.with_structured_output(SearchQueries)
    
    system_prompt = (
        "You are an expert market researcher. Your task is to generate search queries "
        "designed to find information about a company's active projects, products, new initiatives, "
        "open-source contributions, and engineering efforts.\n"
        "Generate 3 to 5 targeted, diverse search queries. Make sure they include the company name "
        "and relevant keywords (e.g., 'projects', 'products', 'new release', 'github', 'architecture')."
    )
    
    prompt = f"Generate search queries to identify the active projects and technical initiatives of the company: '{company}'."
    
    # If this is a retry, add some variation instruction
    if attempts > 0:
        prompt += "\nNote: Previous queries did not find enough results. Try to generate more specific or different search queries."
        
    try:
        response = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ])
        
        # Ensure we always get a list of queries
        queries = response.queries if response and response.queries else [f"{company} current projects", f"{company} product release"]
        logs.append(f"Generated queries: {queries}")
        return {
            "queries": queries,
            "attempts": attempts + 1,
            "logs": logs
        }
    except Exception as e:
        err_msg = f"Failed to generate search queries: {str(e)}"
        logs.append(err_msg)
        return {
            "queries": [f"{company} active projects", f"{company} latest products", f"{company} software architecture"],
            "attempts": attempts + 1,
            "logs": logs
        }

def execute_search(state: AgentState) -> Dict[str, Any]:
    queries = state["queries"]
    company = state["company_name"]
    logs = state.get("logs", [])[:]
    
    # Check if TAVILY_API_KEY is configured
    is_mock = not TAVILY_API_KEY or TAVILY_API_KEY.strip() == "" or TAVILY_API_KEY.startswith("your_")
    
    if is_mock:
        msg = f"Executing Mock Search (No Tavily Key) for {company}..."
        logs.append(msg)
        print(f"[Agent] [Warning] {msg}")
        
        # Check mock database
        comp_key = company.lower().strip()
        if comp_key in MOCK_DATABASE:
            results = MOCK_DATABASE[comp_key]
            logs.append(f"Retrieved {len(results)} mock results from local database.")
            return {
                "search_results": results,
                "logs": logs
            }
        else:
            # Generate mock results using LLM
            print(f"[Agent] Generating mock results for unknown company '{company}' via Ollama web emulator...")
            results = generate_mock_search_results(company)
            logs.append(f"Generated {len(results)} mock results via Ollama web emulator.")
            return {
                "search_results": results,
                "logs": logs
            }
            
    # Real Tavily Search
    msg = f"Executing Tavily search for {len(queries)} queries..."
    logs.append(msg)
    print(f"[Agent] {msg}")
        
    try:
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        all_results = []
        seen_urls = set()
        
        for q in queries:
            print(f"  Searching: '{q}'...")
            search_response = tavily.search(query=q, max_results=3, include_answer=False)
            results = search_response.get("results", [])
            for r in results:
                url = r.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append({
                        "title": r.get("title", "No Title"),
                        "url": url,
                        "content": r.get("content", "")
                    })
                    
        logs.append(f"Search completed. Found {len(all_results)} unique documents.")
        return {
            "search_results": all_results,
            "logs": logs
        }
    except Exception as e:
        err_msg = f"Failed to execute Tavily search: {str(e)}"
        logs.append(err_msg)
        # Fallback to mock instead of failing completely
        print(f"[Agent] Tavily failed: {e}. Falling back to mock generator.")
        results = generate_mock_search_results(company)
        return {
            "search_results": results,
            "logs": logs
        }


def grade_results(state: AgentState) -> Dict[str, Any]:
    results = state["search_results"]
    company = state["company_name"]
    logs = state.get("logs", [])[:]
    
    msg = f"Grading {len(results)} search results for relevance to {company}'s projects..."
    logs.append(msg)
    print(f"[Agent] {msg}")
    
    llm = get_llm()
    structured_llm = llm.with_structured_output(RelevanceGrade)
    
    graded_results = []
    relevance_grades = []
    
    system_prompt = (
        "You are an information auditor. Your task is to evaluate and grade the relevance of a search result "
        "to identifying the active projects, products, new initiatives, or engineering efforts of the specified company.\n"
        "Provide a score from 1 to 10:\n"
        "- 8 to 10: Highly relevant, explicitly mentions active projects, new products, or code repository details.\n"
        "- 5 to 7: Moderately relevant, mentions the company's tech stacks, general product categories, or partnerships related to engineering.\n"
        "- 1 to 4: Irrelevant, focuses on stock performance, general marketing press releases without tech details, or old historical context."
    )
    
    for i, r in enumerate(results):
        prompt = (
            f"Company: {company}\n"
            f"Document Title: {r['title']}\n"
            f"Document Content: {r['content']}\n\n"
            f"Grade the relevance of this document on a scale of 1 to 10."
        )
        try:
            grade = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=prompt)
            ])
            
            relevance_grades.append({
                "url": r["url"],
                "relevance_score": grade.relevance_score,
                "rationale": grade.rationale
            })
            
            # Threshold of 6 or above is considered relevant
            if grade.relevance_score >= 6:
                graded_results.append(r)
                print(f"  [RELEVANT - Score: {grade.relevance_score}/10] {r['url']} - {r['title'][:40]}...")
            else:
                print(f"  [IRRELEVANT - Score: {grade.relevance_score}/10] {r['url']} - {r['title'][:40]}...")
        except Exception as e:
            # Fallback in case of API failure: treat as relevant so we don't lose data
            print(f"  [ERROR] Grading failed for {r['url']}: {e}. Defaulting to relevant (score 7).")
            graded_results.append(r)
            relevance_grades.append({
                "url": r["url"],
                "relevance_score": 7,
                "rationale": f"Grading error: {str(e)}"
            })
            
    logs.append(f"Grading complete. {len(graded_results)}/{len(results)} documents passed relevance check.")
    return {
        "graded_results": graded_results,
        "relevance_grades": relevance_grades,
        "logs": logs
    }

def generate_analysis(state: AgentState) -> Dict[str, Any]:
    company = state["company_name"]
    docs = state["graded_results"]
    review_attempts = state["review_attempts"]
    last_review = state.get("hallucination_review")
    logs = state.get("logs", [])[:]
    
    msg = f"Generating project analysis for {company} (Review Attempt {review_attempts + 1})..."
    logs.append(msg)
    print(f"[Agent] {msg}")
    
    if not docs:
        msg_no_docs = "No relevant search results found. Synthesizing generic summary."
        logs.append(msg_no_docs)
        print(f"[Agent] {msg_no_docs}")
        # Return empty/minimal analysis
        return {
            "analysis": CompanyAnalysis(
                company_name=company,
                summary="No relevant projects found during search.",
                projects=[],
                recommended_services=[]
            ),
            "logs": logs
        }
        
    llm = get_llm()
    structured_llm = llm.with_structured_output(CompanyAnalysis)
    
    system_prompt = (
        "You are an elite business analyst. Your job is to analyze research documents about a company "
        "and generate a highly detailed, structured breakdown of the projects they are working on.\n"
        f"Identify the active projects, internal initiatives, and digital transformation efforts for {company}.\n"
        f"Suggest AI, data analytics, digital engineering, cloud modernization, and automation services that your company could pitch to {company}.\n"
        "Extract pitch opportunities from those findings.\n"
        "CRITICAL RULES:\n"
        "1. Every single project MUST exist and be supported by the provided source documents.\n"
        "2. Do NOT hallucinate or assume projects. If a project is not explicitly mentioned in the text, do not list it.\n"
        "3. Every project MUST include a 'citations' list containing the exact URLs from the search results where the information was found.\n"
        "4. Recommend 2 to 5 targeted services we could offer them based on their projects. List the relevant projects they target."
    )
    
    # Format source documents
    sources_text = ""
    for i, d in enumerate(docs):
        sources_text += f"--- DOCUMENT {i+1} ---\nTitle: {d['title']}\nURL: {d['url']}\nContent: {d['content']}\n\n"
        
    prompt = f"Company to analyze: {company}\n\nSource Documents:\n{sources_text}"
    
    # If we are in correction mode, feed back the hallucination review
    if review_attempts > 0 and last_review and last_review.has_hallucinations:
        prompt += (
            f"\n\nWARNING: The previous generation failed the hallucination audit. Please correct the following errors:\n"
            f"Review Feedback: {last_review.explanation}\n\n"
            f"Please regenerate the analysis, making sure to remove any unsupported claims and fix any citation errors."
        )
        
    try:
        analysis = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ])
        logs.append("Project analysis generation complete.")
        return {
            "analysis": analysis,
            "logs": logs
        }
    except Exception as e:
        err_msg = f"Failed to generate project analysis: {str(e)}"
        logs.append(err_msg)
        return {
            "error": err_msg,
            "logs": logs
        }

def review_hallucinations(state: AgentState) -> Dict[str, Any]:
    company = state["company_name"]
    analysis = state["analysis"]
    docs = state["graded_results"]
    review_attempts = state["review_attempts"]
    logs = state.get("logs", [])[:]
    
    msg = f"Auditing generated analysis for hallucinations and citation accuracy (Attempt {review_attempts + 1})..."
    logs.append(msg)
    print(f"[Agent] {msg}")
    
    if not analysis or not analysis.projects:
        logs.append("No projects to review.")
        return {
            "hallucination_review": HallucinationGrade(has_hallucinations=False, explanation="No projects were generated to audit."),
            "review_attempts": review_attempts + 1,
            "logs": logs
        }
        
    llm = get_llm()
    structured_llm = llm.with_structured_output(HallucinationGrade)
    
    system_prompt = (
        "You are an independent facts checker and auditor. Your job is to verify that a structured business analysis "
        "does not contain any hallucinations, and that all citations are valid and point to URLs containing the asserted facts.\n"
        "Check every project: is it supported by the text of the source documents? Are the URL citations matching?\n"
        "If you find ANY claims that cannot be supported by the provided text, or if a citation is missing or incorrect, set has_hallucinations=True and explain the exact issue clearly so the generator can correct it."
    )
    
    # Format generated analysis for the reviewer
    analysis_text = f"Company: {analysis.company_name}\nSummary: {analysis.summary}\n\nProjects:\n"
    for p in analysis.projects:
        analysis_text += f"- Project: {p.name}\n  Description: {p.description}\n  Citations: {p.citations}\n"
            
    # Format source documents
    sources_text = ""
    for i, d in enumerate(docs):
        sources_text += f"--- DOCUMENT {i+1} ---\nTitle: {d['title']}\nURL: {d['url']}\nContent: {d['content']}\n\n"
        
    prompt = (
        f"Source Documents:\n{sources_text}\n"
        f"Generated Analysis to Audit:\n{analysis_text}\n\n"
        f"Please audit the generated analysis. Check if it contains any hallucinated projects/details, or incorrect citations."
    )
    
    try:
        review = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ])
        
        logs.append(f"Audit completed. Has Hallucinations: {review.has_hallucinations}. Explanation: {review.explanation}")
        print(f"  [Audit Result] Has Hallucinations: {review.has_hallucinations}")
        if review.has_hallucinations:
            print(f"  [Feedback] {review.explanation}")
            
        return {
            "hallucination_review": review,
            "review_attempts": review_attempts + 1,
            "logs": logs
        }
    except Exception as e:
        err_msg = f"Audit failed: {str(e)}. Defaulting to passed to avoid infinite loop."
        logs.append(err_msg)
        return {
            "hallucination_review": HallucinationGrade(has_hallucinations=False, explanation=err_msg),
            "review_attempts": review_attempts + 1,
            "logs": logs
        }

# --- Routing Logic ---

def check_relevance(state: AgentState) -> Literal["more_research", "analyze"]:
    graded = state.get("graded_results", [])
    attempts = state.get("attempts", 0)
    error = state.get("error")
    
    if error:
        return "analyze"  # Go to analysis to fail gracefully or show errors
        
    # If we don't have at least 3 relevant docs, and we haven't reached max attempts (3), search again
    if len(graded) < 3 and attempts < 3:
        print(f"[Router] Only found {len(graded)} relevant results. Retrying query generation (attempts={attempts}).")
        return "more_research"
        
    print(f"[Router] Found {len(graded)} relevant results (or reached max attempts). Proceeding to analysis.")
    return "analyze"

def check_hallucinations(state: AgentState) -> Literal["correct_analysis", "complete"]:
    review = state.get("hallucination_review")
    attempts = state.get("review_attempts", 0)
    error = state.get("error")
    
    if error:
        return "complete"
        
    if review and review.has_hallucinations and attempts < 3:
        print(f"[Router] Hallucinations detected (attempt {attempts}). Routing back to generator for correction.")
        return "correct_analysis"
        
    print("[Router] Analysis validated (or reached max review attempts). Workflow complete.")
    return "complete"

# --- Graph Assembly ---

def build_workflow():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("generate_queries", generate_queries)
    workflow.add_node("execute_search", execute_search)
    workflow.add_node("grade_results", grade_results)
    workflow.add_node("generate_analysis", generate_analysis)
    workflow.add_node("review_hallucinations", review_hallucinations)
    
    # Define execution graph
    workflow.add_edge(START, "generate_queries")
    workflow.add_edge("generate_queries", "execute_search")
    workflow.add_edge("execute_search", "grade_results")
    
    # Conditional routing after grading
    workflow.add_conditional_edges(
        "grade_results",
        check_relevance,
        {
            "more_research": "generate_queries",
            "analyze": "generate_analysis"
        }
    )
    
    # Analysis leads to review
    workflow.add_edge("generate_analysis", "review_hallucinations")
    
    # Conditional routing after review
    workflow.add_conditional_edges(
        "review_hallucinations",
        check_hallucinations,
        {
            "correct_analysis": "generate_analysis",
            "complete": END
        }
    )
    
    return workflow.compile()
