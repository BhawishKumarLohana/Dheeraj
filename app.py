#!/usr/bin/env python3
"""
MarketMind Backend API - FastAPI server for the MarketMind analysis
"""

import os
import base64
import io
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from contextlib import asynccontextmanager
import asyncio

# Import from the main script
import feedparser
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
import numpy as np
import torch
from torch.nn.functional import softmax
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for server
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# --- CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCEO9DNYkhNiFtuQhbQJYg8cUy3eaXKcfo")

CANDIDATE_FEEDS = [
    ("https://www.dawn.com/business/rss", "Dawn - Business (rss)"),
    ("https://www.brecorder.com/feeds/latest-news", "Business Recorder - Latest (rss)"),
    ("https://tribune.com.pk/business", "Express Tribune - Business (section)"),
    ("https://www.thenews.com.pk/business", "The News - Business (section)"),
    ("https://profit.pakistantoday.com.pk/news-feed/", "Profit - News-feed (html)"),
    ("https://profit.pakistantoday.com.pk/feed/", "Profit - Feed (try)"),
    ("https://www.geo.tv/category/business", "Geo - Business (section)"),
    ("https://dailytimes.com.pk/business/feed/", "DailyTimes - business feed (try)"),
    ("https://www.nation.com.pk/business", "The Nation - business section"),
]

# Global model storage
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    print("[*] Loading models...")
    
    if not GEMINI_API_KEY or GEMINI_API_KEY == "PASTE_YOUR_API_KEY_HERE":
        print("[!] Warning: Gemini API key not set properly")
    else:
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
        genai.configure(api_key=GEMINI_API_KEY)
    
    models['vader'] = SentimentIntensityAnalyzer()
    models['finbert_tokenizer'] = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    models['finbert_model'] = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    models['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    models['finbert_model'].to(models['device'])
    models['llm'] = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.2)
    
    print(f"[OK] Models loaded. Device: {models['device']}")
    yield
    print("[*] Shutting down...")

app = FastAPI(
    title="MarketMind API",
    description="Pakistani Business News Sentiment Analysis",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HELPER FUNCTIONS ---
def try_parse_feed(url):
    fp = feedparser.parse(url)
    return fp.entries if hasattr(fp, 'entries') else []

def scrape_section_for_links(section_url, max_links=30):
    try:
        r = requests.get(section_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            text = a.get_text(strip=True)
            if not text:
                continue
            if any(kw in href for kw in ['/news', '/business', '/202', '/story']):
                full_link = urljoin(section_url, href)
                links.append({'title': text, 'link': full_link})
            if len(links) >= max_links:
                break
        return links
    except Exception:
        return []

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:()'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_finbert_probs(texts, model, tokenizer, device, batch_size=8):
    pos_probs, neg_probs = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            probs = softmax(outputs.logits, dim=-1).cpu().numpy()
            pos_probs.extend(probs[:, 0])
            neg_probs.extend(probs[:, 1])
    return np.array(pos_probs), np.array(neg_probs)

# --- LANGGRAPH STATE & NODES ---
class MarketMindState(TypedDict):
    articles: pd.DataFrame
    analyzed_articles: pd.DataFrame
    analysis_context: str
    final_summary: str
    vader_model: any
    finbert_model: any
    finbert_tokenizer: any
    device: str
    llm: any

def fetch_and_clean_data(state: MarketMindState) -> MarketMindState:
    collected_articles = []
    
    for feed_url, name in CANDIDATE_FEEDS:
        entries = try_parse_feed(feed_url)
        
        if entries:
            for e in entries:
                published_time = e.get('published_parsed') or e.get('updated_parsed')
                published_iso = datetime(*published_time[:6]).isoformat() if published_time else ''
                collected_articles.append({
                    'source': name,
                    'title': e.get('title', '').strip(),
                    'link': e.get('link', '').strip(),
                    'published': published_iso,
                    'summary': (e.get('summary') or e.get('description') or '')[:500]
                })
        elif not any(feed_url.lower().endswith(ext) for ext in ['.xml', '.rss', '/rss/', '/feeds/']):
            links = scrape_section_for_links(feed_url, max_links=40)
            if links:
                for item in links:
                    collected_articles.append({
                        'source': f"{name} (scraped)",
                        'title': item.get('title', ''),
                        'link': item.get('link', ''),
                        'published': '',
                        'summary': ''
                    })
    
    df = pd.DataFrame(collected_articles)
    df['clean_summary'] = df['summary'].apply(clean_text)
    df['clean_title'] = df['title'].apply(clean_text)
    df = df.drop_duplicates(subset=['clean_title'])
    df = df[df['clean_summary'].str.len() > 30].reset_index(drop=True)
    
    return {"articles": df}

def analyze_sentiment(state: MarketMindState) -> MarketMindState:
    df = state['articles'].copy()
    
    if df.empty:
        return {"analyzed_articles": df}
    
    vader = state['vader_model']
    finbert_model = state['finbert_model']
    finbert_tokenizer = state['finbert_tokenizer']
    device = state['device']
    
    df['vader_compound'] = df['clean_summary'].apply(lambda t: vader.polarity_scores(t)['compound'])
    texts = df['clean_summary'].tolist()
    pos_arr, neg_arr = compute_finbert_probs(texts, finbert_model, finbert_tokenizer, device)
    df['finbert_pos_prob'] = pos_arr
    df['finbert_neg_prob'] = neg_arr
    df['finbert_score'] = df['finbert_pos_prob'] - df['finbert_neg_prob']
    df['combined_score'] = (0.4 * df['vader_compound']) + (0.6 * df['finbert_score'])
    
    def score_to_label(score, pos_thresh=0.05, neg_thresh=-0.05):
        if score > pos_thresh:
            return 'Positive'
        elif score < neg_thresh:
            return 'Negative'
        else:
            return 'Neutral'
    
    df['final_sentiment'] = df['combined_score'].apply(score_to_label)
    return {"analyzed_articles": df}

def prepare_llm_context(state: MarketMindState) -> MarketMindState:
    df = state['analyzed_articles']
    
    if df.empty:
        return {"analysis_context": "No articles found for analysis."}
    
    top_5_pos = df.sort_values('combined_score', ascending=False).head(5)
    top_5_neg = df.sort_values('combined_score', ascending=True).head(5)
    
    positive_news_str = "\n".join([
        f"  - (Score: {row['combined_score']:.2f}) Title: {row['clean_title']}\n    Summary: {row['clean_summary']}"
        for _, row in top_5_pos.iterrows()
    ])
    
    negative_news_str = "\n".join([
        f"  - (Score: {row['combined_score']:.2f}) Title: {row['clean_title']}\n    Summary: {row['clean_summary']}"
        for _, row in top_5_neg.iterrows()
    ])
    
    context = f"--- MOST POSITIVE NEWS ---\n{positive_news_str}\n\n--- MOST NEGATIVE NEWS ---\n{negative_news_str}"
    return {"analysis_context": context}

def generate_summary(state: MarketMindState) -> MarketMindState:
    if state['analyzed_articles'].empty:
        return {"final_summary": "No articles were found to generate a summary."}
    
    llm = state['llm']
    context = state['analysis_context']
    
    prompt = f"""You are a professional financial analyst for the Pakistani market. Your task is to provide a concise, balanced summary of today's key business and economic news based on the provided articles.
Analyze the sentiment and themes from both the positive and negative sets to synthesize a brief market outlook (3-4 paragraphs).

Your response must follow this format:
1.  **Market Sentiment:** Start with an overall market sentiment (e.g., "Overall sentiment appears mixed...").
2.  **Key Drivers:** Identify the most significant positive and negative stories and their potential impact.
3.  **PSX Outlook:** Conclude with a potential outlook for the next trading day for the Pakistan Stock Exchange (PSX).
4.  **Key Words:** After the summary, add a 'Key Words:' section with 5-7 important keywords from the news (e.g., PSX, Inflation, SBP, Tech Sector, Forex, etc.).

Maintain a professional, analytical tone. Do not just list the articles.
{context}"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content
        return {"final_summary": summary}
    except Exception as e:
        return {"final_summary": f"Error generating summary: {e}"}

def create_sentiment_chart(df):
    """Create sentiment chart and return as base64"""
    if df.empty:
        return None
    
    plt.figure(figsize=(8, 5))
    counts = df['final_sentiment'].value_counts()
    colors = {'Positive': '#10B981', 'Negative': '#EF4444', 'Neutral': '#F59E0B'}
    bar_colors = [colors.get(x, '#6B7280') for x in counts.index]
    
    bars = plt.bar(counts.index, counts.values, color=bar_colors, edgecolor='white', linewidth=2)
    
    for bar, count in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 str(count), ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.title("Article Sentiment Distribution", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Article Count", fontsize=12)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close()
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- API ENDPOINTS ---
class AnalysisResponse(BaseModel):
    success: bool
    summary: str
    chart_base64: str | None
    article_count: int
    sentiment_counts: dict
    timestamp: str
    top_positive: list
    top_negative: list

@app.get("/")
async def root():
    return {"message": "MarketMind API is running!", "status": "ready"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": bool(models)}

@app.post("/analyze", response_model=AnalysisResponse)
async def run_analysis():
    """Run the full MarketMind analysis pipeline"""
    
    if not models:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    try:
        # Build workflow
        workflow = StateGraph(MarketMindState)
        workflow.add_node("fetch_and_clean_data", fetch_and_clean_data)
        workflow.add_node("analyze_sentiment", analyze_sentiment)
        workflow.add_node("prepare_llm_context", prepare_llm_context)
        workflow.add_node("generate_summary", generate_summary)
        
        workflow.set_entry_point("fetch_and_clean_data")
        workflow.add_edge("fetch_and_clean_data", "analyze_sentiment")
        workflow.add_edge("analyze_sentiment", "prepare_llm_context")
        workflow.add_edge("prepare_llm_context", "generate_summary")
        workflow.add_edge("generate_summary", END)
        
        compiled_app = workflow.compile()
        
        # Execute
        initial_state = {
            "vader_model": models['vader'],
            "finbert_model": models['finbert_model'],
            "finbert_tokenizer": models['finbert_tokenizer'],
            "device": models['device'],
            "llm": models['llm']
        }
        
        # Run in thread pool to not block
        final_state = await asyncio.to_thread(compiled_app.invoke, initial_state)
        
        df = final_state['analyzed_articles']
        
        # Generate chart
        chart_b64 = await asyncio.to_thread(create_sentiment_chart, df)
        
        # Prepare response
        sentiment_counts = df['final_sentiment'].value_counts().to_dict() if not df.empty else {}
        
        top_positive = []
        top_negative = []
        
        if not df.empty:
            top_pos_df = df.sort_values('combined_score', ascending=False).head(5)
            top_neg_df = df.sort_values('combined_score', ascending=True).head(5)
            
            for _, row in top_pos_df.iterrows():
                top_positive.append({
                    'title': row['clean_title'],
                    'source': row['source'],
                    'score': round(row['combined_score'], 3),
                    'link': row['link']
                })
            
            for _, row in top_neg_df.iterrows():
                top_negative.append({
                    'title': row['clean_title'],
                    'source': row['source'],
                    'score': round(row['combined_score'], 3),
                    'link': row['link']
                })
        
        return AnalysisResponse(
            success=True,
            summary=final_state['final_summary'],
            chart_base64=chart_b64,
            article_count=len(df),
            sentiment_counts=sentiment_counts,
            timestamp=datetime.now().isoformat(),
            top_positive=top_positive,
            top_negative=top_negative
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

