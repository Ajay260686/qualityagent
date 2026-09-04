# Gemini Customer Address Enrichment Agent

This folder contains a LangChain-powered agent that reads a customer database CSV (which might contain partial, incorrect, or incomplete address details) and uses **Gemini (`ChatGoogleGenAI`)** combined with **Google Maps and Google Places APIs** to enrich, validate, and standardize the addresses.

## Features
* **Google Places Search:** Searches for businesses or places by name or partial text if the street address is missing or wrong.
* **Google Maps Geocoding:** Standardizes street names, parses addresses into clean fields (street address, city, state, zip), and verifies accuracy.
* **Web Search Fallback:** Falls back to Google Custom Search JSON API if Place Search or Geocoding APIs fail to locate the business/address details.
* **State Standardization:** Converts state names to standard 2-letter postal codes (e.g. `Washington` -> `WA`).
* **Confidence Scoring:** Tags each row with `High`, `Medium`, or `Low` enrichment confidence and logs any changes or API errors in a `notes` column.

## Setup & Execution

### 1. Requirements
Ensure the following libraries are installed in your Python environment:
```bash
pip install langchain langchain-core langchain-google-genai langchain-community pydantic requests python-dotenv
```

### 2. Environment Variables
Your `.env` file must contain a `GOOGLE_API_KEY` with access enabled for the **Geocoding API** and **Places API (New or Current)** in your Google Cloud Console.

For web search fallback, you can also define:
* `GOOGLE_CSE_ID`: Google Custom Search Engine ID (CX)

### 3. Run the Agent
Run the script to read `customers.csv` and write the validated/enriched details to `customers_enriched.csv`:
```bash
python customer_enricher.py
```
