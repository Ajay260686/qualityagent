"""
Data Steward Validation Agent — Gemini Enterprise (Vertex-backed) implementation
==================================================================================

Calls Gemini Enterprise Agent Platform with Grounding with Google Search AND
Grounding with Google Maps enabled simultaneously (both are "search tools",
which Gemini explicitly allows combining in a single request — you just can't
mix a search tool with a non-search tool like function calling).

Prerequisites
-------------
1. pip install --upgrade google-genai

2. Set environment variables (Gemini Enterprise / Vertex auth path):

    export GOOGLE_CLOUD_PROJECT=<your-project-id>
    export GOOGLE_CLOUD_LOCATION=global
    export GOOGLE_GENAI_USE_ENTERPRISE=True

   Authentication itself uses Application Default Credentials — run
   `gcloud auth application-default login` locally, or rely on the
   attached service account when deployed on Cloud Run / GKE / Vertex.

3. Google Maps grounding and Google Search grounding must both be enabled
   for your project/model in Agent Studio (or they're on by default for
   supported Gemini models — check the "Supported models" list in the
   Gemini Enterprise grounding docs before relying on this).

Notes on Maps grounding
------------------------
- Maps grounding is a *textual* location search — it behaves like searching
  Google Maps, not like calling the Places API directly. It's a good fit for
  "is there a business like this near this address" reasoning, but it is not
  a structured Places API replacement. Keep using Places/Address Validation
  for your primary structured lookups; use this agent for the residual
  ambiguous/unverified records.
- Supplying lat/lng biases "near me"-style results. For address *validation*
  of a known business, you can omit lat/lng and just put the full address in
  the prompt text — the model still uses Maps grounding to verify it.
"""

import csv
import json
import os
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv

from google import genai
from google.genai import types
from google.genai.types import (
    GenerateContentConfig,
    GoogleMaps,
    GoogleSearch,
    HttpOptions,
    Tool,
)

# ---------------------------------------------------------------------------
# 1. Data Steward system prompt
# ---------------------------------------------------------------------------
# This is the validation prompt built earlier. Keeping it as a separate
# constant makes it easy to version/update independently of the calling code.

DATA_STEWARD_SYSTEM_PROMPT = """
You are a Data Steward Agent responsible for validating and enriching customer
business records. You verify each record against authoritative external sources
and produce a structured, evidence-backed validation report. You never fabricate,
assume, or infer data that is not directly supported by a retrieved source.

OBJECTIVE
For each customer record provided, validate and, where possible, correct or
complete the following five data fields:
1. EIN (Employer Identification Number)
2. Customer / Business Legal Name (and DBA, if different)
3. Customer Business Address
4. Customer Owner / Principal Information
5. Customer Services (line of business / industry classification)

For every field, output: the validated value, a confidence score (0-100),
the source(s) used, and a status flag (Verified / Conflicting / Not found /
Unverifiable via available sources). Do not silently overwrite the input
record. Report findings for human or downstream-system review.

METHODOLOGY
- Use Google Search and Google Maps grounding to look up the business by
  name, address, and any other provided identifiers.
- Cross-check at least two independent sources before marking a field
  "Verified" where possible (e.g., a state registry listing AND a Google
  Business Profile / Maps listing).
- If sources conflict, report both values and flag "Conflicting — needs
  review" rather than silently picking one, unless one source is clearly
  more current.
- If no source returns a match, mark the field "Not found" and state which
  sources were checked — never fabricate a plausible-sounding value.
- EIN is frequently unverifiable via public search alone — if you cannot
  find it on an official filing or registry page, say so explicitly rather
  than guessing.
- Treat owner/principal information as low-confidence unless a source
  explicitly names them in connection with this specific business — do not
  infer an owner's identity from a name that merely resembles the business
  name.

OUTPUT FORMAT
Return one structured record per customer in this exact format:

Customer Record ID: [input record identifier]
Overall Validation Status: [Fully Verified / Partially Verified / Needs Review / Unverifiable]

Field: EIN
  Input value: [as provided, or "Not provided"]
  Validated value: [value found, or "Not found"]
  Status: [Verified / Conflicting / Not found / Unverifiable via available sources]
  Confidence: [0-100]
  Source(s): [named source(s) / URL]
  Notes: [any caveat]

Field: Customer Name / DBA
  [same structure]

Field: Business Address
  [same structure]

Field: Owner / Principal Information
  [same structure]

Field: Customer Services / Industry Classification
  [same structure]

Recommended Action: [Auto-accept / Route to analyst review / Route to manual research / Insufficient data — hold]

GUARDRAILS
- Never fabricate an EIN, address, or owner name, even when a value would be
  "plausible." Report absence explicitly.
- Never present a single-source, unverified finding as "Verified."
- Flag any business that appears inactive, closed, or unlisted anywhere as a
  material finding, not just a missing-data case.
- If multiple businesses could plausibly match (e.g., a common name), report
  the ambiguity and the candidate matches rather than guessing.
"""


# ---------------------------------------------------------------------------
# 2. Client setup
# ---------------------------------------------------------------------------

def build_client() -> genai.Client:
    """
    Builds a Gemini Enterprise (Vertex-backed) client.
    Requires GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, and
    GOOGLE_GENAI_USE_ENTERPRISE=True to already be set as environment
    variables (see module docstring).
    """
    required_env = ["GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GOOGLE_GENAI_USE_ENTERPRISE","GOOGLE_GENAI_API_KEY"]
    missing = [v for v in required_env if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "See module docstring for setup instructions."
        )
    return genai.Client(api_key=os.environ["GOOGLE_GENAI_API_KEY"])


# ---------------------------------------------------------------------------
# 3. Grounded generation config: Google Search + Google Maps combined
# ---------------------------------------------------------------------------

def build_grounded_config(latitude: float | None = None, longitude: float | None = None) -> GenerateContentConfig:
    """
    Builds a GenerateContentConfig with BOTH Google Search and Google Maps
    grounding tools enabled. Both are "search tools", so combining them in
    one request is supported (unlike mixing a search tool with a custom
    function-calling tool, which Gemini does not currently support).

    latitude/longitude are optional — pass them if you have a known
    business location to bias Maps results locally. For most data-steward
    validation calls you can omit these and rely on the address text itself.
    """
    tools = [
        Tool(google_search=GoogleSearch()),
        Tool(google_maps=GoogleMaps()),
    ]

    tool_config = None
    if latitude is not None and longitude is not None:
        tool_config = types.ToolConfig(
            retrieval_config=types.RetrievalConfig(
                lat_lng=types.LatLng(latitude=latitude, longitude=longitude)
            )
        )

    return GenerateContentConfig(
        system_instruction=DATA_STEWARD_SYSTEM_PROMPT,
        tools=tools,
        tool_config=tool_config,
        temperature=1.0,  # Google's recommended setting for grounded search results
    )


# ---------------------------------------------------------------------------
# 4. Per-record validation call
# ---------------------------------------------------------------------------

@dataclass
class CustomerRecord:
    record_id: str
    business_name: str = ""
    ein: str = ""
    address: str = ""
    city: str = ""
    state: str = ""
    owner_name: str = ""
    latitude: float | None = None
    longitude: float | None = None


def build_record_prompt(record: CustomerRecord) -> str:
    """Formats a single customer record into the user turn sent to Gemini."""
    lines = [f"Validate the following customer record (Record ID: {record.record_id}):"]
    if record.business_name:
        lines.append(f"- Business/Customer Name (as provided): {record.business_name}")
    if record.ein:
        lines.append(f"- EIN (as provided): {record.ein}")
    if record.address or record.city or record.state:
        lines.append(f"- Address (as provided): {record.address}, {record.city}, {record.state}".strip(", "))
    if record.owner_name:
        lines.append(f"- Owner/Principal (as provided): {record.owner_name}")
    lines.append(
        "\nUse Google Search and Google Maps grounding to validate this record "
        "against state business registries, BBB, Google Business listings, and "
        "any other authoritative sources you find. Follow the output format and "
        "guardrails in your system instructions exactly."
    )
    return "\n".join(lines)


def validate_record(client: genai.Client, record: CustomerRecord, model: str = "gemini-2.5-flash") -> dict:
    """
    Sends one customer record to Gemini Enterprise with Search + Maps
    grounding enabled, and returns the raw text response plus grounding
    metadata (search queries used, source URLs) for audit purposes.
    """
    config = build_grounded_config(latitude=record.latitude, longitude=record.longitude)
    prompt = build_record_prompt(record)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    result = {
        "record_id": record.record_id,
        "model_output": response.text,
        "grounding_metadata": None,
    }

    # Grounding metadata (search queries executed, source URIs) supports your
    # audit trail requirement — capture it whenever it's present.
    try:
        candidate = response.candidates[0]
        if candidate.grounding_metadata:
            gm = candidate.grounding_metadata
            result["grounding_metadata"] = {
                "web_search_queries": getattr(gm, "web_search_queries", None),
                "grounding_chunks": [
                    {
                        "title": getattr(chunk.web, "title", None) if getattr(chunk, "web", None) else None,
                        "uri": getattr(chunk.web, "uri", None) if getattr(chunk, "web", None) else None,
                    }
                    for chunk in (gm.grounding_chunks or [])
                ] if getattr(gm, "grounding_chunks", None) else [],
            }
    except (AttributeError, IndexError):
        pass  # No grounding metadata returned — leave as None, don't fail the run

    return result


# ---------------------------------------------------------------------------
# 5. Batch runner over a CSV of customer records
# ---------------------------------------------------------------------------

def load_records_from_csv(csv_path: str) -> list[CustomerRecord]:
    """
    Expects a CSV with (at minimum) a record_id column. Other columns are
    optional and mapped if present: business_name, ein, address, city,
    state, owner_name, latitude, longitude.
    """
    records = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                CustomerRecord(
                    record_id=row.get("record_id", "").strip(),
                    business_name=row.get("business_name", "").strip(),
                    ein=row.get("ein", "").strip(),
                    address=row.get("address", "").strip(),
                    city=row.get("city", "").strip(),
                    state=row.get("state", "").strip(),
                    owner_name=row.get("owner_name", "").strip(),
                    latitude=float(row["latitude"]) if row.get("latitude") else None,
                    longitude=float(row["longitude"]) if row.get("longitude") else None,
                )
            )
    return records


def run_batch(
    input_csv: str,
    output_jsonl: str,
    model: str = "gemini-2.5-flash",
    requests_per_minute: int = 30,
) -> None:
    """
    Validates every record in input_csv and writes one JSON line per record
    to output_jsonl. Simple rate limiting is included — tune
    requests_per_minute to your project's quota.
    """
    client = build_client()
    records = load_records_from_csv(input_csv)
    delay_seconds = 60.0 / requests_per_minute

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for i, record in enumerate(records, start=1):
            print(f"[{i}/{len(records)}] Validating record {record.record_id}...")
            try:
                result = validate_record(client, record, model=model)
            except Exception as exc:  # noqa: BLE001 — log and continue batch
                result = {
                    "record_id": record.record_id,
                    "error": str(exc),
                }
                print(f"  -> ERROR: {exc}")

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            if i < len(records):
                time.sleep(delay_seconds)

    print(f"\nDone. Results written to {output_jsonl}")


# ---------------------------------------------------------------------------
# 6. Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Single-record example
    load_dotenv()
    client = build_client()
    sample = CustomerRecord(
        record_id="CUST-00123",
        business_name="Neighborhood Auto",
        #address="142 Main St",
        city="Cumming",
        state="GA",
    )
    result = validate_record(client, sample)
    print(result["model_output"])
    print("\n--- Grounding metadata ---")
    print(json.dumps(result["grounding_metadata"], indent=2))

    # Batch example (uncomment to run against a CSV):
    #run_batch("customers.csv", "validation_results.jsonl")
