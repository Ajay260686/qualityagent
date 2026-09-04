"""
test.py — Data Steward Agent with Prompt Caching
==================================================

Prompt Caching saves cost and reduces latency by uploading the large,
repeated system prompt ONCE and reusing its cached version across all
per-record generation calls.

Cost savings explanation
------------------------
Without caching:
  Every call to generate_content sends the full DATA_STEWARD_SYSTEM_PROMPT
  (≈700 tokens) PLUS the per-record user message. You pay input-token pricing
  for those 700 tokens on EVERY request.

With caching:
  The system prompt is uploaded once and stored server-side for a TTL window
  (default 60 minutes, configurable). Each subsequent call sends only:
    • A tiny cache reference (not the full prompt text)
    • The per-record user message
  The cached tokens are billed at the (much cheaper) cache-read rate.
  For a 6-record batch the savings are modest, but for hundreds/thousands of
  records per day the savings compound quickly.

Supported models (as of mid-2025)
-----------------------------------
  gemini-2.5-flash, gemini-2.5-pro, gemini-1.5-flash, gemini-1.5-pro

  Note: Context caching is not available on gemini-2.0-flash or the free-tier
  Gemini API (it requires a paid/billing-enabled project even for AI Studio).
  See: https://ai.google.dev/gemini-api/docs/caching
"""

import os
from google import genai
from google.genai import types
from google.genai.types import (
    Content,
    CreateCachedContentConfig,
    GenerateContentConfig,
    GoogleMaps,
    GoogleSearch,
    Part,
    Tool,
)
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------
MODEL = "gemini-2.5-flash"   # Must support context caching
CACHE_TTL_SECONDS = 3600     # How long the server keeps the cached prompt (1 hr)

# ---------------------------------------------------------------------------
# 1. Build client via API key (no Vertex / ADC needed)
# ---------------------------------------------------------------------------
required_env = ["GOOGLE_GENAI_API_KEY"]
missing = [v for v in required_env if not os.environ.get(v)]
if missing:
    raise EnvironmentError(f"Missing required environment variable(s): {', '.join(missing)}")

client = genai.Client(api_key=os.environ["GOOGLE_GENAI_API_KEY"])

# ---------------------------------------------------------------------------
# 2. The large, repeated system prompt — this is what we cache
# ---------------------------------------------------------------------------
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
  "Verified" where possible.
- If sources conflict, report both values and flag "Conflicting — needs
  review" rather than silently picking one.
- If no source returns a match, mark the field "Not found" — never fabricate.
- EIN is frequently unverifiable via public search alone — if you cannot
  find it on an official filing or registry page, say so explicitly.
- Treat owner/principal information as low-confidence unless a source
  explicitly names them in connection with this specific business.

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
# 3. Create a cached content object for the system prompt
#    The minimum cacheable token count is 1,024 tokens.
#    DATA_STEWARD_SYSTEM_PROMPT is ~700 tokens so we add a filler section
#    OR simply combine it with the shared tool definitions that every call
#    would otherwise re-send (shown here as inline content).
# ---------------------------------------------------------------------------
print("Creating cached content for system prompt...")
cached_content = client.caches.create(
    model=MODEL,
    config=CreateCachedContentConfig(
        system_instruction=DATA_STEWARD_SYSTEM_PROMPT,
        # Include the tool definitions in the cache too — these are also
        # identical across every record and add to the repeated token count.
        tools=[
            Tool(google_search=GoogleSearch()),
            Tool(google_maps=GoogleMaps()),
        ],
        ttl=f"{CACHE_TTL_SECONDS}s",
        display_name="data_steward_system_prompt_v1",   # human-readable label
    ),
)

print(f"Cache created: {cached_content.name}")
print(f"Cached token count: {cached_content.usage_metadata.total_token_count}")
print(f"Expires at: {cached_content.expire_time}")

# ---------------------------------------------------------------------------
# 4. Use the cache in generation calls
#    Pass cached_content=cached_content.name into GenerateContentConfig.
#    The system_instruction and tools no longer need to be re-sent.
# ---------------------------------------------------------------------------
def validate_record_cached(record_prompt: str) -> str:
    """
    Generates a validation report for one record using the cached system prompt.
    Only the per-record user message is sent as new tokens.
    """
    response = client.models.generate_content(
        model=MODEL,
        contents=record_prompt,
        config=GenerateContentConfig(
            # Reference the pre-cached system prompt + tool definitions
            cached_content=cached_content.name,
            temperature=1.0,
        ),
    )
    return response.text


# ---------------------------------------------------------------------------
# 5. Demo — validate a sample record using the cached prompt
# ---------------------------------------------------------------------------
sample_prompt = """
Validate the following customer record (Record ID: CUST-00123):
- Business/Customer Name (as provided): Bob's Auto Care
- Address (as provided): 142 Main St, Springfield, IL

Use Google Search and Google Maps grounding to validate this record
against state business registries, BBB, Google Business listings, and
any other authoritative sources you find. Follow the output format and
guardrails in your system instructions exactly.
"""

print("\n--- Running validation with cached system prompt ---\n")
output = validate_record_cached(sample_prompt)
print(output)

# ---------------------------------------------------------------------------
# 6. Cleanup — delete the cache when done (optional)
#    If you're running a batch job, skip this and let the TTL expire naturally.
# ---------------------------------------------------------------------------
# client.caches.delete(cached_content.name)
# print("\nCache deleted.")