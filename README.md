# DoseWise

DoseWise is a medication interaction assistant that helps users check combinations of medications using the DDInter interaction database.

It supports:

- Medication name extraction and normalization
- Brand-name and generic-name resolution
- Medication interaction lookup
- Interaction severity detection
- English, Arabic, Arabizi, and mixed-language messages
- Clarification for unknown or ambiguous medication names
- Safety guardrails for overdose and self-harm-related messages
- Session-based conversations and medication follow-ups
- Optional Groq-powered conversational responses
- A browser-based chat interface

> DoseWise is for informational purposes only. It does not replace a doctor, pharmacist, poison control center, or emergency medical service.

## Technology

- Python
- Flask
- Flask-CORS
- Pandas
- RapidFuzz
- DDInter medication interaction data
- Optional Groq API integration
- Vanilla HTML, CSS, and JavaScript frontend
- Pytest test suite

## Project Structure

```text
DoseWise/
├── data/
│   ├── evaluation_cases.json
│   ├── cache/
│   └── processed/
│       ├── drug_mapping.csv
│       ├── enriched_interactions.csv
│       ├── master_interactions.csv
│       ├── test_mapping.csv
│       └── unique_drug_names.csv
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── src/
│   ├── api_lookup.py
│   ├── build_drug_mapping.py
│   ├── clean_merge_ddinter.py
│   ├── drug_mapping.py
│   ├── drug_normalization.py
│   ├── drug_resolver.py
│   ├── groq_chat.py
│   ├── rag_pipeline.py
│   ├── server.py
│   └── validate_drug_mapping.py
├── tests/
├── tools/
├── requirements.txt
├── requirements_pinned.txt
└── wsgi.py
