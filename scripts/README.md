# Synthetic Log Generator Scripts

These standalone Python scripts generate realistic agent log data through the Portkey API for testing the evaluation and recommendation system.

## Overview

| Script | Agent Type | Model | Provider | Expected Logs |
|--------|-----------|-------|----------|---------------|
| `customer_support_agent.py` | Customer Support Bot | `gpt-4o` | OpenAI | 100+ |
| `medical_diagnosis_agent.py` | Medical Diagnosis Bot | `claude-3-5-sonnet-20241022` | Anthropic | 100+ |

## Why These Models?

The models were chosen specifically to test the **recommendation engine**:

### Customer Support Bot (`gpt-4o`)
- **Cost**: High (~$5/1M input, ~$15/1M output)
- **Quality**: Premium
- **Recommendation potential**: 
  - Cheaper alternatives: `gpt-4o-mini`, `claude-3-haiku`, `gemini-1.5-flash`
  - Similar quality: `claude-3-5-sonnet`

### Medical Diagnosis Bot (`claude-3-5-sonnet-20241022`)
- **Cost**: Mid-tier (~$3/1M input, ~$15/1M output)
- **Quality**: Excellent reasoning
- **Recommendation potential**:
  - Cheaper: `claude-3-haiku`, `gpt-4o-mini`, `gemini-1.5-flash`
  - Premium: `gpt-4o`, `claude-3-opus`

## Prerequisites

1. **Portkey Account** with API key
2. **Virtual Keys** configured for OpenAI and/or Anthropic in Portkey
3. **Python 3.10+**

## Setup

1. Install dependencies:
   ```bash
   cd scripts
   pip install -r requirements.txt
   ```

2. Set your Portkey API key:
   ```bash
   # Windows PowerShell
   $env:PORTKEY_API_KEY = "your-portkey-api-key"
   
   # Linux/Mac
   export PORTKEY_API_KEY="your-portkey-api-key"
   ```

   Or create a `.env` file in the `scripts` folder:
   ```
   PORTKEY_API_KEY=your-portkey-api-key
   ```

## Running the Scripts

### Generate Customer Support Logs

```bash
python customer_support_agent.py
```

This will generate ~100+ logs across 10 categories:
- Order status inquiries
- Returns and refunds
- Technical support
- Billing issues
- Product inquiries
- Account issues
- Complaints
- Price match requests
- Shipping issues
- Warranty claims

### Generate Medical Diagnosis Logs

```bash
python medical_diagnosis_agent.py
```

This will generate ~100+ logs across 10 categories:
- Respiratory issues
- Digestive problems
- Pain and musculoskeletal
- Skin conditions
- Cardiac symptoms
- Mental health
- Neurological symptoms
- Endocrine/metabolic
- Infections and immune
- Preventive health

## Log Metadata

Each log includes metadata for filtering and analysis:

```json
{
  "agent_type": "customer_support" | "medical_diagnosis",
  "agent_name": "TechGadgets Support Bot" | "HealthGuide AI Assistant",
  "category": "<scenario_category>",
  "conversation_id": "<uuid>",
  "environment": "synthetic_test",
  "generated_at": "<iso_timestamp>"
}
```

## After Running

1. **View logs** in your Portkey dashboard at https://app.portkey.ai
2. **Sync logs** to your project using the "Refresh from Portkey" feature in the application
3. **Run evaluations** on the imported logs
4. **Get recommendations** from the recommendation engine for alternative models

## Estimated Costs

| Script | Estimated Tokens | Estimated Cost |
|--------|-----------------|----------------|
| Customer Support | ~150,000 | ~$2-3 |
| Medical Diagnosis | ~200,000 | ~$3-4 |

*Costs are approximate and depend on actual response lengths.*

## Customization

You can modify the scripts to:

- **Change number of conversations**: Edit `NUM_CONVERSATIONS` variable
- **Use different models**: Edit `MODEL` and `PROVIDER` variables
- **Add new scenarios**: Add entries to the `SCENARIOS` list
- **Modify prompts**: Edit `SYSTEM_PROMPT` variable

## Troubleshooting

### "PORTKEY_API_KEY environment variable not set"
Make sure you've set the environment variable or created a `.env` file.

### Rate limiting errors
The scripts include delays between requests. If you still hit rate limits, increase the `await asyncio.sleep()` value.

### Authentication errors
Verify your Portkey API key is correct and has access to the specified providers.

### Model not found
Ensure you have virtual keys configured in Portkey for the provider (OpenAI/Anthropic).
