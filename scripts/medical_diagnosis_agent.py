#!/usr/bin/env python3
"""
Medical Diagnosis Bot Agent - Synthetic Log Generator

This standalone script generates realistic medical consultation agent logs
through the Portkey API for testing the evaluation and recommendation system.

Model: claude-3-5-sonnet-20241022 (mid-tier, excellent reasoning)
- Allows recommendation engine to suggest both cheaper and premium alternatives
- Cheaper: claude-3-haiku, gpt-4o-mini, gemini-1.5-flash
- Premium: gpt-4o, claude-3-opus

Usage:
    pip install portkey-ai python-dotenv
    export PORTKEY_API_KEY="your-api-key"
    python medical_diagnosis_agent.py
"""

import asyncio
import os
import random
import uuid
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from portkey_ai import Portkey

# Load environment variables
load_dotenv()

# Configuration
PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
MODEL = "claude-3-5-sonnet-20241022"  # Mid-tier - recommendations can go both ways
PROVIDER = "anthropic"
NUM_CONVERSATIONS = 50  # Target: 100+ logs (2+ turns per conversation)

# System prompt for medical diagnosis assistant
SYSTEM_PROMPT = """You are a helpful medical information assistant. You provide general health information and help users understand symptoms and potential conditions.

IMPORTANT DISCLAIMERS:
- You are NOT a replacement for professional medical advice, diagnosis, or treatment
- Always recommend consulting a healthcare provider for proper diagnosis
- In emergencies, advise users to call emergency services or go to the ER
- Do not prescribe medications or specific treatments

Your role:
- Help users understand their symptoms
- Provide general information about common conditions
- Suggest when to seek professional medical care
- Explain medical terminology in simple terms
- Discuss general wellness and prevention

Guidelines:
- Be empathetic and reassuring but not dismissive
- Ask clarifying questions about symptoms (duration, severity, associated symptoms)
- Provide balanced information without causing unnecessary alarm
- Always emphasize the importance of professional medical evaluation
- Keep responses informative but concise (under 250 words unless detailed explanation needed)
- Never diagnose definitively - use phrases like "this could be", "symptoms are consistent with"

You can discuss:
- Symptom analysis and possible causes
- When symptoms warrant urgent care
- General information about conditions and diseases
- Lifestyle and preventive health measures
- Questions to ask your doctor"""

# Realistic medical consultation scenarios
SCENARIOS = [
    # Respiratory Issues
    {
        "category": "respiratory",
        "conversations": [
            [
                "I've had a persistent cough for 3 weeks now. No fever, just a dry, annoying cough.",
                "It gets worse at night. I don't smoke but I do have allergies. Should I be worried?"
            ],
            [
                "I'm having trouble breathing when I climb stairs. I'm 45 and a bit overweight.",
                "It started about a month ago. No chest pain, just shortness of breath."
            ],
            [
                "My child has been wheezing since yesterday. She's 6 years old.",
                "She doesn't have a fever and is eating normally. Should I take her to the doctor?"
            ],
            [
                "I think I have a sinus infection. Headache, facial pressure, and green mucus.",
                "This is the third time this year. Why do I keep getting them?"
            ],
            [
                "I've been snoring really badly according to my partner. Sometimes I wake up gasping.",
                "I'm tired all the time even after 8 hours of sleep. Could this be sleep apnea?"
            ],
        ]
    },
    # Digestive Issues
    {
        "category": "digestive",
        "conversations": [
            [
                "I've been having stomach pain after eating for the past week.",
                "It's in the upper middle area, kind of burning. Worse after spicy food."
            ],
            [
                "I'm experiencing constant bloating and gas. It's really uncomfortable.",
                "I've tried cutting out dairy but it doesn't seem to help. What else could it be?"
            ],
            [
                "I noticed blood in my stool this morning. It was bright red.",
                "I've had hemorrhoids before. Is this the same thing or something more serious?"
            ],
            [
                "I've had diarrhea for 4 days now. Came back from a trip to Mexico.",
                "I've been staying hydrated but I'm starting to feel weak."
            ],
            [
                "I keep getting heartburn, especially when I lie down at night.",
                "I've been taking antacids but they're not helping much anymore."
            ],
        ]
    },
    # Pain and Musculoskeletal
    {
        "category": "pain",
        "conversations": [
            [
                "I have sharp lower back pain that started 2 days ago after lifting something heavy.",
                "The pain shoots down my left leg when I sit. Is this a slipped disc?"
            ],
            [
                "My knees have been aching for months. I'm 55 and used to run a lot.",
                "They're stiff in the morning and crack when I bend them. Arthritis?"
            ],
            [
                "I've had a headache for 5 days straight. Over-the-counter meds barely help.",
                "It's on one side, near my temple. I also see some light sensitivity."
            ],
            [
                "My shoulder has been painful for weeks. I can't lift my arm above my head.",
                "I don't remember injuring it. It just started gradually."
            ],
            [
                "I keep getting muscle cramps in my legs at night. Very painful.",
                "I drink plenty of water. Could I be missing some nutrient?"
            ],
        ]
    },
    # Skin Conditions
    {
        "category": "skin",
        "conversations": [
            [
                "I have a rash on my arms that's been spreading for a week.",
                "It's red and itchy. I haven't changed any products or tried new foods."
            ],
            [
                "There's a mole on my back that seems to have changed shape recently.",
                "My wife noticed it looks different. It's slightly darker than before."
            ],
            [
                "I've had dry, flaky patches on my elbows and knees for years.",
                "It gets worse in winter. My doctor mentioned psoriasis but I never followed up."
            ],
            [
                "I woke up with hives all over my body. No new foods or medications.",
                "They're extremely itchy. Should I go to the ER?"
            ],
            [
                "I have a cut that's been healing very slowly. It's been 2 weeks.",
                "I'm diabetic. Is this related to my condition?"
            ],
        ]
    },
    # Cardiac Symptoms
    {
        "category": "cardiac",
        "conversations": [
            [
                "I've been having heart palpitations randomly throughout the day.",
                "My heart suddenly beats really fast then goes back to normal. No chest pain."
            ],
            [
                "I had chest tightness during my workout yesterday. It went away when I stopped.",
                "I'm 50 years old with high cholesterol. Should I be concerned?"
            ],
            [
                "My blood pressure reading at the pharmacy was 160/100. Is that dangerous?",
                "I haven't been to a doctor in years. I don't take any medications."
            ],
            [
                "I sometimes feel dizzy when I stand up quickly. Almost fainted once.",
                "My heart rate seems normal. Could this be a heart problem?"
            ],
            [
                "My ankles have been swelling, especially by the end of the day.",
                "I'm on my feet a lot for work. But it seems worse than usual lately."
            ],
        ]
    },
    # Mental Health
    {
        "category": "mental_health",
        "conversations": [
            [
                "I've been feeling really anxious lately, can't seem to relax.",
                "My heart races, I sweat, and my thoughts won't stop. Is this anxiety disorder?"
            ],
            [
                "I've had trouble sleeping for weeks. I just lie awake thinking.",
                "I'm exhausted during the day but wired at night. Any suggestions?"
            ],
            [
                "I've been feeling sad and unmotivated for about a month now.",
                "Things I used to enjoy don't interest me anymore. Should I see someone?"
            ],
            [
                "I keep having panic attacks. They come out of nowhere.",
                "I feel like I can't breathe and I'm going to die. It's terrifying."
            ],
            [
                "I think I'm burned out from work. I can't concentrate on anything.",
                "Even small tasks feel overwhelming. Is this depression or just stress?"
            ],
        ]
    },
    # Neurological Symptoms
    {
        "category": "neurological",
        "conversations": [
            [
                "I've been having tingling in my hands and feet for a few weeks.",
                "It's like pins and needles. Sometimes my hands feel weak too."
            ],
            [
                "I keep getting dizzy spells. The room spins for about 30 seconds.",
                "It happens when I turn my head quickly. I also feel nauseous."
            ],
            [
                "I've noticed my memory isn't as good as it used to be. I'm 65.",
                "I forget names and where I put things. Is this normal aging or dementia?"
            ],
            [
                "My husband had a seizure for the first time. He's never had one before.",
                "He's 48 and healthy. The doctors are running tests. What could cause this?"
            ],
            [
                "I've been having frequent headaches with visual disturbances.",
                "I see zig-zag lines before the headache starts. Are these migraines?"
            ],
        ]
    },
    # Endocrine/Metabolic
    {
        "category": "endocrine",
        "conversations": [
            [
                "I'm always tired even though I sleep 9 hours. I've also gained weight.",
                "My skin is dry and I'm always cold. Could this be thyroid related?"
            ],
            [
                "I'm urinating very frequently and always thirsty. Lost 10 pounds.",
                "I'm 35 and these symptoms started suddenly. Should I get tested for diabetes?"
            ],
            [
                "My periods have become very irregular over the past 6 months.",
                "I'm 47. Is this perimenopause? I've also been having hot flashes."
            ],
            [
                "I've been losing hair more than usual. It's thinning all over.",
                "I'm a 30-year-old woman. My nails are also brittle."
            ],
            [
                "My blood sugar readings have been high even though I take my medication.",
                "I'm type 2 diabetic. What could be causing this?"
            ],
        ]
    },
    # Infections and Immune
    {
        "category": "infections",
        "conversations": [
            [
                "I have a fever of 101°F, body aches, and chills. Started yesterday.",
                "I also have a sore throat. Is this the flu or could it be COVID?"
            ],
            [
                "I think I have a UTI. Burning when I urinate and constant urge to go.",
                "I've had UTIs before. Can I treat this at home or do I need antibiotics?"
            ],
            [
                "I was bitten by a tick last week. Found a red ring around the bite.",
                "The ring has expanded to about 3 inches. Is this Lyme disease?"
            ],
            [
                "My child has had a fever for 3 days. Pediatrician said it's viral.",
                "When should I be worried? The fever keeps coming back."
            ],
            [
                "I cut myself and the wound is now red, swollen, and warm to touch.",
                "There's some pus coming out. Is this infected? Should I see a doctor?"
            ],
        ]
    },
    # Preventive Health
    {
        "category": "preventive",
        "conversations": [
            [
                "I'm turning 50 and want to know what health screenings I should get.",
                "I'm generally healthy, no major medical history. What tests are important?"
            ],
            [
                "What vaccines should an adult get? I don't think I'm up to date.",
                "I had my childhood vaccines but nothing since then. I'm 35."
            ],
            [
                "I want to lose weight and improve my health. Where do I start?",
                "I'm 40 pounds overweight and have prediabetes. Diet or exercise first?"
            ],
            [
                "My family has a history of heart disease. How can I reduce my risk?",
                "My father had a heart attack at 55. I'm 40 now."
            ],
            [
                "What are the signs of cancer that I should watch out for?",
                "I want to catch things early. Are there symptoms that shouldn't be ignored?"
            ],
        ]
    },
]


def generate_metadata(category: str, conversation_id: str) -> dict[str, Any]:
    """Generate metadata for the log entry."""
    return {
        "agent_type": "medical_diagnosis",
        "agent_name": "HealthGuide AI Assistant",
        "category": category,
        "conversation_id": conversation_id,
        "environment": "synthetic_test",
        "generated_at": datetime.utcnow().isoformat(),
        "disclaimer": "For informational purposes only - not medical advice",
    }


async def run_conversation(
    client: Portkey,
    scenario: dict,
    conversation_messages: list[str],
    conversation_num: int,
) -> list[dict]:
    """Run a single conversation and return log info."""
    conversation_id = str(uuid.uuid4())
    trace_id = f"md-{conversation_id[:8]}"
    logs_generated = []
    
    # Build conversation history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    print(f"\n--- Conversation {conversation_num} ({scenario['category']}) ---")
    print(f"Trace ID: {trace_id}")
    
    for i, user_message in enumerate(conversation_messages):
        # Add user message
        messages.append({"role": "user", "content": user_message})
        print(f"\nUser: {user_message[:80]}...")
        
        try:
            # Call the API through Portkey
            response = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=600,
                temperature=0.7,
                metadata=generate_metadata(scenario["category"], conversation_id),
                # Portkey-specific headers for tracing
                extra_headers={
                    "x-portkey-trace-id": trace_id,
                    "x-portkey-span-id": f"{trace_id}-{i}",
                },
            )
            
            assistant_message = response.choices[0].message.content
            messages.append({"role": "assistant", "content": assistant_message})
            
            print(f"Assistant: {assistant_message[:80]}...")
            print(f"Tokens: {response.usage.total_tokens}")
            
            logs_generated.append({
                "trace_id": trace_id,
                "turn": i + 1,
                "tokens": response.usage.total_tokens,
                "category": scenario["category"],
            })
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)
            
        except Exception as e:
            print(f"Error in conversation: {e}")
            logs_generated.append({
                "trace_id": trace_id,
                "turn": i + 1,
                "error": str(e),
                "category": scenario["category"],
            })
    
    return logs_generated


async def main():
    """Main function to generate medical diagnosis logs."""
    if not PORTKEY_API_KEY:
        print("Error: PORTKEY_API_KEY environment variable not set")
        print("Please set it: export PORTKEY_API_KEY='your-api-key'")
        return
    
    print("=" * 60)
    print("Medical Diagnosis Bot - Synthetic Log Generator")
    print("=" * 60)
    print(f"Model: {MODEL} ({PROVIDER})")
    print(f"Target Conversations: {NUM_CONVERSATIONS}")
    print(f"Expected Logs: {NUM_CONVERSATIONS * 2}+ (2+ turns per conversation)")
    print("=" * 60)
    print("\n⚠️  DISCLAIMER: This generates synthetic medical conversations")
    print("    for testing purposes only. Not for actual medical advice.\n")
    
    # Initialize Portkey client
    client = Portkey(
        api_key=PORTKEY_API_KEY,
        provider=PROVIDER,
    )
    
    all_logs = []
    conversation_count = 0
    
    # Distribute conversations across scenarios
    conversations_per_scenario = NUM_CONVERSATIONS // len(SCENARIOS)
    extra_conversations = NUM_CONVERSATIONS % len(SCENARIOS)
    
    for scenario_idx, scenario in enumerate(SCENARIOS):
        # Calculate how many conversations for this scenario
        num_convos = conversations_per_scenario
        if scenario_idx < extra_conversations:
            num_convos += 1
        
        print(f"\n\n{'='*40}")
        print(f"Category: {scenario['category'].upper()}")
        print(f"Conversations: {num_convos}")
        print(f"{'='*40}")
        
        # Run conversations for this scenario
        for i in range(num_convos):
            conversation_count += 1
            
            # Pick a conversation from this scenario (cycle through available ones)
            conv_messages = scenario["conversations"][i % len(scenario["conversations"])]
            
            # Add some variation by occasionally adding follow-up turns
            if random.random() > 0.4:  # 60% chance of additional follow-up
                follow_ups = [
                    "Thank you. Should I make a doctor's appointment soon?",
                    "What should I tell my doctor about these symptoms?",
                    "Are there any red flags I should watch for?",
                    "Is there anything I can do at home to help with this?",
                    "How long should I wait before seeking medical attention?",
                    "Should I go to urgent care or can this wait for a regular appointment?",
                    "Are there any tests I should ask my doctor about?",
                ]
                conv_messages = list(conv_messages) + [random.choice(follow_ups)]
            
            logs = await run_conversation(
                client, scenario, conv_messages, conversation_count
            )
            all_logs.extend(logs)
            
            # Progress update
            if conversation_count % 10 == 0:
                print(f"\n>>> Progress: {conversation_count}/{NUM_CONVERSATIONS} conversations complete")
                print(f">>> Total logs generated: {len(all_logs)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total Conversations: {conversation_count}")
    print(f"Total Logs Generated: {len(all_logs)}")
    print(f"Model Used: {MODEL}")
    print(f"Provider: {PROVIDER}")
    
    # Category breakdown
    category_counts = {}
    for log in all_logs:
        cat = log.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nLogs by Category:")
    for cat, count in sorted(category_counts.items()):
        print(f"  - {cat}: {count}")
    
    # Token statistics
    token_counts = [log.get("tokens", 0) for log in all_logs if "tokens" in log]
    if token_counts:
        print(f"\nToken Statistics:")
        print(f"  - Total tokens: {sum(token_counts):,}")
        print(f"  - Average per request: {sum(token_counts) // len(token_counts):,}")
        print(f"  - Min: {min(token_counts):,}, Max: {max(token_counts):,}")
    
    # Error summary
    errors = [log for log in all_logs if "error" in log]
    if errors:
        print(f"\nErrors encountered: {len(errors)}")
        for err in errors[:5]:
            print(f"  - {err.get('error', 'Unknown')}")
    
    print("\n" + "=" * 60)
    print("Logs are now available in your Portkey dashboard!")
    print("You can sync them to your project using the log sync feature.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
