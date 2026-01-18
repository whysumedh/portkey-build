#!/usr/bin/env python3
"""
Customer Support Bot Agent - Synthetic Log Generator

This standalone script generates realistic customer support agent logs
through the Portkey API for testing the evaluation and recommendation system.

Model: gpt-4o (expensive, high-quality)
- Allows recommendation engine to suggest cheaper alternatives
- Examples: gpt-4o-mini, claude-3-haiku, gemini-1.5-flash

Usage:
    pip install portkey-ai python-dotenv
    export PORTKEY_API_KEY="your-api-key"
    python customer_support_agent.py
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
MODEL = "gpt-4o"  # Expensive model - recommendations can suggest cheaper alternatives
PROVIDER = "openai"
NUM_CONVERSATIONS = 50  # Target: 100+ logs (2+ turns per conversation)

# System prompt for customer support agent
SYSTEM_PROMPT = """You are a helpful customer support agent for TechGadgets Inc., an online electronics retailer.

Your responsibilities:
- Help customers with order issues, returns, refunds, and exchanges
- Provide product information and recommendations
- Troubleshoot technical issues with products
- Handle billing and account inquiries
- Escalate complex issues when necessary

Guidelines:
- Be polite, professional, and empathetic
- Provide clear, concise responses
- Ask clarifying questions when needed
- Offer solutions proactively
- Keep responses under 200 words unless detailed explanation is needed

Company Policies:
- 30-day return policy for unopened items
- 14-day return for opened items with 15% restocking fee
- Free shipping on orders over $50
- Price match guarantee within 7 days of purchase
- Warranty: 1 year manufacturer, optional 2-year extended"""

# Realistic customer support scenarios
SCENARIOS = [
    # Order Issues
    {
        "category": "order_status",
        "conversations": [
            [
                "Hi, I placed an order 5 days ago and haven't received any shipping update. Order #TG-78234",
                "I need it for my son's birthday this weekend. Can you expedite it?"
            ],
            [
                "My order shows delivered but I never received it. Order number TG-91827",
                "I checked with my neighbors and the building office, nothing there."
            ],
            [
                "I ordered the wrong color laptop. Can I change it before it ships?",
                "It's order TG-45632, placed about 2 hours ago"
            ],
            [
                "Where is my order? It's been 2 weeks! Order TG-33821",
                "This is unacceptable. I want a refund if it doesn't arrive today."
            ],
            [
                "Can you tell me when my order will arrive? TG-55912",
                "Thanks! Also, will I get a notification when it's out for delivery?"
            ],
        ]
    },
    # Returns and Refunds
    {
        "category": "returns",
        "conversations": [
            [
                "I want to return the headphones I bought. They hurt my ears.",
                "I've used them for about a week. What's the return process?"
            ],
            [
                "The TV I received has a crack in the screen. I need a replacement.",
                "It was definitely damaged during shipping. The box was crushed."
            ],
            [
                "I need to return a gift I received. I don't have the receipt.",
                "My aunt bought it from your store. Can you look it up by her name?"
            ],
            [
                "How long does it take to get a refund after returning an item?",
                "I returned a laptop 10 days ago and still no refund."
            ],
            [
                "Can I exchange my smartwatch for a different model?",
                "I want to upgrade to the Pro version and pay the difference."
            ],
        ]
    },
    # Technical Support
    {
        "category": "technical",
        "conversations": [
            [
                "My new laptop won't turn on. I charged it overnight.",
                "Yes, the charging light was on. But nothing happens when I press power."
            ],
            [
                "The Bluetooth speaker I bought keeps disconnecting from my phone.",
                "It's an iPhone 14 Pro. The speaker worked fine for the first week."
            ],
            [
                "How do I set up the smart home hub I just bought?",
                "I downloaded the app but it can't find the device."
            ],
            [
                "My wireless earbuds have terrible battery life. Only 2 hours!",
                "The box says 8 hours. Is this a defect?"
            ],
            [
                "The monitor I bought has dead pixels. Is this covered under warranty?",
                "There are about 5 dead pixels in the center of the screen."
            ],
        ]
    },
    # Billing Issues
    {
        "category": "billing",
        "conversations": [
            [
                "I was charged twice for my order. Can you fix this?",
                "Both charges are for $299.99. Order TG-77123."
            ],
            [
                "Why was I charged a shipping fee? I thought orders over $50 are free.",
                "My order total was $75 before shipping was added."
            ],
            [
                "I have a promo code that didn't work during checkout.",
                "The code is SAVE20. It says it's valid until end of month."
            ],
            [
                "Can you remove the extended warranty I accidentally added?",
                "I didn't mean to click it. The order hasn't shipped yet."
            ],
            [
                "I need an invoice for my company purchase. Order TG-88456.",
                "It needs to include the company name: Acme Corporation"
            ],
        ]
    },
    # Product Inquiries
    {
        "category": "product_info",
        "conversations": [
            [
                "What's the difference between the laptop models A500 and A700?",
                "Is the A700 worth the extra $200?"
            ],
            [
                "Do you have the new iPhone in stock?",
                "I need the 256GB in blue. When will it be available?"
            ],
            [
                "Is this tablet compatible with the stylus sold separately?",
                "Which stylus would you recommend for drawing?"
            ],
            [
                "What's the warranty coverage for refurbished items?",
                "I'm considering a refurbished MacBook to save money."
            ],
            [
                "Can you recommend a good webcam for video conferencing?",
                "Budget is around $100. Need good low-light performance."
            ],
        ]
    },
    # Account Issues
    {
        "category": "account",
        "conversations": [
            [
                "I can't log into my account. It says my password is wrong.",
                "I tried resetting it but never received the email."
            ],
            [
                "How do I update my shipping address?",
                "I'm moving next week and have orders pending."
            ],
            [
                "Can you merge my two accounts? I accidentally created a duplicate.",
                "The emails are john@email.com and johnsmith@email.com"
            ],
            [
                "I want to delete my account and all my data.",
                "How long will it take to process this request?"
            ],
            [
                "Can you add my rewards points from my old account?",
                "I had about 5000 points that disappeared when I updated my email."
            ],
        ]
    },
    # Complaints
    {
        "category": "complaints",
        "conversations": [
            [
                "This is the third time I'm contacting support about the same issue!",
                "Nobody seems to be able to help me. I want to speak to a manager."
            ],
            [
                "Your delivery driver left my package in the rain!",
                "The box was soaked and the product inside is damaged."
            ],
            [
                "I've been on hold for 45 minutes. This is ridiculous.",
                "I just need a simple answer about my return."
            ],
            [
                "The product description was misleading. This isn't what I expected.",
                "The listing said 4K but it's only 1080p."
            ],
            [
                "Your competitor has better prices and service.",
                "I'm thinking of taking my business elsewhere."
            ],
        ]
    },
    # Price Match
    {
        "category": "price_match",
        "conversations": [
            [
                "I found this same laptop cheaper on Amazon. Can you price match?",
                "It's $50 less. Here's the link to the listing."
            ],
            [
                "I bought this TV last week and now it's on sale. Can I get the difference?",
                "The price dropped by $100. Order TG-22345."
            ],
            [
                "Does your price match policy include used items?",
                "I found a refurbished version elsewhere for less."
            ],
            [
                "Can you match Best Buy's price for this monitor?",
                "They have it for $279 with free installation."
            ],
            [
                "The item I want is cheaper with a coupon code at another store.",
                "Do you accept competitor coupons?"
            ],
        ]
    },
    # Shipping Issues
    {
        "category": "shipping",
        "conversations": [
            [
                "Can I change my shipping address after ordering?",
                "I realized I put my old address. The order is TG-99001."
            ],
            [
                "Do you offer same-day delivery?",
                "I need a charger urgently for a presentation tomorrow."
            ],
            [
                "My package was shipped to the wrong address.",
                "The tracking shows it was delivered to a city I've never heard of."
            ],
            [
                "Can I pick up my order from your warehouse instead?",
                "I live nearby and need it faster than shipping would take."
            ],
            [
                "Why doesn't my area qualify for free shipping?",
                "I'm only 20 miles from your distribution center."
            ],
        ]
    },
    # Warranty Claims
    {
        "category": "warranty",
        "conversations": [
            [
                "My laptop screen cracked. Is this covered under warranty?",
                "It just happened while it was in my bag. No drops."
            ],
            [
                "How do I claim the extended warranty I purchased?",
                "My phone's battery is draining very fast after 18 months."
            ],
            [
                "The manufacturer says to contact you for warranty service.",
                "It's a blender that stopped working after 6 months."
            ],
            [
                "Can I purchase extended warranty after the initial period?",
                "I forgot to add it when I bought the TV."
            ],
            [
                "What's not covered under the warranty?",
                "I want to understand before filing a claim."
            ],
        ]
    },
]


def generate_metadata(category: str, conversation_id: str) -> dict[str, Any]:
    """Generate metadata for the log entry."""
    return {
        "agent_type": "customer_support",
        "agent_name": "TechGadgets Support Bot",
        "category": category,
        "conversation_id": conversation_id,
        "environment": "synthetic_test",
        "generated_at": datetime.utcnow().isoformat(),
    }


async def run_conversation(
    client: Portkey,
    scenario: dict,
    conversation_messages: list[str],
    conversation_num: int,
) -> list[dict]:
    """Run a single conversation and return log info."""
    conversation_id = str(uuid.uuid4())
    trace_id = f"cs-{conversation_id[:8]}"
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
                max_tokens=500,
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
    """Main function to generate customer support logs."""
    if not PORTKEY_API_KEY:
        print("Error: PORTKEY_API_KEY environment variable not set")
        print("Please set it: export PORTKEY_API_KEY='your-api-key'")
        return
    
    print("=" * 60)
    print("Customer Support Bot - Synthetic Log Generator")
    print("=" * 60)
    print(f"Model: {MODEL} ({PROVIDER})")
    print(f"Target Conversations: {NUM_CONVERSATIONS}")
    print(f"Expected Logs: {NUM_CONVERSATIONS * 2}+ (2+ turns per conversation)")
    print("=" * 60)
    
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
            
            # Add some variation by occasionally adding a third turn
            if random.random() > 0.5:
                follow_ups = [
                    "Thanks for your help!",
                    "One more question - do you have an email confirmation?",
                    "Is there anything else I should know?",
                    "How long will this take to resolve?",
                    "Can I get a reference number for this conversation?",
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
