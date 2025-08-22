import pandas as pd
import json
import asyncio
import httpx
import random
import re
import html
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

# --- Configuration ---
# IMPORTANT: Paste your Gemini API key here

VALIDATION_FILENAME = "cleaned_dataset.xlsx"
LLM_TO_TEST = "gemini-1.5-flash-latest" 
REQUEST_DELAY_SECONDS = 2
MAX_RETRIES = 3
NUM_EMAILS_TO_TEST = 175

# --- Prompt Definition ---
PROMPTS = {
    'prompt_1_direct_json': """
You are an expert email classifier. Your task is to accurately classify the given email into a two-level category system.

The main categories are:
- 'recruiting'
- 'general'

If the main_category is 'recruiting', the sub_category must be one of the following: ['add_to_calender', 'assignment', 'availability', 'cancelled_call', 'deadline_change', 'document_request', 'document_submission', 'feedback', 'follow_up', 'interview_cancel', 'interview_confirmation', 'interview_feedback', 'interview_invite', 'interview_prep', 'interview_reschedule', 'interview_schedule', 'next_interview', 'next_round', 'no_text', 'phone_screen', 'post_interview_debrief', 'referral_confirmation', 'rejection', 'rescheduling', 'role_outreach', 'schedule_delay', 'scheduling', 'screening_call', 'shortlisted', 'status_update', 'status_update_pending', 'work_location'].
If the main_category is 'general', the sub_category must be 'N/A'.

Analyze the content of the email and provide the classification in a single, valid JSON object. Do not include any text before or after the JSON object.

Example output:
{{"main_category": "recruiting", "sub_category": "interview_schedule"}}

Now, classify this email:
---
{email_text}
---
"""
}

def clean_email_text(text):
    """Cleans email text by removing HTML tags and handling non-string data."""
    text = str(text)
    text = html.unescape(text)
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

def parse_llm_json_response(raw_response_text):
    """Extracts a JSON object from a raw string response."""
    match = re.search(r'\{.*\}', raw_response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {"main_category": "parse_error", "sub_category": "parse_error"}
    return {"main_category": "parse_error", "sub_category": "parse_error"}

async def get_gemini_response(client, email_text, prompt_template, retries=MAX_RETRIES):
    """Sends a request to the Gemini API and handles retries."""
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{LLM_TO_TEST}:generateContent?key={GEMINI_API_KEY}"
    prompt_text = prompt_template.format(email_text=email_text)
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    
    for attempt in range(retries):
        try:
            await asyncio.sleep(random.uniform(0.5, REQUEST_DELAY_SECONDS))
            response = await client.post(api_url, json=payload, timeout=60)
            
            if response.status_code == 200:
                response_data = response.json()
                raw_text = response_data['candidates'][0]['content']['parts'][0]['text']
                return parse_llm_json_response(raw_text)
            elif response.status_code == 429:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited. Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
            else:
                print(f"Request failed with status {response.status_code}: {response.text}")
                if attempt == retries - 1:
                    return {"main_category": "api_error", "sub_category": "api_error"}
        
        except httpx.RequestError as e:
            print(f"An HTTPX error occurred: {e}")
            if attempt == retries - 1:
                return {"main_category": "api_error", "sub_category": "api_error"}

    return {"main_category": "retry_failed", "sub_category": "retry_failed"}

async def main():
    """Main function to run the classification process."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_API_KEY_HERE":
        print("Error: Please replace 'YOUR_API_KEY_HERE' with your actual Gemini API key in the script.")
        return

    try:
        validation_df = pd.read_excel(VALIDATION_FILENAME, nrows=NUM_EMAILS_TO_TEST)
    except FileNotFoundError:
        print(f"Error: Input file '{VALIDATION_FILENAME}' not found.")
        return

    validation_df['cleaned_text'] = validation_df['text'].apply(clean_email_text)
    
    async with httpx.AsyncClient() as client:
        tasks = [
            get_gemini_response(client, row['cleaned_text'], PROMPTS['prompt_1_direct_json'])
            for _, row in validation_df.iterrows()
        ]
        results = await asyncio.gather(*tasks)

    classified_df = validation_df.copy()
    classified_df['pred_main'] = [res.get('main_category', 'error') for res in results]
    classified_df['pred_sub'] = [res.get('sub_category', 'error') for res in results]

    output_filename = 'gemini_classified_output.xlsx'
    classified_df.to_excel(output_filename, index=False)
    print(f"Classification complete. Results saved to '{output_filename}'")


if __name__ == "__main__":
    asyncio.run(main())
