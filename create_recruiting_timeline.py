import pandas as pd
import re

def save_timeline_as_html(timeline, output_filename='recruiting_timeline.html'):
    """Saves the timeline data as a styled HTML file."""
    
    # Basic CSS for styling the HTML page
    html_style = """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 960px; margin: 20px auto; }
        h1 { color: #1a1a1a; border-bottom: 2px solid #f0f0f0; padding-bottom: 10px; }
        h2 { color: #333; background-color: #f7f7f7; padding: 10px; border-radius: 5px; border-left: 5px solid #007bff; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 25px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        tr:hover { background-color: #f5f5f5; }
        .category { font-weight: 500; }
        .subject { color: #555; }
    </style>
    """
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(f"<!DOCTYPE html><html><head><title>Recruiting Timeline</title>{html_style}</head><body>")
        f.write("<h1>Recruiting Activity Timeline</h1>")
        
        for domain, events in sorted(timeline.items()):
            if domain == "Unknown":
                continue
            f.write(f"<h2>Timeline for {domain}</h2>")
            f.write("<table>")
            f.write("<tr><th>Date</th><th>Category</th><th>From</th><th>Subject</th></tr>")
            
            for event in events:
                date = event.get('date', 'N/A')
                category = event.get('sub_category', 'N/A')
                sender = event.get('from', 'N/A')
                subject = event.get('subject', 'N/A')
                f.write(f"<tr><td>{date}</td><td class='category'>{category}</td><td>{sender}</td><td class='subject'>{subject}</td></tr>")
                
            f.write("</table>")
        
        f.write("</body></html>")
    print(f"\nSuccess! Your timeline has been saved to '{output_filename}'")


def create_timeline(input_file='gemini_classified_output.xlsx'):
    """
    Loads classified email data and builds a chronological timeline of
    recruiting activities, grouped by company domain.
    """
    try:
        df = pd.read_excel(input_file)
        print(f"Successfully loaded '{input_file}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        return

    # Define correct column names
    date_col, from_col, subject_col = 'headers.date', 'headers.from', 'headers.subject'
    
    # Filter for Recruiting Emails
    recruiting_df = df[df['pred_main'] == 'recruiting'].copy()
    print(f"Found {len(recruiting_df)} recruiting-related emails to analyze.")

    # Clean and Prepare Data
    recruiting_df['date_dt'] = pd.to_datetime(recruiting_df[date_col], errors='coerce')
    recruiting_df.dropna(subset=['date_dt'], inplace=True)

    def get_domain(email):
        if not isinstance(email, str): return "Unknown"
        match = re.search(r'@([\w.-]+)', email)
        return match.group(1) if match else "Unknown"

    recruiting_df['domain'] = recruiting_df[from_col].apply(get_domain)
    recruiting_df.sort_values(by='date_dt', inplace=True)

    # Build the Timeline Dictionary
    timeline = {}
    for _, row in recruiting_df.iterrows():
        domain = row['domain']
        if domain not in timeline:
            timeline[domain] = []
        
        timeline[domain].append({
            'date': row['date_dt'].strftime('%Y-%m-%d'),
            'from': row[from_col],
            'subject': row[subject_col],
            'sub_category': row['pred_sub']
        })

    # Print the timeline to the console (as before)
    print("\n--- Recruiting Activity Timeline (Console Output) ---")
    for domain, events in sorted(timeline.items()):
        if domain == "Unknown": continue
        print(f"\n▼▼▼ Timeline for {domain} ▼▼▼")
        for event in events:
            subject = event.get('subject', 'No Subject')
            sender = event.get('from', 'Unknown Sender')
            category = event.get('sub_category', 'No Category')
            print(f"  - {event['date']} | {category:<25} | From: {sender:<40} | Subject: {subject}")
            
    # Save the timeline to a beautiful HTML file
    save_timeline_as_html(timeline)


if __name__ == "__main__":
    create_timeline()