import gradio as gr
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yake


summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
analyzer = SentimentIntensityAnalyzer()
kw_extractor = yake.KeywordExtractor(top=5, stopwords=None)


def perception_analysis(linkedin, github, tweets):
    combined_text = f"{linkedin}\n{github}\n{tweets}"
    summary = summarizer(combined_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    sentiment = analyzer.polarity_scores(combined_text)
    keywords = [kw for kw, score in kw_extractor.extract_keywords(combined_text)]

    # Styled Markdown Output
    return f"""
### 🧠 AI Perception Summary
{summary}

---

### 💬 Sentiment Analysis
- **Positive**: {sentiment['pos']}
- **Neutral**: {sentiment['neu']}
- **Negative**: {sentiment['neg']}
- **Compound**: {sentiment['compound']}

---

### 🔑 Top Perceived Keywords
- {keywords[0]}
- {keywords[1]}
- {keywords[2]}
- {keywords[3]}
- {keywords[4]}
"""


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🧠 Perception Hacker  
AI that analyzes your public content and tells you how others might perceive you.  
_Powered by LLMs, sentiment & keyword extraction._
""")

    with gr.Row():
        linkedin = gr.Textbox(placeholder="Paste your LinkedIn summary here", label="LinkedIn Summary", lines=3)
        github = gr.Textbox(placeholder="Paste your GitHub README here", label="GitHub README", lines=3)
        tweets = gr.Textbox(placeholder="Paste your tweets or short-form content here", label="Tweets (combine all)", lines=3)

    output = gr.Markdown(label="Output")

    with gr.Row():
        submit = gr.Button("🚀 Submit", variant="primary")
        clear = gr.Button("🧹 Clear", variant="secondary")

    submit.click(perception_analysis, inputs=[linkedin, github, tweets], outputs=output)
    clear.click(lambda: ("", "", "", ""), inputs=[], outputs=[linkedin, github, tweets, output])

demo.launch()
