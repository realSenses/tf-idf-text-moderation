import gradio as gr
from inference import ToxicCommentClassifier
import pandas as pd

# Initialize the classifier
classifier = ToxicCommentClassifier()

def analyze_comment(text):
    if not text or text.strip() == "":
        return "Please enter a comment to analyze.", None, None
    
    result = classifier.predict(text)
    
    # Create status message
    if result['is_toxic']:
        status = f"⚠️ This comment appears to be toxic!\n\nDetected categories: {', '.join(result['toxic_categories'])}"
    else:
        status = "✓ This comment appears to be non-toxic."
    
    # Create detailed results DataFrame
    categories_data = []
    for category in classifier.toxic_categories:
        categories_data.append({
            'Category': category.replace('_', ' ').title(),
            'Toxic': '⚠️ Yes' if result['predictions'][category] else '✓ No',
            'Confidence': f"{result['probabilities'][category]:.1%}"
        })
    
    df_results = pd.DataFrame(categories_data)
    
    # Create probability chart data
    prob_data = pd.DataFrame({
        'Category': [cat.replace('_', ' ').title() for cat in classifier.toxic_categories],
        'Probability': [result['probabilities'][cat] for cat in classifier.toxic_categories]
    })
    
    return status, df_results, prob_data

# Example comments for testing
examples = [
    "This is a great article, thank you for sharing!",
    "I disagree with your opinion but respect your viewpoint.",
    "You're an idiot and don't know what you're talking about!!!",
    "I HATE people like you, you should just disappear",
    "This is the WORST thing I've ever seen... complete garbage!!!",
    "Your analysis is flawed but interesting nonetheless",
    "This content is inappropriate and offensive.",
    "Thanks for the helpful information!"
]

# Create Gradio interface
with gr.Blocks(title="Toxic Comment Moderation") as demo:
    gr.Markdown("""
    # 🛡️ Toxic Comment Moderation System
    
    This AI-powered system analyzes comments to detect various types of toxic content including:
    - **Toxic**: Generally toxic or harmful content
    - **Severe Toxic**: Extremely toxic content
    - **Obscene**: Obscene or vulgar language
    - **Threat**: Threatening language
    - **Insult**: Insulting or demeaning content
    - **Identity Hate**: Hate speech targeting identity groups
    
    Enter a comment below to analyze it for toxic content.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Comment to Analyze",
                placeholder="Enter a comment here...",
                lines=3
            )
            
            analyze_btn = gr.Button("🔍 Analyze Comment", variant="primary")
            
            gr.Markdown("### Example Comments:")
            example_btns = []
            for i, example in enumerate(examples):
                btn = gr.Button(
                    f"Example {i+1}: {example[:50]}...",
                    variant="secondary",
                    size="sm"
                )
                example_btns.append(btn)
        
        with gr.Column(scale=1):
            status_output = gr.Textbox(
                label="Analysis Result",
                lines=3,
                interactive=False
            )
            
            results_table = gr.Dataframe(
                label="Detailed Analysis",
                headers=["Category", "Toxic", "Confidence"],
                datatype=["str", "str", "str"],
                interactive=False
            )
            
            prob_chart = gr.BarPlot(
                label="Toxicity Probabilities",
                x="Category",
                y="Probability",
                title="Probability of Each Toxic Category",
                y_lim=[0, 1],
                interactive=False
            )
    
    # Set up event handlers
    analyze_btn.click(
        fn=analyze_comment,
        inputs=input_text,
        outputs=[status_output, results_table, prob_chart]
    )
    
    # Set up example button handlers
    for i, btn in enumerate(example_btns):
        btn.click(
            lambda x=examples[i]: x,
            outputs=input_text
        )
    
    gr.Markdown("""
    ---
    ### About this Model
    
    This model uses machine learning to identify toxic comments across multiple categories. It combines:
    - Advanced text preprocessing with pattern recognition
    - Word and character-level TF-IDF features
    - Statistical text features
    - Logistic regression classifiers with balanced class weights
    
    **Note**: This is an AI system and may not be 100% accurate. Always use human judgment for final moderation decisions.
    
    ### API Usage
    
    This model is also available via API. Check the documentation for integration details.
    """)

if __name__ == "__main__":
    demo.launch()