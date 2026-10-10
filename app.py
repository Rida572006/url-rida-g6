import gradio as gr
from predict import hybrid_predict_url

interface = gr.Interface(
    fn=hybrid_predict_url,
    inputs=gr.Textbox(label="Enter Website URL", placeholder="https://example.com"),
    outputs=gr.Textbox(label="Detection Result"),
    title="🛡️ AI-Based Malicious URL Detection",
    description=(
        "Enter a website URL to detect whether it is "
        "Safe or Potentially Malicious/Phishing. "
        "The system uses Machine Learning with trusted-domain verification."
    ),
)

if __name__ == "__main__":
    interface.launch(share=True)
