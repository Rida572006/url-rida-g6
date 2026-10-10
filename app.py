from flask import Flask, request, render_template_string
from predict import hybrid_predict_url

app = Flask(__name__)

PAGE = """
<!doctype html>
<title>Malicious URL Detection</title>
<body style="font-family:sans-serif;max-width:600px;margin:40px auto;padding:0 16px">
  <h2>🛡️ AI-Based Malicious URL Detection</h2>
  <p>Enter a website URL to check if it is Safe or Potentially Malicious/Phishing.</p>
  <form method="post">
    <input name="url" value="{{ url }}" placeholder="https://example.com"
           style="width:100%;padding:10px" required>
    <button style="margin-top:10px;padding:10px 20px">Check URL</button>
  </form>
  {% if result %}
    <h3>Detection Result</h3>
    <pre style="white-space:pre-wrap;background:#f4f4f4;padding:12px">{{ result }}</pre>
  {% endif %}
</body>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    url, result = "", None
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        try:
            result = hybrid_predict_url(url)
        except Exception as e:
            result = f"Error: {e}"
    return render_template_string(PAGE, url=url, result=result)
