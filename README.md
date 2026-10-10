# Malicious URL Detection (VS Code)

1. `python -m venv venv` then activate it
2. `pip install -r requirements.txt`
3. Put `malicious_phish.csv` (or the dataset .zip) inside the `data/` folder
4. `python train_models.py`  (trains, prints metrics, saves models + confusion matrix PNGs)
5. `python predict.py`       (quick test)
6. `python app.py`           (launch the Gradio web app)
