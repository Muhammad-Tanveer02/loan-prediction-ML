import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    int_values = np.array([int(x) for x in request.form.values()])
    final_values = [np.array(int_values)]
    prediction = model.predict(final_values)

    if prediction == 1:
        output = "You are eligible for a Loan"
        text_hex = "#008000"
    elif prediction == 0:
        output = "You are NOT eligible for a Loan"
        text_hex = "#AF0000"

    return render_template(
        "index.html", loan_status_result=output, col_code=text_hex
    )


if __name__ == "__main__":
    app.run(debug=True)
