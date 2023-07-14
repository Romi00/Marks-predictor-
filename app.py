from flask import Flask, render_template, request
import joblib
from sklearn.linear_model import LinearRegression
lr=LinearRegression()

app = Flask('__name__')
model=joblib.load("Student_marks_prediction_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict" , methods=['POST'])
def predict():
    form_data = request.form['hours']
    form_data = float(form_data)
    output = model.predict([[form_data]])
    output = str(output[0][0].round(2))

    check_percent = float(output)
    if check_percent >= 100:
        output = '100'
    
    if check_percent <= 33:
        output = 'nalla phad le bhai hdd krdi bare minimum bhi ni nikl raha'

    return render_template('index.html', some_text = "YOU WILL GET " + str(output) + '% WHEN YOU STUDY ' + str(form_data) + ' HOURS PER DAY')


if __name__ == "__main__":
    # app.run(host = '0.0.0.0', port=5000)
    app.run(debug=True)
