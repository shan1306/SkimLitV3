import tensorflow as tf
from helper import preprocess_data_for_inference, make_predictions, print_segregated_abstract
from flask import Flask, render_template, request

app = Flask(__name__)

loaded_model = tf.keras.models.load_model("Skimlit_20K_best_model_6")  

def predict(text):
  sample_inputs, sample_sentences = preprocess_data_for_inference(text)
  model_preds = make_predictions(loaded_model, sample_inputs)
  final_abstract = print_segregated_abstract(model_preds, sample_sentences)
  return final_abstract


# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index', methods=['GET', 'POST'])
def predict_text():
    try:
        if request.method == 'POST':
            text = request.form['text']
            prediction = predict(text)
            return render_template('index.html', prediction=prediction)
        # If it's a GET request, just render the form again
        return render_template('index.html')
    except:
        return render_template('index.html', prediction='ABSTRACT NOT FOUND')


if __name__ == '__main__':
    app.run(debug=True)
