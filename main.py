from flask import Flask, render_template
import os
import joblib            # Para carregar o modelo salvo
import pandas as pd      
from flask import Flask, request, jsonify, render_template

# template_dir = os.path.join(os.getcwd(), '/templates')

app = Flask(__name__)

# from routes.routes import *

@app.route("/")
def index():
  return render_template('index.html')

modelo_path = os.path.join(os.getcwd(), 'modelo_diabetes.pkl')
modelo = joblib.load(modelo_path)  # Carrega o modelo salvo

@app.route("/predict", methods=['POST'])  # Corrigido para 'methods'
def prever():
    
    try:
        # Obter os dados JSON do corpo da requisição
        dados = request.get_json()
        # print(dados)

        # Converter os dados para um DataFrame
        df = pd.DataFrame(dados)

        # Fazer a previsão usando o modelo
        previsao = modelo.predict(df)
        print(previsao)

        return jsonify({
            "status": "success",
            "message": "Previsão realizada com sucesso",
            "data": {
                "diabetes": int(previsao[0])
            }
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Ocorreu um erro: {str(e)}"
        }), 400
    

if __name__ == "__main__":
    # app.debug = True
    app.run(host='0.0.0.0', debug=True)
