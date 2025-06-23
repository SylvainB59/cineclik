import requests
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import random 
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)





@app.route('/')
def home():
    return render_template("index.html")



# recherche
@app.route('/recherche')

def recherche():
    
    return render_template('recherche.html')
   


# filmtofilm 
@app.route("/filmtofilm", methods=["GET"])
def filmtofilm():
    query = request.args.get("query")
    # Pour l'instant, pas de logique, juste afficher la page
    print(f"Recherche film to film : {query}")

    return render_template("filmtofilm.html")


# mood to film 
@app.route("/moodtofilm")
def moodtofilm():
    return render_template("moodtofilm.html")



# group to film
@app.route('/grouptofilm')
def grouptofilm():
    group = request.args.get('group')
    if group:
        # Logique de recommandation à partir du groupe (à adapter selon tes données)
        recommended_films = get_films_by_group(group)
        return render_template('grouptofilm_result.html', group=group, films=recommended_films)
    return render_template('grouptofilm.html')

def get_films_by_group(group):
    # Exemple simple, à remplacer par ta base ou logique réelle
    films_dict = {
        
    }
    return films_dict.get(group, [])





# Configuration Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel("gemini-1.5-flash")
gemini_chat = model.start_chat(history=[])

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        return render_template("chat.html")
    else:  # POST
        try:
            user_input = request.json["message"]
            response = gemini_chat.send_message(user_input)
            return jsonify({"reply": response.text})
        except Exception as e:
            print(f"Erreur : {e}")
            return jsonify({"reply": "Désolé, une erreur est survenue."}), 500


if __name__ == "__main__":
    app.run(debug=True)