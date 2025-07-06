from flask import Flask, render_template, request
import pickle
import numpy as np
import random

app = Flask(__name__)

# Load the trained model
with open('model/horse_model.pkl', 'rb') as f:
    model = pickle.load(f)

horse_names = [
    "Thunderbolt", "Shadow Dancer", "Midnight Rider", "Wind Whisperer", "Crimson Comet",
    "Blazing Star", "Golden Mane", "Night Fury", "Storm Runner", "Velvet Hooves",
    "Rocket Dash", "Silent Arrow", "Whirlwind", "Silver Streak", "Black Stallion",
    "Firestorm", "Steel Spirit", "Ocean Breeze", "Misty Blaze", "Lightning Strike",
    "Frostbite", "Jungle King", "Majestic Moon", "Savage Ember", "Ghost Rider",
    "Desert Mirage", "Lone Wolf", "Phantom Blaze", "Icy Thunder", "Sun Chaser",
    "Dark Knight", "Golden Arrow", "Tornado Twist", "Ruby Racer", "Nebula Dash",
    "Galaxy Star", "Twilight Bolt", "Inferno Fury", "Snow Phantom", "Dream Runner",
    "Rapid Shadow", "Night Gale", "Windstorm", "Copper Cloud", "Sky Jumper",
    "Midday Thunder", "Crimson Flash", "Eclipse Rider", "Whiskey Wind", "Zephyr King"
]

@app.route('/', methods=['GET', 'POST'])
def index():
    horses = []
    predicted_winner = None
    form_data = {
        'track_condition': 'Dry',
        'weather': 'Sunny',
        'race_length': 'Short',
        'surface': 'Turf',
        'jockey_experience': 'Intermediate'
    }

    if request.method == 'POST':
        form_data['track_condition'] = request.form.get('track_condition')
        form_data['weather'] = request.form.get('weather')
        form_data['race_length'] = request.form.get('race_length')
        form_data['surface'] = request.form.get('surface')
        form_data['jockey_experience'] = request.form.get('jockey_experience')

        track_map = {'Dry': 0, 'Wet': 1, 'Muddy': 2}
        weather_map = {'Sunny': 0, 'Cloudy': 1, 'Rainy': 2}
        length_map = {'Short': 0, 'Medium': 1, 'Long': 2}
        surface_map = {'Turf': 0, 'Dirt': 1, 'Synthetic': 2}
        jockey_exp_map = {'Beginner': 0, 'Intermediate': 1, 'Expert': 2}

        inputs = {
            'track': track_map[form_data['track_condition']],
            'weather': weather_map[form_data['weather']],
            'length': length_map[form_data['race_length']],
            'surface': surface_map[form_data['surface']],
            'jockey_exp': jockey_exp_map[form_data['jockey_experience']],
        }

        for name in horse_names:
            horse = {
                'name': name,
                'speed': round(random.uniform(40, 60), 2),
                'stamina': round(random.uniform(30, 50), 2),
                'jockey_rating': round(random.uniform(1, 10), 2),
                'track_encoded': inputs['track'],
                'weather_encoded': inputs['weather'],
                'length_encoded': inputs['length'],
                'surface_encoded': inputs['surface'],
                'jockey_exp_encoded': inputs['jockey_exp'],
            }
            horses.append(horse)

        X = np.array([
            [
                h['speed'], h['stamina'], h['jockey_rating'],
                h['track_encoded'], h['weather_encoded'], h['length_encoded'],
                h['surface_encoded'], h['jockey_exp_encoded']
            ] for h in horses
        ])

        predictions = model.predict(X)

        for idx, h in enumerate(horses):
            h['predicted_class'] = int(predictions[idx])

        predicted_winner = max(horses, key=lambda h: h['predicted_class'])

    return render_template('index.html', horses=horses, winner=predicted_winner, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)
