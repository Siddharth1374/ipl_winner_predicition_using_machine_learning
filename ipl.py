import numpy as np
import pandas as pd
import pickle
import os


def train_and_save_model():

    # Here We Upload the Data 
    match = pd.read_csv('matches.csv', encoding='latin1', on_bad_lines='skip')
    delivery = pd.read_csv('deliveries.csv', encoding='latin1', on_bad_lines='skip')


    delivery['total_runs'] = pd.to_numeric(delivery['total_runs'], errors='coerce')

    total_score_df = delivery.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
    total_score_df = total_score_df[total_score_df['inning'] == 1]

    match_df = match.merge(total_score_df[['match_id', 'total_runs']], left_on='id', right_on='match_id')

    teams = [
        'Sunrisers Hyderabad',
        'Mumbai Indians',
        'Royal Challengers Bangalore',
        'Kolkata Knight Riders',
        'Kings XI Punjab',
        'Chennai Super Kings',
        'Rajasthan Royals',
        'Delhi Capitals'
    ]

    match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
    match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
    match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')

    match_df = match_df[match_df['team1'].isin(teams)]
    match_df = match_df[match_df['team2'].isin(teams)]
    match_df = match_df[match_df['dl_applied'] == 0]
    match_df = match_df[['match_id', 'city', 'winner', 'total_runs']]

    delivery_df = match_df.merge(delivery, on='match_id')
    delivery_df = delivery_df[delivery_df['inning'] == 2]

    delivery_df['total_runs_y'] = pd.to_numeric(delivery_df['total_runs_y'], errors='coerce').fillna(0)
    delivery_df['total_runs_x'] = pd.to_numeric(delivery_df['total_runs_x'], errors='coerce').fillna(0)

    delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()
    delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']
    delivery_df['balls_left'] = 126 - (delivery_df['over'] * 6 + delivery_df['ball'])

    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: x if x == "0" else "1")
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')
    wickets = delivery_df.groupby('match_id')['player_dismissed'].cumsum().values
    delivery_df['wickets'] = 10 - wickets

    delivery_df['crr'] = (delivery_df['current_score'] * 6) / (120 - delivery_df['balls_left'])
    delivery_df['rrr'] = (delivery_df['runs_left'] * 6) / delivery_df['balls_left']

    def result(row):
        return 1 if row['batting_team'] == row['winner'] else 0

    delivery_df['result'] = delivery_df.apply(result, axis=1)

    final_df = delivery_df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left',
                             'wickets', 'total_runs_x', 'crr', 'rrr', 'result']]
    final_df = final_df.sample(final_df.shape[0])
    final_df.dropna(inplace=True)
    final_df = final_df[final_df['balls_left'] != 0]

    X = final_df.iloc[:, :-1]
    y = final_df.iloc[:, -1]

     #  Train Test Split

    from sklearn.model_selection import train_test_split

    # HERE WE USE 80% For Training And 20% For Testing 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # WE use The Logistic Regression Model

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    trf = ColumnTransformer([
        ('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])
    ], remainder='passthrough')

    pipe = Pipeline(steps=[
        ('step1', trf),
        ('step2', LogisticRegression(solver='liblinear'))
    ])

    pipe.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {round(acc * 100, 2)}%")

    pickle.dump(pipe, open('pipe.pkl', 'wb'))
    print("Model saved as pipe.pkl")

    return delivery_df, pipe


#Part -2 : Streamlit 

def run_streamlit_app():
    import streamlit as st

    # PAGE CONFIG 
    st.set_page_config(page_title="IPL Winner Predictor ", page_icon="🏏", layout="centered")

    # HERE WE APPLY CSS FOR SOME GOOD LOOKING 
    st.markdown("""
        <style>
            /* Main background gradient */
            .stApp {
                background: linear-gradient(135deg, #022c1a, #064e3b, #065f46);

                color: white;
            }

            /* Title */
            h1 {
                text-align: center;
                color: #f5c518 !important;
                font-size: 3rem !important;
                text-shadow: 2px 2px 8px rgba(245,197,24,0.4);
                margin-bottom: 10px;
            }

            /* Subtitle / cricket emoji banner */
            .subtitle {
                text-align: center;
                font-size: 1.1rem;
                color: #cccccc;
                margin-bottom: 30px;
            }

            /* Labels */
            label, .stSelectbox label, .stNumberInput label {
                color: #f0f0f0 !important;
                font-weight: 600 !important;
                font-size: 0.95rem !important;
            }

            /* Selectbox & input boxes */
            .stSelectbox > div > div,
            .stNumberInput > div > div > input {
                background-color: rgba(255,255,255,0.08) !important;
                color: white !important;
                border: 1px solid rgba(245,197,24,0.4) !important;
                border-radius: 10px !important;
            }

            /* Predict button */
            .stButton > button {
                background: linear-gradient(90deg, #f5c518, #ff6b35);
                color: #1a1a2e;
                font-weight: bold;
                font-size: 1.1rem;
                border: none;
                border-radius: 12px;
                padding: 12px 40px;
                width: 100%;
                margin-top: 20px;
                cursor: pointer;
                transition: transform 0.2s;
                box-shadow: 0 4px 15px rgba(245,197,24,0.3);
            }
            .stButton > button:hover {
                transform: scale(1.03);
                box-shadow: 0 6px 20px rgba(245,197,24,0.5);
            }

            /* Result cards */
            .result-card {
                background: rgba(255,255,255,0.07);
                border: 1px solid rgba(245,197,24,0.3);
                border-radius: 15px;
                padding: 20px 30px;
                margin: 10px 0;
                text-align: center;
                backdrop-filter: blur(10px);
            }
            .result-card h2 {
                color: #f5c518 !important;
                font-size: 1.8rem !important;
                margin: 0 !important;
            }
            .result-card .team-name {
                color: #ffffff;
                font-size: 1rem;
                opacity: 0.8;
            }

            /* Divider */
            hr {
                border-color: rgba(245,197,24,0.2);
            }

            /* Hide streamlit branding */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    teams = [
        'Sunrisers Hyderabad',
        'Mumbai Indians',
        'Royal Challengers Bangalore',
        'Kolkata Knight Riders',
        'Kings XI Punjab',
        'Chennai Super Kings',
        'Rajasthan Royals',
        'Delhi Capitals'
    ]

    cities = [
        'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
        'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
        'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
        'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
        'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
        'Sharjah', 'Mohali', 'Bengaluru'
    ]

    if not os.path.exists('pipe.pkl'):
        train_and_save_model()

    pipe = pickle.load(open('pipe.pkl', 'rb'))

    # HERE WE DEFINE HEADER 
    st.markdown("<h1>🏏 IPL Winner Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Predict match outcomes using live match data</div>", unsafe_allow_html=True)
    st.markdown("---")

    # HERE WE CHOOSE TEAM SELECTION 
    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox('🏏 Batting Team', sorted(teams))
    with col2:
        bowling_team = st.selectbox('🎯 Bowling Team', sorted(teams))

    selected_city = st.selectbox('📍 Host City', sorted(cities))

    st.markdown("---")

    # THIS TELL ABOUT THE MATCH SITUATION 
    target = st.number_input('🎯 Target Score', min_value=0, step=1)

    col3, col4, col5 = st.columns(3)
    with col3:
        score = st.number_input('📊 Current Score', min_value=0, step=1)
    with col4:
        overs = st.number_input('⏱️ Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
    with col5:
        wickets = st.number_input('❌ Wickets Out', min_value=0, max_value=10, step=1)

    # HERE WE PREDICT 
    if st.button('⚡ Predict Win Probability'):
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_remaining = 10 - wickets
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_remaining],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        result = pipe.predict_proba(input_df)
        loss = round(result[0][0] * 100)
        win = round(result[0][1] * 100)

        st.markdown("---")
        st.markdown("<h3 style='text-align:center; color:#f5c518;'>📊 Win Probability</h3>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
                <div class='result-card'>
                    <div class='team-name'>{batting_team}</div>
                    <h2> {win}%</h2>
                </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
                <div class='result-card'>
                    <div class='team-name'>{bowling_team}</div>
                    <h2> {loss}%</h2>
                </div>
            """, unsafe_allow_html=True)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_and_save_model()
    else:
        run_streamlit_app()