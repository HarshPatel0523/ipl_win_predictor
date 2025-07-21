import streamlit as st
import pickle
import pandas as pd

teams = ['Mumbai Indians',
 'Kolkata Knight Riders',
 'Rajasthan Royals',
 'Chennai Super Kings',
 'Sunrisers Hyderabad',
 'Delhi Capitals',
 'Punjab Kings',
 'Lucknow Super Giants',
 'Gujarat Titans',
 'Royal Challengers Bengaluru']

cities = ['Delhi', 'Kolkata', 'Abu Dhabi', 'Chennai', 'Bengaluru',
       'Chandigarh', 'Hyderabad', 'Mumbai', 'Jaipur', 'Dharamsala',
       'Nagpur', 'Centurion', 'East London', 'Cape Town', 'Navi Mumbai',
       'Ahmedabad', 'Dubai', 'Visakhapatnam', 'Ranchi', 'Bangalore',
       'Sharjah', 'Pune', 'Bloemfontein', 'Durban', 'Indore',
       'Johannesburg', 'Guwahati', 'Raipur', 'Lucknow', 'Cuttack',
       'Port Elizabeth', 'Mohali', 'Kimberley']

pipe = pickle.load(open('./pipe.pkl', 'rb'))

st.title("IPL Win Predictor")

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))

with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target', min_value=0, step=1)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0, step=1)

with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1)

with col5:
    wickets = st.number_input('Wickets lost', min_value=0, max_value=10, step=1)

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = (20 - overs) * 6
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
    loss = result[0][0]
    win = result[0][1]
    
    st.success(f"Probability of {batting_team} winning: {win:.2%}")
    st.error(f"Probability of {bowling_team} winning: {loss:.2%}")
