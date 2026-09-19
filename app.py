import os
import sqlite3
import joblib
import pandas as pd
import streamlit as st


# 1. SQLite Database Setup Function
def init_db():
  conn = sqlite3.connect('predictions.db')
  cursor = conn.cursor()
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand TEXT,
            age INTEGER,
            km_driven INTEGER,
            engine INTEGER,
            mileage REAL,
            owners INTEGER,
            fuel TEXT,
            transmission TEXT,
            predicted_price REAL
        )
    """)
  conn.commit()
  conn.close()


# Database initialize karein
init_db()


# Save prediction to SQLite
def save_to_db(
    brand,
    age,
    km,
    engine,
    mileage,
    owners,
    fuel,
    transmission,
    predicted_price,
):
  conn = sqlite3.connect('predictions.db')
  cursor = conn.cursor()
  cursor.execute(
      """
        INSERT INTO history (brand, age, km_driven, engine, mileage, owners, fuel, transmission, predicted_price)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
      (
          brand,
          age,
          km,
          engine,
          mileage,
          owners,
          fuel,
          transmission,
          predicted_price,
      ),
  )
  conn.commit()
  conn.close()


# UI Configuration
st.set_page_config(
    page_title='Used Car Price Prediction', page_icon='🚗', layout='centered'
)

# Load Artifacts
try:
  model = joblib.load('car_price_model.pkl')
  encoder = joblib.load('encoder.pkl')
except:
  model = joblib.load('model/car_price_model.pkl')
  encoder = joblib.load('model/encoder.pkl')

st.title('🚗 Used Car Price Prediction System')
st.write('Fill the vehicle specifications to estimate market selling price.')

st.markdown('---')

col1, col2 = st.columns(2)

with col1:
  car_brand = st.selectbox(
      'Car Brand',
      [
          'BMW',
          'Mercedes',
          'Audi',
          'Jaguar',
          'Land Rover',
          'Maruti',
          'Hyundai',
          'Honda',
          'Toyota',
          'Ford',
      ],
  )
  car_age = st.number_input('Car Age (Years)', 1, 30, 5)
  km_driven = st.number_input('Kilometers Driven', 1000, 300000, 45000)
  engine = st.number_input('Engine Capacity (CC)', 600, 5000, 1200)

with col2:
  mileage = st.number_input('Mileage (kmpl)', 5.0, 40.0, 18.0)
  owners = st.selectbox('Number of Previous Owners', [1, 2, 3, 4, 5])
  fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG'])
  transmission = st.selectbox('Transmission Type', ['Manual', 'Automatic'])

st.markdown('---')

if st.button('Predict Price', use_container_width=True):
  # Preprocessing
  cat_inputs = pd.DataFrame(
      [[car_brand, fuel, transmission]],
      columns=['car_brand', 'fuel_type', 'transmission'],
  )
  encoded_cat = encoder.transform(cat_inputs)
  encoded_cat_df = pd.DataFrame(
      encoded_cat, columns=encoder.get_feature_names_out()
  )

  num_inputs = pd.DataFrame(
      [[car_age, km_driven, engine, mileage, owners]],
      columns=[
          'car_age',
          'kilometers_driven',
          'engine_capacity',
          'mileage',
          'owners',
      ],
  )

  final_input = pd.concat([num_inputs, encoded_cat_df], axis=1)
  prediction = float(model.predict(final_input)[0])

  if prediction < 0:
    prediction = 50000.0

  # Save to SQLite Database
  save_to_db(
      car_brand,
      car_age,
      km_driven,
      engine,
      mileage,
      owners,
      fuel,
      transmission,
      prediction,
  )

  st.success(f'### 💰 Estimated Selling Price: ₹{prediction:,.2f}')
  st.info('✅ Prediction saved to SQLite Database (`predictions.db`)')

# Database History Viewer in Streamlit Sidebar
st.sidebar.title('📊 Database Options')
if st.sidebar.checkbox('Show Saved Predictions History'):
  st.subheader('Saved History in SQLite (`predictions.db`)')
  conn = sqlite3.connect('predictions.db')
  history_df = pd.read_sql_query('SELECT * FROM history', conn)
  conn.close()
  st.dataframe(history_df)