# app.py

from flask import Flask, request, jsonify
import sqlite3
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("flower_class.keras")

# Configure SQLite database
DATABASE = "my_database.db"


# Function to create database table if not exists
def create_table():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_prediction TEXT
        )
    """
    )
    conn.commit()
    conn.close()


# Create the table when the app starts
create_table()


# API endpoint for image classification
@app.route("/imgs", methods=["POST"])
def classify_image():
    # Assuming you receive the image data in the request
    image_data = request.get_json()["image_data"]

    # Perform inference using the trained model
    prediction = model.predict(tf.constant([image_data]))

    # Store the result in SQLite database
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO results (image_prediction) VALUES (?)", (str(prediction),)
    )
    conn.commit()
    conn.close()

    return jsonify({"prediction": prediction.tolist()})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
