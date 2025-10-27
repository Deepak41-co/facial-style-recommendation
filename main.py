from flask import Flask, send_from_directory
from augmented import app as flask_app
import os

@flask_app.route('/')
def serve_frontend():
    return send_from_directory('.', 'front.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    flask_app.run(host="0.0.0.0", port=port)
