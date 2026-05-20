from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class AnalysisRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=True) # links to raw sql auth table
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    patient_name = db.Column(db.String(100), nullable=True, default="Unknown Patient")
    image_path = db.Column(db.String(255), nullable=False)
    patient_symptoms = db.Column(db.Text, nullable=True)
    user_query = db.Column(db.Text, nullable=True)
    cnn_label = db.Column(db.String(50), nullable=True)
    cnn_confidence = db.Column(db.String(50), nullable=True)
    cnn_probability = db.Column(db.Float, nullable=True)
    llm_report = db.Column(db.Text, nullable=True)
    heatmap_path = db.Column(db.String(255), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S") if self.timestamp else None,
            "patient_name": self.patient_name,
            "image_path": self.image_path,
            "patient_symptoms": self.patient_symptoms,
            "user_query": self.user_query,
            "cnn_label": self.cnn_label,
            "cnn_confidence": self.cnn_confidence,
            "cnn_probability": self.cnn_probability,
            "llm_report": self.llm_report,
            "heatmap_path": self.heatmap_path
        }
