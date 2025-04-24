# Email Classification System for Support Team

## Overview
This project implements an email classification system for a support team. It masks personally identifiable information (PII) and payment card information (PCI) in emails before classifying them into predefined categories. The system exposes a REST API for classification.

## Features
- PII and PCI masking using regex and spaCy NER (no LLMs).
- Email classification using a Random Forest model with TF-IDF features.
- FastAPI-based REST API with a POST endpoint for email classification.
- Strict JSON output format with masked entities and original data positions.
- Easy to extend with other classification models or masking methods.

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository.
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download spaCy English model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Training the Model
- Prepare your dataset CSV with columns `email_body` and `category`.
- Run the training script:
  ```bash
  python -c "from models import train_model; train_model('path_to_dataset.csv')"
  ```
- This will save the trained model as `email_classifier.pkl`.

### Running the API
- Start the API server:
  ```bash
  python app.py
  ```
- The API will be available at `http://localhost:8000`.

## API Usage

### Endpoint
- `POST /classify_email`

### Request Body
```json
{
  "email_body": "string containing the email"
}
```

### Response Body
```json
{
  "input_email_body": "string containing the email",
  "list_of_masked_entities": [
    {
      "position": [start_index, end_index],
      "classification": "entity_type",
      "entity": "original_entity_value"
    }
  ],
  "masked_email": "string containing the masked email",
  "category_of_the_email": "string containing the class"
}
```

## Deployment
- The application can be deployed on Hugging Face Spaces or any other cloud platform supporting Python and FastAPI.
- Ensure the model file `email_classifier.pkl` is present in the deployment environment.
- Follow Hugging Face Spaces documentation for deployment.

## Code Quality
- The code follows PEP8 guidelines.
- Proper comments and modular structure for maintainability.

## Contact
For any questions or issues, please open an issue in the repository.
