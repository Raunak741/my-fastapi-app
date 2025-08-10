Intelligent Document Analysis System
This project is a high-performance, AI-powered query and retrieval system designed to analyze large, unstructured documents and provide accurate, cited answers to complex questions. It was developed for a competitive hackathon, focusing on accuracy, speed, and stability.

The system uses a sophisticated Retrieval-Augmented Generation (RAG) pipeline to handle multiple document formats (PDFs, DOCX, EMLs) and delivers responses through a secure FastAPI endpoint deployed on AWS EC2.

Features
Multi-Document Support: Ingests and processes PDFs, DOCX, and email files.

High Accuracy RAG Pipeline: Uses a FAISS vector database for efficient semantic search and Google's Gemini 1.5 Pro for high-fidelity answer synthesis.

Scalable & Stable: Leverages an on-disk FAISS index to handle documents of any size without memory limitations.

Explainable Answers: Provides not just the answer, but also the source (e.g., page number or section) and a direct quote from the document to support the conclusion.

Secure API: Deployed as a RESTful API using FastAPI with Bearer Token authentication.

Tech Stack
Backend: Python, FastAPI, Gunicorn

AI/ML: Google Gemini 1.5 Pro, FAISS, Retrieval-Augmented Generation (RAG), NumPy

Document Processing: PyMuPDF, python-docx, extract-msg

Cloud/DevOps: AWS EC2, systemd

API Usage
Endpoint
[POST /hackrx/run](http://13.60.253.107:10000/hackrx/run)

Request Body
The API expects a JSON body with the URL of the document and a list of questions.

Sample Request:

{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=...",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?"
    ]
}
![alt text](image.png)


Response Body
The API returns a JSON object containing a list of structured answers.

Sample Response:

{
    "answers": [
        {
            "answer": "The grace period for premium payment is thirty days...",
            "source": "Page 2",
            "quote": "The Grace Period for payment of the premium shall be thirty days."
        },
        {
            "answer": "Pre-existing diseases are covered after 36 months...",
            "source": "Page 9",
            "quote": "Expenses related to the treatment of a Pre-Existing Disease (PED)..."
        }
    ]
}
![alt text](image-1.png)

Setup and Deployment
Local Setup
Clone the repository:

git clone https://github.com/your-username/your-repository.git
cd your-repository

Create a virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Create a .env file with your GEMINI_API_KEY and TEAM_TOKEN.

Run the server:

uvicorn app:app --reload

AWS Deployment
The application is deployed on an AWS EC2 instance and runs as a persistent background service using systemd. The deployment process involves:

Launching an EC2 instance (e.g., t2.micro or larger).

Configuring security groups to allow traffic on ports 22 (SSH) and 10000 (HTTP).

Cloning the repository and setting up the Python environment.

Creating a systemd service file to manage the Gunicorn server process.