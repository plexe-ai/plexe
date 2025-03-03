# smolmodels

A docker container for creating, managing, and serving machine learning models using the [smolmodels](https://github.com/plexe-ai/smolmodels) library.

> **Warning**: This is an early alpha release. Expect bugs and limitations. More features are coming in future releases!

## Overview

smolmodels provides a comprehensive API-driven solution for building, managing, and deploying machine learning models with minimal code. It transforms the capabilities of smolmodels into a platform that supports:

- Natural language model definition
- Automated model training and evaluation
- RESTful prediction endpoints
- Simple model management

## Quick Start

### Prerequisites

- Docker and Docker Compose
- An API key for an LLM provider (OpenAI, Anthropic, etc.)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/plexe-ai/smolmodels.git
cd docker
```

2. Configure your environment:
```bash
cp .env.example .env
# Edit .env with your LLM provider API key
```

3. Start the services:
```bash
docker-compose up -d
```

4. The API is now available at http://localhost:8000

### Create Your First Model

```bash
curl -X POST http://localhost:8000/models \
  -H "Content-Type: application/json" \
  -d '{
    "intent": "Predict heart attack risk",
    "input_schema": {
      "age": "int",
      "cholesterol": "float",
      "exercise": "bool"
    },
    "output_schema": {
      "heart_attack": "bool"
    }
  }'
```

### Make a Prediction

```bash
curl -X POST http://localhost:8000/predictions/models/{model_id} \
  -H "Content-Type: application/json" \
  -d '{"data": {"age": 65, "cholesterol": 280.0, "exercise": false}}'
```

> **Warning**: This is an early alpha release. Expect bugs and limitations. More features are coming in future releases!

## Architecture

```mermaid
graph TD
    A[API Layer] -->|Jobs| B[Queue]
    B --> C[Workers]
    C --> D[Storage]
    A -.->|Predictions| E[Model Cache]
    C -.->|Save Models| E
    C -.->|Update Status| F[Metadata Store]
    A -.->|Query Status| F
```

### Components

- **API Layer**: FastAPI-powered REST interface for model management and predictions
- **Queue**: Redis-based job queue for distributed processing
- **Workers**: Python workers for model training and evaluation
- **Storage**: MongoDB for model metadata and job information
- **Model Cache**: File-based storage for trained models

## API Reference

### Models

- `POST /models`: Create a new model
- `GET /models/{model_id}`: Get model metadata
- `GET /models`: List all models

### Predictions

- `POST /predictions/models/{model_id}`: Make a prediction

### Jobs

- `GET /jobs/{job_id}`: Get job status
