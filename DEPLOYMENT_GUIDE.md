# Streamlit App Deployment Guide

This guide provides instructions for deploying your Sentiment Analysis App to various cloud platforms.

## Deploying to Streamlit Cloud

1. **Visit Streamlit Cloud**:

   - Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account

2. **Deploy Your App**:

   - Click "New app"
   - Select your repository: `anirudh-pedro/Sentiment-Analysis-App`
   - Set Main file path: `app.py`
   - Click "Deploy"

3. **Environment Setup** (automatically handled by Streamlit Cloud):
   - Your `requirements.txt` will be used to install dependencies
   - The NLTK resources will be downloaded during the first run

## Deploying to Heroku

1. **Prerequisites**:

   - Install Heroku CLI
   - Login to Heroku (`heroku login`)

2. **Create a Procfile**:

   - Create a file named `Procfile` in your project root
   - Add the line: `web: streamlit run --server.port $PORT app.py`

3. **Deploy**:
   ```
   git add .
   git commit -m "Add Heroku deployment files"
   heroku create your-app-name
   git push heroku main
   ```

## Other Deployment Options

### Docker Deployment

1. Create a `Dockerfile`:

   ```Dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .

   # Download NLTK resources
   RUN python setup_nltk.py

   # Generate feature columns
   RUN python generate_feature_columns.py

   EXPOSE 8501

   ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501"]
   ```

2. Build and run:
   ```
   docker build -t sentiment-analyzer .
   docker run -p 8501:8501 sentiment-analyzer
   ```

### AWS/GCP/Azure Deployment

- Use the Docker container approach above
- Deploy to your cloud provider's container service:
  - AWS ECS/Fargate
  - Google Cloud Run
  - Azure Container Instances

## Troubleshooting

1. **Missing NLTK Resources**: If you encounter NLTK resource errors after deployment:

   - SSH into your deployment environment
   - Run `python setup_nltk.py` manually
   - Restart the application

2. **Model File Issues**: If model files are not loading:
   - Make sure the `model` directory exists in your deployment
   - Check that all required pickle files are present
   - Use `fix_dimensions.py` if feature dimension mismatches occur
