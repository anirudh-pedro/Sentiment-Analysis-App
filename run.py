"""
Run Script - Ensures all dependencies are installed and resources are downloaded before starting the app
"""
import os
import sys
import subprocess
import time

def run_app():
    """Run the sentiment analyzer app with proper setup"""
    print("Starting Sentiment Analyzer Setup...")
    
    # Ensure dependencies are installed
    print("\n1. Checking and installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("Please manually run: pip install -r requirements.txt")
        return False
    
    # Setup NLTK resources
    print("\n2. Setting up NLTK resources...")
    try:
        subprocess.run([sys.executable, "setup_nltk.py"], check=True)
        print("✓ NLTK resources set up successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error setting up NLTK resources: {e}")
        print("Please manually run: python setup_nltk.py")
        return False
    
    # Ensure feature columns file exists
    print("\n3. Checking feature columns file...")
    try:
        subprocess.run([sys.executable, "generate_feature_columns.py"], check=True)
        print("✓ Feature columns file verified")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error checking feature columns: {e}")
        return False
    
    # All checks passed, run the app
    print("\n✅ All setup steps completed successfully!")
    print("\n🚀 Starting Sentiment Analyzer App...")
    time.sleep(1)
    
    # Run Streamlit app
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    
    return True

if __name__ == "__main__":
    run_app()
