"""
Streamlit App Runner
"""
import subprocess
import sys
import os

def run_app():
    """Runs the Streamlit app with all necessary setup"""
    print("Starting Sentiment Analysis App...")
    
    # Ensure all NLTK resources are available
    try:
        print("Setting up NLTK resources...")
        subprocess.run([sys.executable, 'setup_nltk.py'], check=True)
    except subprocess.CalledProcessError:
        print("Warning: NLTK setup encountered issues, but will try to continue")
    
    # Check model compatibility
    try:
        print("Verifying model compatibility...")
        subprocess.run([sys.executable, 'check_model_compatibility.py'], check=True)
    except subprocess.CalledProcessError:
        print("Warning: Model compatibility check failed, but will try to continue")
    
    # Run the streamlit app
    print("Starting Streamlit server...")
    try:
        subprocess.run(['streamlit', 'run', 'app.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        return False
    except FileNotFoundError:
        print("Error: Streamlit not found. Please install it with: pip install streamlit")
        return False
    
    return True

if __name__ == "__main__":
    run_app()
