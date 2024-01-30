import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')


"""
# Specify download directory for NLTK data for streamlit
nltk.download('stopwords', download_dir='/home/appuser/nltk_data')
nltk.download('vader_lexicon', download_dir='/home/appuser/nltk_data')
nltk.download('punkt', download_dir='/home/appuser/nltk_data', quiet=True)  # Add 'quiet=True' to suppress NLTK download messages
nltk.data.path.append('/home/appuser/nltk_data')
"""
