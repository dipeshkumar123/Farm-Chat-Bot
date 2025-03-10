# Farm-Chat-Bot

An intelligent chatbot system designed to assist farmers with agricultural queries using a hybrid approach of local machine learning and advanced language models.

## Features

- Intelligent query handling with local ML model
- Fallback to advanced language model for complex queries
- Self-learning capabilities through automatic model retraining
- High accuracy answers for common agricultural questions
- Real-time confidence scoring for responses
- Web-based user interface

## Technology Stack

- **Backend**: Python, Flask
- **ML/NLP**: scikit-learn, TF-IDF Vectorization
- **API Integration**: OpenRouter API (DeepSeek Chat model)
- **Frontend**: HTML, CSS, JavaScript
- **Data Storage**: CSV-based dataset management

## Setup Instructions

1. Clone the repository
```bash
git clone https://github.com/dipeshkumar123/Farm-Chat-Bot.git
cd Farm-Chat-Bot
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

5. Start the application
```bash
python app.py
```

6. Access the application at `http://localhost:5000`

## Dataset Structure

The application uses two main datasets:
- `faq_dataset.csv`: Pre-trained Q&A pairs
- `farmer_queries.csv`: Historical farmer queries for training

## Configuration

Key configuration parameters in `app.py`:
- Confidence thresholds for model selection
- Weight distributions for confidence scoring
- API settings for DeepSeek integration

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenRouter API for providing access to DeepSeek Chat model
- Contributors to the agricultural knowledge base
