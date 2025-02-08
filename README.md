# Sherlock_QA_SLM

A simple yet powerful Question Answering (QA) system built using DistilBERT, specifically designed to answer questions about Sherlock Holmes stories.

## Features

- Interactive question-answering about Sherlock Holmes
- Built on the DistilBERT model fine-tuned for QA tasks
- Provides confidence scores for answers
- Easy-to-use command-line interface

## Project Structure

```
SLM/
├── README.md
├── data/
│   └── book.txt          # Sherlock Holmes text
├── examples/
│   └── qa_demo.py        # Interactive demo script
├── requirements.txt      # Project dependencies
└── src/
    └── qa_model.py       # Core QA model implementation
```

## GitHub Repository

The complete codebase is available at: [https://github.com/musu2004/Sherlock_QA_SLM](https://github.com/musu2004/Sherlock_QA_SLM)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/musu2004/Sherlock_QA_SLM.git
cd Sherlock_QA_SLM
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the interactive demo:

```bash
python examples/qa_demo.py data/book.txt
```

Example questions you can ask:

1. Where did Sherlock Holmes live?
2. What were Holmes's methods of investigation?
3. Who was Holmes's nemesis?
4. What was Holmes's opinion about emotions?
5. Who was the woman that Holmes respected?
6. What did Holmes observe in his investigations?
7. Where did Holmes's final confrontation with Moriarty take place?
8. How did Dr. Watson help Holmes?
9. What made Holmes a unique detective?
10. What was Holmes's attitude towards love?

You can also ask your own questions about Sherlock Holmes!

## Model Details

The system uses the DistilBERT model (`distilbert-base-cased-distilled-squad`) fine-tuned on SQuAD (Stanford Question Answering Dataset). The model processes text in chunks and provides answers based on the highest confidence score.

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- See requirements.txt for complete list

## License

MIT License
