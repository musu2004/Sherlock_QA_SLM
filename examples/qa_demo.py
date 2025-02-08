import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qa_model import QAModel
from typing import List
import torch

def load_book(file_path: str) -> str:
    """Load book content from a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def split_text(text: str, max_length: int = 500) -> List[str]:
    """Split text into chunks with sentence boundaries."""
    # Split into sentences and clean them
    sentences = [s.strip() + '.' for s in text.replace('\n', ' ').split('.') if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Rough estimate of tokens (words + punctuation)
        sentence_length = len(sentence.split())
        
        if current_length + sentence_length > max_length and current_chunk:
            # Join current chunk and add to chunks
            chunks.append(' '.join(current_chunk))
            # Keep some overlap for context
            overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk[-1:]
            current_chunk = overlap_sentences
            current_length = sum(len(s.split()) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_length += sentence_length
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def answer_question(model: QAModel, question: str, chunks: List[str]) -> dict:
    """Get the best answer from all chunks."""
    best_answer = None
    best_score = float('-inf')
    
    for chunk in chunks:
        result = model.get_answer(question, chunk)
        if result and result['score'] > best_score:
            best_score = result['score']
            best_answer = result
    
    return best_answer

def main():
    # Check if book file is provided
    if len(sys.argv) != 2:
        print("Usage: python qa_demo.py <path_to_book.txt>")
        sys.exit(1)
    
    book_path = sys.argv[1]
    if not os.path.exists(book_path):
        print(f"Error: Book file '{book_path}' not found.")
        sys.exit(1)
    
    # Load the model
    print("Loading QA model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QAModel().to(device)
    
    # Load and preprocess the book
    print("Loading and preprocessing book...")
    book_text = load_book(book_path)
    chunks = split_text(book_text)
    print(f"Split book into {len(chunks)} chunks for processing")
    
    # Interactive QA loop
    print("\nModel ready! Enter your questions about the book (type 'exit' to quit)")
    print("\nExample questions:")
    print("1. Where did Sherlock Holmes live?")
    print("2. What were Holmes's methods of investigation?")
    print("3. Who was Holmes's nemesis?")
    print("4. What was Holmes's opinion about emotions?")
    print("5. Who was the woman that Holmes respected?")
    print("6. What did Holmes observe in his investigations?")
    print("7. Where did Holmes's final confrontation with Moriarty take place?")
    print("8. How did Dr. Watson help Holmes?")
    print("9. What made Holmes a unique detective?")
    print("10. What was Holmes's attitude towards love?\n")
    
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == 'exit':
            break
            
        if not question:
            continue
        
        # Handle numbered questions
        if question.isdigit():
            questions = {
                "1": "Where did Sherlock Holmes live?",
                "2": "What were Holmes's methods of investigation?",
                "3": "Who was Holmes's nemesis?",
                "4": "What was Holmes's opinion about emotions?",
                "5": "Who was the woman that Holmes respected?",
                "6": "What did Holmes observe in his investigations?",
                "7": "Where did Holmes's final confrontation with Moriarty take place?",
                "8": "How did Dr. Watson help Holmes?",
                "9": "What made Holmes a unique detective?",
                "10": "What was Holmes's attitude towards love?"
            }
            if question in questions:
                original_question = question
                question = questions[question]
                print(f"Asking: {question}")
        
        try:
            result = answer_question(model, question, chunks)
            if result:
                print(f"\nAnswer: {result['answer']}")
                # Adjust confidence display based on the score range we're seeing
                if abs(result['score']) < 10:  # Scores are typically small negative or positive numbers
                    confidence = "High" if result['score'] > 5 else "Medium" if result['score'] > 0 else "Low"
                else:  # Larger magnitude scores
                    confidence = "High" if result['score'] > 0 else "Low"
                print(f"Confidence: {confidence}")
            else:
                print("\nI couldn't find a reliable answer to that question. Try rephrasing it or asking something else about the book.")
        except Exception as e:
            print(f"\nError processing question: {str(e)}")

if __name__ == '__main__':
    main()
