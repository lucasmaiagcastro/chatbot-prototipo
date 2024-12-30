import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
import gradio as gr

# Função para extrair texto do PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text + "\n\n"
    if not text.strip():
        raise ValueError("O PDF parece não conter texto processável.")
    return text

# Função para criar embeddings com Sentence-Transformers
def create_embeddings(text):
    # Substituímos por um modelo especializado em QA
    model = SentenceTransformer("gtr-t5-large")
    paragraphs = text.split("\n\n")  # Usa parágrafos como unidade
    embeddings = model.encode(paragraphs, convert_to_tensor=True)
    return paragraphs, embeddings, model

# Função para responder perguntas com contexto
def ask_question(question, paragraphs, embeddings, model, context_range=2):
    question_embedding = model.encode(question, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(question_embedding, embeddings)

    # Ordena por similaridade
    sorted_scores = similarity_scores[0].argsort(descending=True)

    # Retorna os `context_range` parágrafos mais similares
    best_matches = sorted_scores[:context_range]
    response = "\n\n".join([paragraphs[idx] for idx in best_matches])

    return response

# Função principal do chatbot
def chatbot(pdf_file, question):
    # Extrai texto do PDF
    text = extract_text_from_pdf(pdf_file.name)
    # Cria embeddings do texto
    paragraphs, embeddings, model = create_embeddings(text)
    # Responde à pergunta
    response = ask_question(question, paragraphs, embeddings, model)
    return response

# Interface Gradio ajustada para respostas longas
interface = gr.Interface(
    fn=chatbot,
    inputs=["file", "text"],
    outputs=gr.Textbox(lines=15, max_lines=50, label="Resposta Completa"),
    title="Chatbot com Hugging Face",
    description="Envie um PDF e pergunte sobre o conteúdo."
)

if __name__ == "__main__":
    interface.launch()
