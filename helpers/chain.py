
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_groq import ChatGroq

load_dotenv()

SYSTEM = (
    "You are a helpful assistant that answers strictly using the provided context.\n"
    "If the answer is not present, say you don't know. Be concise."
)

QA_PROMPT = PromptTemplate(
    template=(
        "Answer the QUESTION using only the CONTEXT (movie scripts/subtitles).\n"
        "If helpful, quote short snippets and include timestamps if present.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {question}\n"
    ),
    input_variables=["context", "question"],
)

FUNNY_PROMPT = PromptTemplate(
    template=(
        "You are given CONTEXT from movie subtitles/scripts.\n"
        "Extract a bullet list of the **funniest quotes** (witty one-liners, jokes, sarcasm).\n"
        "For each item, include: `\n- \"quote\" — (Character if obvious) [timestamp if provided] (Movie)`\n"
        "Only use what's in context. If none found, say so.\n\n"
        "CONTEXT:\n{context}\n\n"
        "HINT: {question}\n"
    ),
    input_variables=["context", "question"],
)


def _groq():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.2,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
    )


def _run_chain(retriever, question: str, scope_movie: Optional[str], prompt: PromptTemplate):
    # scoped retrieve
    docs: List[Document] = retriever.get_relevant_documents(question)
    if scope_movie:
        docs = [d for d in docs if (d.metadata or {}).get("movie") == scope_movie]
    context_text = "\n\n".join(d.page_content for d in docs)
    messages = [("system", SYSTEM), ("user", prompt.format(context=context_text, question=question))]
    model = _groq()
    resp = model.invoke(messages)
    return {"answer": resp.content, "contexts": docs}


def build_qa_chain(retriever):
    class Chain:
        def invoke(self, inputs: Dict[str, Any]):
            return _run_chain(retriever, inputs["question"], inputs.get("scope_movie"), QA_PROMPT)
    return Chain()


def build_funny_quotes_chain(retriever):
    class Chain:
        def invoke(self, inputs: Dict[str, Any]):
            return _run_chain(retriever, inputs["question"], inputs.get("scope_movie"), FUNNY_PROMPT)
    return Chain()
