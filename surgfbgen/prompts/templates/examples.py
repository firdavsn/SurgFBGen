"""Example prompt templates for demonstration purposes."""
from surgfbgen.prompts.base import PromptTemplate, prompt_library

# Example query generation prompt
EXAMPLE_QUERY_GENERATION_TEMPLATE = """\
You are a helpful assistant that generates effective search queries.
Please generate a search query based on the following user question.
The search query should be optimized for retrieving relevant documents from a vector database.

User question: {question}

Search query:"""

example_query_generation_prompt = PromptTemplate(
    template=EXAMPLE_QUERY_GENERATION_TEMPLATE,
    name="example_query_generation",
    description="EXAMPLE: Generates optimized search queries from user questions",
    version="1.0",
    metadata={"type": "search", "model": "general", "example": True},
    parameters={
        "question": {
            "description": "The user's original question",
            "required": True,
            "type": "string"
        }
    }
)

# Example answer generation prompt
EXAMPLE_ANSWER_GENERATION_TEMPLATE = """\
You are a helpful assistant that answers questions based on the provided context.
Please answer the following question using only the information from the context provided.
If the context does not contain enough information to answer the question fully,
acknowledge what you don't know.

Question: {question}

Context:
{context}

Answer:"""

example_answer_generation_prompt = PromptTemplate(
    template=EXAMPLE_ANSWER_GENERATION_TEMPLATE,
    name="example_answer_generation",
    description="EXAMPLE: Generates answers from retrieved documents",
    version="1.0",
    metadata={"type": "generation", "model": "general", "example": True},
    parameters={
        "question": {
            "description": "The user's question to answer",
            "required": True,
            "type": "string"
        },
        "context": {
            "description": "The retrieved documents or context to use for answering",
            "required": True,
            "type": "string"
        }
    }
)

# Example document summarization prompt
EXAMPLE_DOCUMENT_SUMMARIZATION_TEMPLATE = """\
Please provide a concise summary of the following document.
Focus on the main points and key information.

Document:
{document}

Summary:"""

example_document_summarization_prompt = PromptTemplate(
    template=EXAMPLE_DOCUMENT_SUMMARIZATION_TEMPLATE,
    name="example_document_summarization",
    description="EXAMPLE: Summarizes documents into concise form",
    version="1.0",
    metadata={"type": "summarization", "model": "general", "example": True},
    parameters={
        "document": {
            "description": "The document content to summarize",
            "required": True,
            "type": "string"
        }
    }
)

# Example source attribution prompt
EXAMPLE_SOURCE_ATTRIBUTION_TEMPLATE = """\
Based on the following question and retrieved documents,
identify which parts of your answer are attributable to which documents.
Provide source attributions in your response.

Question: {question}

Your draft answer: {draft_answer}

Retrieved documents:
{documents}

Answer with attributions:"""

example_source_attribution_prompt = PromptTemplate(
    template=EXAMPLE_SOURCE_ATTRIBUTION_TEMPLATE,
    name="example_source_attribution",
    description="EXAMPLE: Adds source attributions to generated answers",
    version="1.0",
    metadata={"type": "attribution", "model": "general", "example": True},
    parameters={
        "question": {
            "description": "The original user question",
            "required": True,
            "type": "string"
        },
        "draft_answer": {
            "description": "A draft answer generated without attributions",
            "required": True,
            "type": "string"
        },
        "documents": {
            "description": "The retrieved source documents to attribute from",
            "required": True,
            "type": "string"
        }
    }
)

# For backward compatibility
query_generation_prompt = example_query_generation_prompt
answer_generation_prompt = example_answer_generation_prompt
document_summarization_prompt = example_document_summarization_prompt
source_attribution_prompt = example_source_attribution_prompt

# Register all example prompts with the library
prompt_library.add(example_query_generation_prompt)
prompt_library.add(example_answer_generation_prompt)
prompt_library.add(example_document_summarization_prompt)
prompt_library.add(example_source_attribution_prompt)