import os
import typing
from functools import reduce
from operator import itemgetter

from fastapi import APIRouter, Depends, Form, Path, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import chain
from langchain_groq import ChatGroq
from loguru import logger
from sqlalchemy.orm import Session

from config import EMBEDDINGS_PATH, GROQ_API_KEY
from cookies import get_current_user
from embeddings import CachedOpenAIEmbeddings
from models import Chat, ChatMessage, get_db
from utils import render_markdown_safely

chat_model = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
)
router = APIRouter(prefix="/chat")
templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
def chat_page(
        request: Request,
        db: Session = Depends(get_db),
        user_id: int | None = Depends(get_current_user),
):
    if not user_id:
        return RedirectResponse("/login", status_code=303)
    # files = db.query(File).filter(File.user_id == user_id).all()
    # Initialize empty chat history in session
    chats = (
        db.query(Chat)
        .filter(Chat.user_id == user_id)
        .order_by(Chat.created_at.desc())
        .all()
    )

    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "title": "Code Chat",
            # "files": files,
            "chats": chats,
            "chat_history": [],
            "chat_id": -1,
        },
    )


@router.get("/{chat_id}", response_class=HTMLResponse)
def chat_page_with_chat_id(
        request: Request,
        chat_id: int = Path(...),
        db: Session = Depends(get_db),
        user_id: int | None = Depends(get_current_user),
):
    if not user_id:
        return RedirectResponse("/login", status_code=303)

    chats = db.query(ChatMessage).filter(ChatMessage.chat_id == chat_id).all()
    return templates.TemplateResponse(
        "message_list.html",
        {
            "request": request,
            "chat_history": [
                {
                    "type": chat.type,
                    "content": chat.content,
                    "source": chat.source_file,
                    "url": f"/uploads/{user_id}/{chat.source_file}",
                }
                for chat in chats
            ],
            "chat_id": chat_id,
        },
    )


@router.delete("/{chat_id}", response_class=HTMLResponse)
def delete_chat(
        chat_id: int,
        db: Session = Depends(get_db),
        user_id: int = Depends(get_current_user),
):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user_id).first()
    if not chat:
        return ""
    db.delete(chat)
    db.commit()
    return ""


@router.get("/chats")
def chat_list(
        request: Request,
        db: Session = Depends(get_db),
        user_id: int | None = Depends(get_current_user),
):
    if not user_id:
        return RedirectResponse("/login", status_code=303)
    chats = (
        db.query(Chat)
        .filter(Chat.user_id == user_id)
        .order_by(Chat.created_at.desc())
        .all()
    )
    return templates.TemplateResponse(
        "chat_list.html",
        {
            "request": request,
            "title": "Code Chat",
            "chats": chats,
        },
    )


@router.get("/chats/latest")
def chat_list_latest(
        request: Request,
        db: Session = Depends(get_db),
        user_id: int | None = Depends(get_current_user),
):
    if not user_id:
        return RedirectResponse("/login", status_code=303)
    chats = (
        db.query(Chat)
        .filter(Chat.user_id == user_id)
        .order_by(Chat.created_at.desc())
        .limit(1)
        .all()
    )
    return templates.TemplateResponse(
        "chat_list.html",
        {
            "request": request,
            "title": "Code Chat",
            "chats": chats,
        },
    )


from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough



@router.post("/", response_class=HTMLResponse)
def new_chat(
        chat_id: int = Form(...),
        query: str = Form(...),
        db: Session = Depends(get_db),
        user_id: int = Depends(get_current_user),
        request: Request = None,
):
    embeddings_model = CachedOpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    llm_8b = ChatGroq(model="llama3-8b-8192")
    chat = None
    if chat_id != -1:
        chat = (
            db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user_id).first()
        )
    else:
        prompt = ChatPromptTemplate.from_template(
            f""""
            This is the user question : {query}
            based on this generate appropriate title
            RETURN ONLY TITLE
            """
        )
        llm_chain = prompt | llm_8b | StrOutputParser()
        title = llm_chain.invoke({"query": query}).strip('"').strip("'")
        logger.info("Title: {}", title)
        chat = Chat(user_id=user_id, title=title)
        db.add(chat)
        db.commit()
        db.refresh(chat)
    # get latest 5 chats
    chat_history: typing.List[typing.Type[ChatMessage]] = (
        db.query(ChatMessage)
        .filter(ChatMessage.chat_id == chat.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(2)
        .all()
    )

    rephrased_question = ""
    rephrase_prompt = ChatPromptTemplate.from_template("""
    <History>{history}</History>
    <CurrentQuestion>{question}</CurrentQuestion>
    <Task>combine the old questions and new question if they are relevant and create a single new question 
    DONT ADD INFO THAT IS NOT GIVEN and RETURN ONLY THE NEW QUESTION IN A SINGLE STATEMENT AND NOTHING ELSE</Task>""")
    rephrased_chain = rephrase_prompt | llm_8b | StrOutputParser()
    rephrased_question = rephrased_chain.invoke({
        "history":[
            hist.content if len(str(hist.rephrased_question)) ==0 else hist.rephrased_question for hist in chat_history if hist.type=="human"
        ],
        "question":query
    }) if len(chat_history) > 1 else query
    print(rephrased_question)

    embeddings_path = EMBEDDINGS_PATH
    vectorstore = FAISS.load_local(
        embeddings_path, embeddings_model, allow_dangerous_deserialization=True
    )

    file_names = list(set(v.metadata['source'] for _, v in vectorstore.docstore._dict.items()))
    file_names = [os.path.basename(file) for file in file_names]
    file_base_store = FAISS.from_texts(
        file_names,
        embeddings_model,
    )
    print(
        file_base_store.similarity_search(rephrased_question,5)
    )
    file_base_retriever = file_base_store.as_retriever(k=10)
    files = file_base_retriever.invoke(rephrased_question)
    files = [f.page_content for f in files]
    def _split_docs(initial:typing.List, sequence:Document):
        if os.path.basename(sequence.metadata['source']) in files:
            initial[0].append(sequence)
        else:
            initial[1].append(sequence)
        return initial

    docs = reduce(_split_docs,vectorstore.docstore._dict.values(), [[], []])

    filtered_store = FAISS.from_documents(
        docs[0],
        embeddings_model,
    )

    filtered_compliment = FAISS.from_documents(
        docs[1],
        embeddings_model,
    )
    filtered_retriever = filtered_store.as_retriever()
    filtered_compliment = filtered_compliment.as_retriever()
    context = filtered_retriever.invoke(rephrased_question) + filtered_compliment.invoke(rephrased_question)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You Are a Coding Expert and Assistant for a Software Engineer.\nAnswer the question and provide file path with each code snippet u provide based on the following context and chat history:",
            ),
            (
                "human",
                "Context: {context}\nChat History: {chat_history}\nQuestion: {question}",
            ),
        ]
    )

    @chain
    def log(a):
        print(a)
        return a

    @chain
    def get_retriever(store: FAISS):
        return store.as_retriever()

    @chain
    def retrieve(context):
        return

    llm_chain = (
            {
                "chat_history": lambda _: [
                    (
                        HumanMessage(content=f"{message.content}")
                        if message.type == "human"
                        else AIMessage(content=str(message.content))
                    )
                    for message in chat_history
                ],
                "question": RunnablePassthrough(),
                "context":lambda _ : context,
            }
            | prompt
            | chat_model
            | StrOutputParser()
    )

    result = llm_chain.invoke(rephrased_question)
    content = render_markdown_safely(result)
    # Update chat history
    ai_message = {
        "type": "ai",
        "content": content,
        "url": f"#",
    }
    db.add(ChatMessage(chat_id=chat.id, content=query, type="human",rephrased_question=rephrased_question))
    db.add(
        ChatMessage(
            chat_id=chat.id,
            content=content,
            type="ai",
            source_file="",
        )
    )
    db.commit()

    return (
            templates.TemplateResponse(
                "partials/message.html",
                {
                    "request": request,
                    "message": {"type": "human", "content": query},
                    "chat_id": chat.id,
                },
            ).body
            + templates.TemplateResponse(
        "partials/message.html",
        {"request": request, "message": ai_message, "chat_id": chat.id},
    ).body
    )
