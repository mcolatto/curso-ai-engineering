import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from openai import APIError
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent / ".env")

model = ChatOpenAI(model="gpt-6.1-sol", temperature=1)
parser = StrOutputParser()


prompt = ChatPromptTemplate.from_messages([
 ("system", "Eres un experto en {topic}."),
 ("human", "{pregunta}")
])


chain = prompt | model | parser

async def main():
 try:
    resultado = await chain.ainvoke(
        {"topic": "Inteligencia Artificial", "pregunta": "Explícame los beneficios de LCEL en 3 puntos."}
    )
    print(resultado)
 except APIError as e:
    print(f"Error de API: {e}")
  
 
if __name__ == "__main__":
 asyncio.run(main())