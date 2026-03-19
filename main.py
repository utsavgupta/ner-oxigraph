import re
from typing import Literal

import boto3
from boto3 import Session
from langchain_aws import ChatBedrockConverse
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from pyoxigraph import DefaultGraph, Literal as RDFLiteral, NamedNode, Quad, Store


SAMPLE_TEXT = """
Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California.
Elon Musk leads Tesla and SpaceX, both headquartered in the United States.
The United Nations held a summit in New York City attended by representatives from France and Germany.
"""

RDF_TYPE = NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
RDFS_LABEL = NamedNode("http://www.w3.org/2000/01/rdf-schema#label")
ENTITY_TYPE = NamedNode("http://example.org/schema#entityType")
ENTITY_CLASS = NamedNode("http://example.org/schema#Entity")


class Entity(BaseModel):
    name: str
    type: Literal["PERSON", "ORGANIZATION", "LOCATION", "OTHER"]
    label: str


class EntityList(BaseModel):
    entities: list[Entity]


def slugify(name: str) -> str:
    slug = re.sub(r"\s+", "_", name.strip())
    return re.sub(r"[^\w\-.]", "", slug)


def extract_entities(text: str) -> list[Entity]:
    session = Session()

    llm = ChatBedrockConverse(
        model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        client=session.client("bedrock-runtime"),
        temperature=0.0,
        max_tokens=1024,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a named entity recognition system. Extract all named entities from the text."),
        ("user", "Extract all named entities from the following text:\n\n{text}"),
    ])

    chain = prompt | llm.with_structured_output(EntityList)
    result: EntityList = chain.invoke({"text": text})
    return result.entities


def load_entities_to_rdf(entities: list[Entity]) -> Store:
    store = Store()
    graph = DefaultGraph()

    for entity in entities:
        uri = NamedNode(f"http://example.org/entity/{slugify(entity.name)}")

        store.add(Quad(uri, RDF_TYPE, ENTITY_CLASS, graph))
        store.add(Quad(uri, RDFS_LABEL, RDFLiteral(entity.label), graph))
        store.add(Quad(uri, ENTITY_TYPE, RDFLiteral(entity.type), graph))

    return store


def query_and_print(store: Store) -> None:
    sparql = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX ex:   <http://example.org/schema#>

    SELECT ?entity ?label ?type
    WHERE {
        ?entity a ex:Entity ;
                rdfs:label ?label ;
                ex:entityType ?type .
    }
    ORDER BY ?type ?label
    """

    results = store.query(sparql)

    print(f"\n{'Entity URI':<55}  {'Label':<25}  {'Type'}")
    print("-" * 90)
    for row in results:
        entity_uri = row["entity"].value
        label = row["label"].value
        etype = row["type"].value
        print(f"{entity_uri:<55}  {label:<25}  {etype}")

    entity_count = len(list(store.query(
        "SELECT ?e WHERE { ?e a <http://example.org/schema#Entity> }"
    )))
    print(f"\nTotal entities: {entity_count}  ({len(store)} quads)")


def main():
    print("Extracting named entities via AWS Bedrock (LangChain)...")
    entities = extract_entities(SAMPLE_TEXT)
    print(f"Extracted {len(entities)} entities.")

    print("\nLoading into Oxigraph RDF store...")
    store = load_entities_to_rdf(entities)

    print("Running SPARQL query to verify:")
    query_and_print(store)


if __name__ == "__main__":
    main()
