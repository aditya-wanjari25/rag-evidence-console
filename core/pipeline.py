class RAGPipeline:
    def __init__(self, chunker, retriever, generator):
        self.chunker = chunker
        self.retriever = retriever
        self.generator = generator

    def ingest(self, documents):
        for doc in documents:
            chunks = self.chunker.chunk(
                doc["text"],
                metadata={"source": doc["source"]}
            )
            self.retriever.add_chunks(chunks)

    def run(self, query: str):
        retrieved = self.retriever.retrieve(query)
        return self.generator.generate(query, retrieved)
