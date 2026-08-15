from qdrant_client import QdrantClient

qdrant_client = QdrantClient(
    url="https://e49740d2-4373-4bde-b467-df00ac24c58b.eu-west-2-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MmJmNjFkMmEtZDQ1OS00ZDYxLWEwMmItMTViNDMzYzIyY2M0In0.sEo-19li9f-L1l91h93SeZaG8t0cDjanbL7KjvwX1bw",
)

print(qdrant_client.get_collections())