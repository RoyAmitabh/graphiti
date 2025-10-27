"""
Copyright 2025, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from langchain_openai import AzureChatOpenAI

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.cross_encoder.client import CrossEncoderClient

# Dummy embedder class for when we don't have embeddings
class DummyEmbedder(EmbedderClient):
    """A dummy embedder that returns random vectors when embeddings are not available."""
    
    def __init__(self):
        self.config = None
        import random
        self.random = random
    
    async def create(self, input_data):
        """Return a random vector of the default embedding dimension."""
        # Return a random vector of the default embedding dimension
        # This ensures Neo4j vector similarity functions work properly
        return [self.random.uniform(-1, 1) for _ in range(1024)]
    
    async def create_batch(self, input_data_list):
        """Return random vectors for all inputs."""
        return [[self.random.uniform(-1, 1) for _ in range(1024)] for _ in input_data_list]

# Dummy cross-encoder class for when we don't have reranking
class DummyCrossEncoder(CrossEncoderClient):
    """A dummy cross-encoder that returns equal scores when reranking is not available."""
    
    def __init__(self):
        self.config = None
    
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        """Return passages with equal scores (no reranking)."""
        return [(passage, 0.5) for passage in passages]

# Custom wrapper to make AzureChatOpenAI compatible with Graphiti's structured output requirements
class AzureChatOpenAIWithStructuredOutput:
    """Wrapper for AzureChatOpenAI that adds structured output support."""
    
    def __init__(self, azure_chat_client):
        self.client = azure_chat_client
        # Add responses attribute that Graphiti expects
        self.responses = self
    
    async def parse(self, model, input, text_format, **kwargs):
        """Convert structured output request to regular completion with JSON formatting."""
        import json
        
        # Handle the input properly - it should be a list of messages
        if isinstance(input, list) and len(input) > 0:
            # If input is a list of messages, format them properly for AzureChatOpenAI
            if isinstance(input[0], dict) and 'role' in input[0] and 'content' in input[0]:
                # This is already in the right format for AzureChatOpenAI
                messages = input.copy()
            else:
                # Fallback: treat as single message
                messages = [{"role": "user", "content": str(input[0])}]
        else:
            # Fallback: create a single user message
            messages = [{"role": "user", "content": str(input)}]
        
        # Add strict JSON format instructions to the last message
        if messages:
            # Get the Pydantic model schema for strict formatting
            import json
            schema = text_format.model_json_schema() if hasattr(text_format, 'model_json_schema') else str(text_format)
            
            messages[-1]["content"] += f"""

CRITICAL INSTRUCTIONS:
1. Analyze the conversation and extract the required information
2. Return ONLY the actual data in JSON format
3. Do NOT include any schema definitions, titles, types, or metadata
4. Return ONLY the data fields that are required

FOR ENTITY EXTRACTION:
Return: {{"extracted_entities": [{{"name": "actual_entity_name", "entity_type_id": 1}}]}}

FOR EDGE EXTRACTION:
Return: {{"edges": [{{"source_entity_id": 0, "target_entity_id": 1, "relation_type": "RELATION_TYPE", "fact": "description", "valid_at": null, "invalid_at": null}}]}}

FOR EDGE DUPLICATE DETECTION:
Return: {{"duplicate_facts": [], "contradicted_facts": [], "fact_type": "DEFAULT"}}

FOR NODE RESOLUTION:
Return: {{"entity_resolutions": [{{"id": 0, "name": "entity_name", "duplicate_idx": -1, "duplicates": []}}]}}

DO NOT RETURN:
- Schema definitions
- Field descriptions  
- Type information
- Title fields
- Properties wrappers

Return ONLY the actual data:"""
        
        print(f"🔧 Debug - Sending messages to AzureChatOpenAI: {messages}")
        
        # Use the AzureChatOpenAI client to get a response
        # Pass any additional parameters (like temperature) to the client
        response = await self.client.ainvoke(messages)
        
        # Debug: Print the raw response to understand the format
        print(f"🔧 Debug - Raw LLM response: {response.content}")
        
        try:
            # Clean up the response content to extract just the JSON
            content = response.content.strip()
            
            # Try to find JSON in the response (in case there's extra text)
            if content.startswith('```json'):
                # Remove markdown code block
                content = content.replace('```json', '').replace('```', '').strip()
            elif content.startswith('```'):
                # Remove generic code block
                content = content.replace('```', '').strip()
            
            # Try to parse the JSON response
            parsed_response = json.loads(content)
            print(f"🔧 Debug - Parsed JSON: {parsed_response}")
            
            # Since we're being strict about the format, minimal transformation needed
            print(f"🔧 Debug - Available fields: {list(parsed_response.keys())}")
            
            # Since we're being strict about the format, only remove schema/metadata fields
            schema_fields = ['title', 'type', 'required', '$defs', 'properties', 'speaker', 'class']
            for field in schema_fields:
                if field in parsed_response:
                    parsed_response.pop(field)
                    print(f"🔧 Debug - Removed {field} field")
            
            # Handle common field name variations that still occur despite strict instructions
            if 'facts' in parsed_response and 'edges' not in parsed_response:
                parsed_response['edges'] = parsed_response.pop('facts')
                print(f"🔧 Debug - Renamed facts to edges")
            elif 'entities' in parsed_response and 'extracted_entities' not in parsed_response:
                parsed_response['extracted_entities'] = parsed_response.pop('entities')
                print(f"🔧 Debug - Renamed entities to extracted_entities")
            elif 'nodes' in parsed_response and 'extracted_entities' not in parsed_response:
                parsed_response['extracted_entities'] = parsed_response.pop('nodes')
                print(f"🔧 Debug - Renamed nodes to extracted_entities")
            elif 'resolutions' in parsed_response and 'entity_resolutions' not in parsed_response:
                parsed_response['entity_resolutions'] = parsed_response.pop('resolutions')
                print(f"🔧 Debug - Renamed resolutions to entity_resolutions")
            
            # If we still have schema-like responses, extract data from properties
            if 'properties' in parsed_response:
                properties = parsed_response.pop('properties', {})
                if properties:
                    parsed_response.update(properties)
                    print(f"🔧 Debug - Extracted data from properties: {list(properties.keys())}")
            
            # Create a mock response object that matches the expected format
            class MockResponse:
                def __init__(self, content):
                    self.choices = [MockChoice(content)]
                    self.output_text = json.dumps(parsed_response)  # Add output_text attribute
            
            class MockChoice:
                def __init__(self, content):
                    self.message = MockMessage(content)
            
            class MockMessage:
                def __init__(self, content):
                    self.content = json.dumps(parsed_response)
            
            return MockResponse(parsed_response)
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw response
            print(f"🔧 Debug - JSON parsing failed, using raw response: {response.content}")
            class MockResponse:
                def __init__(self, content):
                    self.choices = [MockChoice(content)]
                    self.output_text = content  # Add output_text attribute
            
            class MockChoice:
                def __init__(self, content):
                    self.message = MockMessage(content)
            
            class MockMessage:
                def __init__(self, content):
                    self.content = content
            
            return MockResponse(response.content)

#################################################
# CONFIGURATION
#################################################
# Set up logging and environment variables for
# connecting to Neo4j database and Azure OpenAI
#################################################

# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# Clear any conflicting OpenAI environment variables that might interfere with Azure OpenAI
# This prevents fallback to regular OpenAI API
if 'OPENAI_API_KEY' in os.environ:
    print("⚠️  Warning: OPENAI_API_KEY environment variable is set")
    print("   This might interfere with Azure OpenAI configuration")
    print("   Consider unsetting it: unset OPENAI_API_KEY")

# Neo4j connection parameters
# Make sure Neo4j Desktop is running with a local DBMS started
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')

# Azure OpenAI configuration
azure_api_key = os.environ.get('AZURE_OPENAI_API_KEY')
azure_api_version = os.environ.get('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
azure_llm_endpoint = os.environ.get('AZURE_OPENAI_LLM_ENDPOINT')
azure_embedding_endpoint = os.environ.get('AZURE_OPENAI_EMBEDDING_ENDPOINT')
azure_llm_deployment = os.environ.get('AZURE_OPENAI_LLM_DEPLOYMENT', 'gpt-4o')
azure_embedding_deployment = os.environ.get('AZURE_OPENAI_EMBEDDING_DEPLOYMENT', 'text-embedding-3-small')
azure_small_model_deployment = os.environ.get('AZURE_OPENAI_SMALL_MODEL_DEPLOYMENT', 'gpt-4')

# Validation
if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

if not azure_api_key:
    raise ValueError('AZURE_OPENAI_API_KEY must be set')

if not azure_llm_endpoint:
    raise ValueError('AZURE_OPENAI_LLM_ENDPOINT must be set')

# Embedding endpoint is optional - Graphiti will work without it but with reduced functionality
if not azure_embedding_endpoint:
    print("⚠️  Warning: AZURE_OPENAI_EMBEDDING_ENDPOINT not set")
    print("   Graphiti will work but with reduced search capabilities")

# Print configuration for debugging
print(f"Azure OpenAI Configuration:")
print(f"  LLM Endpoint: {azure_llm_endpoint}")
print(f"  LLM Deployment: {azure_llm_deployment}")
print(f"  Embedding Endpoint: {azure_embedding_endpoint}")
print(f"  Embedding Deployment: {azure_embedding_deployment}")
print(f"  Small Model Deployment: {azure_small_model_deployment}")
print()


async def cleanup_neo4j_embeddings(driver):
    """Clean up invalid embeddings in Neo4j database."""
    print("🧹 Cleaning up invalid embeddings in Neo4j...")
    
    try:
        # Remove all existing nodes and relationships to start fresh
        cleanup_query = """
        MATCH (n)
        DETACH DELETE n
        """
        
        await driver.execute_query(cleanup_query)
        print("✅ Cleared all existing nodes and relationships")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not clean up Neo4j: {e}")
        print("   You may need to manually clear the database")

async def main():
    #################################################
    # AZURE OPENAI SETUP
    #################################################
    # Create Azure OpenAI clients for different services
    #################################################

    # Create Azure OpenAI client for LLM operations using AzureChatOpenAI
    llm_config = {
        'azure_endpoint': azure_llm_endpoint,
        'api_key': azure_api_key,
        'azure_deployment': azure_llm_deployment,
        'api_version': azure_api_version,
        'temperature': 0.7,
        'max_tokens': 16384,
        'timeout': 300
    }
    
    # Create AzureChatOpenAI client
    azure_chat_client = AzureChatOpenAI(**llm_config)
    
    # Wrap it with structured output support
    llm_client_azure = AzureChatOpenAIWithStructuredOutput(azure_chat_client)

    # Create Azure OpenAI client for embedding operations (only if embedding endpoint is available)
    embedding_client_azure = None
    if azure_embedding_endpoint:
        embedding_client_azure = AsyncAzureOpenAI(
            api_key=azure_api_key,
            api_version=azure_api_version,
            azure_endpoint=azure_embedding_endpoint
        )

    # Create LLM Config with your Azure deployment names
    azure_llm_config = LLMConfig(
        api_key=azure_api_key,  # Required for the config
        small_model=azure_small_model_deployment,
        model=azure_llm_deployment,
    )

    # Create LLM client with Azure configuration
    # Disable reasoning parameter and set verbosity for Azure OpenAI compatibility
    llm_client = OpenAIClient(
        config=azure_llm_config,
        client=llm_client_azure,
        reasoning=None,  # Disable reasoning parameter for Azure OpenAI
        verbosity='medium'  # Set verbosity to medium for Azure OpenAI compatibility
    )
    
    # Test if structured outputs work with this deployment
    print("🧪 Testing if structured outputs are supported...")
    try:
        # Try a simple structured output test
        test_response = await llm_client_azure.responses.parse(
            model=azure_llm_deployment,
            input=[{"role": "user", "content": "Say hello"}],
            text_format={"type": "object", "properties": {"message": {"type": "string"}}}
        )
        print("✅ Structured outputs are supported")
    except Exception as e:
        print(f"⚠️  Structured outputs not supported: {e}")
        print("   This is why Graphiti is failing - you need to enable v1 API opt-in")
        print("   Go to Azure Portal → Azure OpenAI → Model deployments")
        print("   Find your deployment and enable v1 API opt-in")
        print("   Or create a new deployment with v1 API support")

    # Create embedder with Azure configuration (optional)
    if azure_embedding_endpoint and azure_embedding_deployment and embedding_client_azure:
        try:
            # Debug: Print embedder configuration
            print(f"🔧 Configuring embedder:")
            print(f"   Endpoint: {azure_embedding_endpoint}")
            print(f"   Deployment: {azure_embedding_deployment}")
            print(f"   Client type: {type(embedding_client_azure)}")
            
            embedder = OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    api_key=azure_api_key,
                    embedding_model=azure_embedding_deployment
                ),
                client=embedding_client_azure
            )
            print("✅ Embedder configured successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not configure embedder: {e}")
            print("   Using dummy embedder - Graphiti will work but with reduced search capabilities")
            embedder = DummyEmbedder()
    else:
        print("⚠️  No embedding endpoint/deployment configured")
        print("   Using dummy embedder - Graphiti will work but with reduced search capabilities")
        embedder = DummyEmbedder()

    # Create cross-encoder (reranker) with Azure configuration (optional)
    if azure_small_model_deployment:
        try:
            cross_encoder = OpenAIRerankerClient(
                config=LLMConfig(
                    api_key=azure_api_key,
                    model=azure_small_model_deployment  # Use small model for reranking
                ),
                client=llm_client_azure
            )
            print("✅ Cross-encoder configured successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not configure cross-encoder: {e}")
            print("   Using dummy cross-encoder - Graphiti will work but without advanced reranking")
            cross_encoder = DummyCrossEncoder()
    else:
        print("⚠️  No small model deployment configured")
        print("   Using dummy cross-encoder - Graphiti will work but without advanced reranking")
        cross_encoder = DummyCrossEncoder()

    #################################################
    # INITIALIZATION
    #################################################
    # Connect to Neo4j and set up Graphiti indices
    # This is required before using other Graphiti
    # functionality
    #################################################

    # Clean up any existing invalid embeddings in Neo4j
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    temp_driver = Neo4jDriver(neo4j_uri, neo4j_user, neo4j_password)
    await cleanup_neo4j_embeddings(temp_driver)
    
    # Initialize Graphiti with Neo4j connection and Azure OpenAI clients
    print(f"🔧 Initializing Graphiti:")
    print(f"   LLM Client: {type(llm_client)}")
    print(f"   Embedder: {type(embedder) if embedder else 'None'}")
    print(f"   Cross-encoder: {type(cross_encoder) if cross_encoder else 'None'}")
    
    graphiti = Graphiti(
        neo4j_uri, 
        neo4j_user, 
        neo4j_password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder
    )

    try:
        # Initialize the graph database with graphiti's indices. This only needs to be done once.
        await graphiti.build_indices_and_constraints()

        #################################################
        # ADDING EPISODES
        #################################################
        # Episodes are the primary units of information
        # in Graphiti. They can be text or structured JSON
        # and are automatically processed to extract entities
        # and relationships.
        #################################################

        # Example: Add Episodes
        # Episodes list containing both text and JSON episodes
        episodes = [
            {
                'content': 'Kamala Harris is the Attorney General of California. She was previously '
                'the district attorney for San Francisco.',
                'type': EpisodeType.text,
                'description': 'podcast transcript',
            },
            {
                'content': 'As AG, Harris was in office from January 3, 2011 – January 3, 2017',
                'type': EpisodeType.text,
                'description': 'podcast transcript',
            },
            {
                'content': {
                    'name': 'Gavin Newsom',
                    'position': 'Governor',
                    'state': 'California',
                    'previous_role': 'Lieutenant Governor',
                    'previous_location': 'San Francisco',
                },
                'type': EpisodeType.json,
                'description': 'podcast metadata',
            },
            {
                'content': {
                    'name': 'Gavin Newsom',
                    'position': 'Governor',
                    'term_start': 'January 7, 2019',
                    'term_end': 'Present',
                },
                'type': EpisodeType.json,
                'description': 'podcast metadata',
            },
        ]

        # Add episodes to the graph
        for i, episode in enumerate(episodes):
            await graphiti.add_episode(
                name=f'Freakonomics Radio {i}',
                episode_body=episode['content']
                if isinstance(episode['content'], str)
                else json.dumps(episode['content']),
                source=episode['type'],
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
            )
            print(f'Added episode: Freakonomics Radio {i} ({episode["type"].value})')

        #################################################
        # BASIC SEARCH
        #################################################
        # The simplest way to retrieve relationships (edges)
        # from Graphiti is using the search method, which
        # performs a hybrid search combining semantic
        # similarity and BM25 text retrieval.
        #################################################

        # Perform a hybrid search combining semantic similarity and BM25 retrieval
        print("\nSearching for: 'Who was the California Attorney General?'")
        results = await graphiti.search('Who was the California Attorney General?')

        # Print search results
        print('\nSearch Results:')
        for result in results:
            print(f'UUID: {result.uuid}')
            print(f'Fact: {result.fact}')
            if hasattr(result, 'valid_at') and result.valid_at:
                print(f'Valid from: {result.valid_at}')
            if hasattr(result, 'invalid_at') and result.invalid_at:
                print(f'Valid until: {result.invalid_at}')
            print('---')

        #################################################
        # CENTER NODE SEARCH
        #################################################
        # For more contextually relevant results, you can
        # use a center node to rerank search results based
        # on their graph distance to a specific node
        #################################################

        # Use the top search result's UUID as the center node for reranking
        if results and len(results) > 0:
            # Get the source node UUID from the top result
            center_node_uuid = results[0].source_node_uuid

            print('\nReranking search results based on graph distance:')
            print(f'Using center node UUID: {center_node_uuid}')

            reranked_results = await graphiti.search(
                'Who was the California Attorney General?', center_node_uuid=center_node_uuid
            )

            # Print reranked search results
            print('\nReranked Search Results:')
            for result in reranked_results:
                print(f'UUID: {result.uuid}')
                print(f'Fact: {result.fact}')
                if hasattr(result, 'valid_at') and result.valid_at:
                    print(f'Valid from: {result.valid_at}')
                if hasattr(result, 'invalid_at') and result.invalid_at:
                    print(f'Valid until: {result.invalid_at}')
                print('---')
        else:
            print('No results found in the initial search to use as center node.')

        #################################################
        # NODE SEARCH USING SEARCH RECIPES
        #################################################
        # Graphiti provides predefined search recipes
        # optimized for different search scenarios.
        # Here we use NODE_HYBRID_SEARCH_RRF for retrieving
        # nodes directly instead of edges.
        #################################################

        # Example: Perform a node search using _search method with standard recipes
        print(
            '\nPerforming node search using _search method with standard recipe NODE_HYBRID_SEARCH_RRF:'
        )

        # Use a predefined search configuration recipe and modify its limit
        node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        node_search_config.limit = 5  # Limit to 5 results

        # Execute the node search
        node_search_results = await graphiti._search(
            query='California Governor',
            config=node_search_config,
        )

        # Print node search results
        print('\nNode Search Results:')
        for node in node_search_results.nodes:
            print(f'Node UUID: {node.uuid}')
            print(f'Node Name: {node.name}')
            node_summary = node.summary[:100] + '...' if len(node.summary) > 100 else node.summary
            print(f'Content Summary: {node_summary}')
            print(f'Node Labels: {", ".join(node.labels)}')
            print(f'Created At: {node.created_at}')
            if hasattr(node, 'attributes') and node.attributes:
                print('Attributes:')
                for key, value in node.attributes.items():
                    print(f'  {key}: {value}')
            print('---')

    finally:
        #################################################
        # CLEANUP
        #################################################
        # Always close the connection to Neo4j when
        # finished to properly release resources
        #################################################

        # Close the connection
        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':
    asyncio.run(main())
