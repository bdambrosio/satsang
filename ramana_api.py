#!/usr/bin/env python3
"""
Ramana Website API Server
==========================

Flask API server for the Ramana Maharshi website.
Provides endpoints for:
- Random Nan_Yar passage
- Query generation with aliveness critic
- Expanded response generation
- Sidebar RAG retrieval from filtered passages
"""

import json
import logging
import random
from pathlib import Path

from flask import Flask, jsonify, request, render_template

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from aliveness_critic import ContemplativeGenerator, LocalCritic
from contemplative_rag import ContemplativeRAGProvider
from filtered_passages_rag import FilteredPassagesRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global state (initialized on startup)
nan_yar_passages = []
generator = None
filtered_passages_rag = None


def load_nan_yar_passages():
    """Load Nan_Yar passages from file."""
    global nan_yar_passages
    
    nan_yar_path = Path(__file__).parent / "ramana" / "nan-yar.txt"
    if not nan_yar_path.exists():
        logger.warning(f"Nan_Yar file not found: {nan_yar_path}")
        nan_yar_passages = []
        return
    
    with open(nan_yar_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newlines or single newlines with empty lines
    # Filter out empty strings
    passages = [
        p.strip() 
        for p in content.split('\n\n') 
        if p.strip()
    ]
    
    # If no double newlines, try single newlines
    if not passages:
        passages = [
            p.strip() 
            for p in content.split('\n') 
            if p.strip()
        ]
    
    nan_yar_passages = passages
    logger.info(f"Loaded {len(nan_yar_passages)} Nan_Yar passages")


def initialize_generator():
    """Initialize ContemplativeGenerator with RAG provider and critic."""
    global generator
    
    try:
        # Initialize critic
        critic = LocalCritic(
            backend="local",
            base_url="http://localhost:5000/v1",
            threshold=6.0,
        )
        
        # Initialize RAG provider
        rag_provider = ContemplativeRAGProvider(
            commentaries_path="./ramana/Commentaries_qa_excert.txt",
            backend="local",
            local_url="http://localhost:5000/v1",
        )
        
        # Initialize generator
        generator = ContemplativeGenerator(
            inference_provider=rag_provider,
            critic=critic,
        )
        
        logger.info("ContemplativeGenerator initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        generator = None


def initialize_filtered_passages_rag():
    """Initialize filtered passages RAG system."""
    global filtered_passages_rag
    
    corpus_path = Path(__file__).parent / "filtered_guten" / "filtered_passages" / "corpus.jsonl"
    
    try:
        filtered_passages_rag = FilteredPassagesRAG(
            corpus_path=str(corpus_path),
            llm_url="http://localhost:5000/v1",
        )
        logger.info("FilteredPassagesRAG initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize FilteredPassagesRAG: {e}")
        filtered_passages_rag = None


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/nan-yar', methods=['GET'])
def get_nan_yar():
    """Get a random Nan_Yar passage."""
    if not nan_yar_passages:
        return jsonify({
            'error': 'No Nan_Yar passages loaded'
        }), 500
    
    passage = random.choice(nan_yar_passages)
    return jsonify({
        'passage': passage
    })


@app.route('/api/query', methods=['POST'])
def handle_query():
    """Handle user query and return response + expanded response."""
    if generator is None:
        return jsonify({
            'error': 'Generator not initialized'
        }), 500
    
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({
            'error': 'Missing query field'
        }), 400
    
    user_query = data['query'].strip()
    if not user_query:
        return jsonify({
            'error': 'Query cannot be empty'
        }), 400
    
    try:
        # Generate response
        result = generator.generate(user_query)
        response_text = result.get('response', '')
        is_silence = result.get('is_silence', False)
        
        if is_silence or not response_text or response_text == '[silence]':
            return jsonify({
                'response': '',
                'expanded_response': '',
                'is_silence': True,
                'message': 'Silence is sometimes the most appropriate response.'
            })
        
        # Generate expanded response
        expanded_response = generator.expand(user_query, response_text)
        if not expanded_response:
            expanded_response = ''
        
        return jsonify({
            'response': response_text,
            'expanded_response': expanded_response,
            'is_silence': False,
        })
    
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        return jsonify({
            'error': f'Generation failed: {str(e)}'
        }), 500


@app.route('/api/retrieve-passages', methods=['POST'])
def retrieve_passages():
    """Retrieve passages from filtered passages corpus."""
    logger.info("Received retrieve-passages request")
    
    if filtered_passages_rag is None:
        logger.warning("FilteredPassagesRAG not initialized")
        return jsonify({
            'error': 'FilteredPassagesRAG not initialized',
            'passages': []
        }), 200  # Return empty list instead of error
    
    data = request.get_json()
    if not data:
        logger.warning("Missing request body in retrieve-passages")
        return jsonify({
            'error': 'Missing request body'
        }), 400
    
    # Extract dialogue components for semantic rewriting
    user_input = data.get('user_input', '')
    response = data.get('response', '')
    expanded_response = data.get('expanded_response', '')
    
    logger.info(f"Retrieve request - user_input length: {len(user_input)}, response length: {len(response)}, expanded_response length: {len(expanded_response)}")
    
    if not user_input and not response:
        logger.info("No user_input or response, returning empty passages")
        return jsonify({
            'passages': []
        })
    
    top_k = data.get('top_k', 5)
    
    try:
        # Rewrite dialogue into semantic threads, then retrieve
        passages = filtered_passages_rag.retrieve_with_rewrite(
            user_input=user_input,
            response=response,
            expanded_response=expanded_response,
            top_k=top_k,
        )
        logger.info(f"Retrieved {len(passages)} passages")
        
        # Format passages for frontend (remove internal fields)
        formatted_passages = []
        for passage in passages:
            formatted_passage = {
                'passage_id': passage.get('passage_id'),
                'text': passage.get('text', ''),
                'author': passage.get('author', 'Unknown'),
                'title': passage.get('title', 'Untitled'),
                'tradition': passage.get('tradition', 'unknown'),
                'themes': passage.get('themes', []),
                'similarity': passage.get('similarity', 0.0),
            }
            formatted_passages.append(formatted_passage)
        
        logger.info(f"Returning {len(formatted_passages)} formatted passages")
        return jsonify({
            'passages': formatted_passages
        })
    
    except Exception as e:
        logger.error(f"Error retrieving passages: {e}", exc_info=True)
        return jsonify({
            'error': f'Retrieval failed: {str(e)}',
            'passages': []
        }), 500


if __name__ == '__main__':
    # Initialize on startup
    # Note: In debug mode, Flask restarts the server which causes initialization to happen twice
    # This is expected behavior - the first initialization is for the main process,
    # the second is for the reloader subprocess
    logger.info("Initializing Ramana API server...")
    load_nan_yar_passages()
    initialize_generator()
    initialize_filtered_passages_rag()
    
    # Run Flask app
    # Set use_reloader=False to prevent double initialization, or keep debug=True for auto-reload
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=True)
