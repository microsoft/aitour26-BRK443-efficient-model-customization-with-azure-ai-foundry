"""Reformat AI conversation JSON with document citations and quotes."""

import json
from typing import Dict, List, Any
from .format_documents import extract_documents, reformat_user_documents, reformat_quotes_with_citations


def reformat_ai_conversation(conversation: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[Dict[str, str]]]:
    """
    Reformat an AI conversation to improve document references and quotes.
    
    This function:
    1. Extracts documents from user messages
    2. Reformats <DOCUMENT> tags to numbered sources
    3. Reformats ##begin_quote## markers to backticks with citations
    4. Returns reformatted conversation as JSON
    
    Args:
        conversation: Dictionary containing 'messages' key with list of message objects
        
    Returns:
        Reformatted conversation dictionary
    """
    if not conversation or 'messages' not in conversation:
        return {"messages": []}
    
    # Create a copy to avoid modifying the original
    reformatted_conv = {"messages": []}
    documents_list = []
    
    for message in conversation['messages']:
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        
        reformatted_message = {
            "role": role,
            "content": content
        }
        
        # Process user messages to extract documents and reformat
        if role == 'user' and '<DOCUMENT>' in content:
            # Extract documents for later use
            documents_list = extract_documents(content)
            # Reformat the user message
            reformatted_content = reformat_user_documents(content)
            reformatted_message['content'] = reformatted_content
        
        # Process assistant messages to reformat quotes with citations
        elif role == 'assistant' and '##begin_quote##' in content:
            reformatted_content = reformat_quotes_with_citations(content, documents_list)
            reformatted_message['content'] = reformatted_content
        
        reformatted_conv['messages'].append(reformatted_message)
    
    return reformatted_conv


# Example usage
if __name__ == "__main__":
    test_conversation = {
        "messages": [
            {
                "content": "The following is a conversation with an AI assistant. The assistant is helpful, clever, friendly and gives concise and accurate answers.",
                "role": "system"
            },
            {
                "content": "<DOCUMENT>I have used a few different kinds of sprayers at all different price points, and all have been easy to use and have given \nan impeccable finish. Step 6: Sand and Wipe Down Again\nKILZ 3® PREMIUM Primer is ready for paint in just one hour. After the primer has dried, lightly sand the surfaces once more to ensure a smooth finish.</DOCUMENT>\n<DOCUMENT>High-VOC paints release harmful chemicals that affect people, plants, animals, \nand aquatic life during application and curing. Prioritize low-VOC or zero-VOC formulations for outdoor furniture projects. These environmentally responsible options protect your family's health while \nminimizing ecological impact.</DOCUMENT>\n<DOCUMENT>Ensure adequate ventilation even in outdoor spaces, \nparticularly when using oil-based products. Step 2: Thorough Furniture Cleaning \nDeep cleaning forms the foundation of professional results. For weathered outdoor furniture, use soap, water, and scrub brushes to remove accumulated \ndirt, grime, and surface contaminants. This process eliminates barriers that prevent proper paint adhesion.</DOCUMENT>\n<DOCUMENT>Remember that outdoor-specific paints should remain outdoors.</DOCUMENT>\nWhat type of paint should be prioritized for outdoor furniture projects?",
                "role": "user"
            },
            {
                "content": "Step-by-step reasoning:\n\n1. The question asks what type of paint should be prioritized for outdoor furniture projects.\n2. The context discusses the environmental and health impacts of different types of paint, specifically mentioning high-VOC paints and alternatives.\n3. To answer, I need to identify which paint type is recommended in the context.\n4. The context states: ##begin_quote##High-VOC paints release harmful chemicals that affect people, plants, animals, and aquatic life during application and curing.##end_quote##\n5. This means high-VOC paints are harmful and should be avoided.\n6. The context further states: ##begin_quote##Prioritize low-VOC or zero-VOC formulations for outdoor furniture projects. These environmentally responsible options protect your family's health while minimizing ecological impact.##end_quote##\n7. Therefore, the recommended paint types are low-VOC or zero-VOC formulations.\n\n<ANSWER>: Low-VOC or zero-VOC paint formulations should be prioritized for outdoor furniture projects.",
                "role": "assistant"
            }
        ]
    }
    
    # Reformat and print as JSON
    formatted = reformat_ai_conversation(test_conversation)
    print(json.dumps(formatted, indent=2))