"""Convert AI conversation JSON to formatted Markdown document."""

import json
from typing import Dict, List, Any


def conversation_to_markdown(conversation: Dict[str, List[Dict[str, str]]]) -> str:
    """
    Convert an AI conversation dictionary to a nicely formatted Markdown document.
    
    Args:
        conversation: Dictionary containing 'messages' key with list of message objects
        
    Returns:
        Formatted Markdown string representation of the conversation
    """
    if not conversation or 'messages' not in conversation:
        return "*No messages found*"

    messages = conversation['messages']
    
    for i, message in enumerate(messages):
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        
        markdown_lines.append(f"## `{role}`\n")
        # Escape XML tags
        import re
        
        # Pattern to match any XML-like tag (opening, closing, or self-closing)
        pattern = r'(</?[A-Z][A-Z0-9_]*[^>]*>)'
        
        # Replace all tags with escaped versions
        escaped_content = re.sub(pattern, lambda m: f'`{m.group(0)}`', content, flags=re.IGNORECASE)
        
        # Escape ##begin_quote## and ##end_quote## markers
        escaped_content = escaped_content.replace('##begin_quote##', '`##begin_quote##`').replace('##end_quote##', '`##end_quote##`')
        
        markdown_lines.append(f"{escaped_content}\n")
    
    return '\n'.join(markdown_lines)


# Example usage
if __name__ == "__main__":
    # Test with the provided conversation
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
    
    # Convert and print
    markdown = conversation_to_markdown(test_conversation)
    print(markdown)