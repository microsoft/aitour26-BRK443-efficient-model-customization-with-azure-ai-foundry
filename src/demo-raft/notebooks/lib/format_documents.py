"""Format document tags in user messages."""

import re


def reformat_user_documents(message: str) -> str:
    """
    Reformat user message with <DOCUMENT> tags into numbered sources format.
    
    Args:
        message: User message containing <DOCUMENT> tags
        
    Returns:
        Reformatted message with numbered sources
    """
    # Pattern to match document tags and their content
    pattern = r'<DOCUMENT>(.*?)</DOCUMENT>'
    
    # Find all documents
    documents = re.findall(pattern, message, re.DOTALL)
    
    if not documents:
        # No documents found, return as is
        return message
    
    # Extract the question (everything after the last </DOCUMENT>)
    parts = re.split(r'</DOCUMENT>', message)
    question = parts[-1].strip() if len(parts) > 1 else ''
    
    # Build formatted output
    formatted_lines = ["Sources:"]
    
    for i, doc in enumerate(documents, 1):
        # Clean up the document content (remove extra whitespace)
        cleaned_doc = doc.strip()
        formatted_lines.append(f"[{i}] {cleaned_doc}")
    
    # Add the question at the end
    if question:
        formatted_lines.append("")  # Empty line before question
        formatted_lines.append(question)
    
    return '\n'.join(formatted_lines)


# Example usage
if __name__ == "__main__":
    test_message = """<DOCUMENT>I have used a few different kinds of sprayers at all different price points, and all have been easy to use and have given 
an impeccable finish. Step 6: Sand and Wipe Down Again
KILZ 3® PREMIUM Primer is ready for paint in just one hour. After the primer has dried, lightly sand the surfaces once more to ensure a smooth finish.</DOCUMENT>
<DOCUMENT>High-VOC paints release harmful chemicals that affect people, plants, animals, 
and aquatic life during application and curing. Prioritize low-VOC or zero-VOC formulations for outdoor furniture projects. These environmentally responsible options protect your family's health while 
minimizing ecological impact.</DOCUMENT>
<DOCUMENT>Ensure adequate ventilation even in outdoor spaces, 
particularly when using oil-based products. Step 2: Thorough Furniture Cleaning 
Deep cleaning forms the foundation of professional results. For weathered outdoor furniture, use soap, water, and scrub brushes to remove accumulated 
dirt, grime, and surface contaminants. This process eliminates barriers that prevent proper paint adhesion.</DOCUMENT>
<DOCUMENT>Remember that outdoor-specific paints should remain outdoors.</DOCUMENT>
What type of paint should be prioritized for outdoor furniture projects?"""
    
    formatted = reformat_user_documents(test_message)
    print(formatted)