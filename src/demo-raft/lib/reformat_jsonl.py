"""Reformat all conversations in a JSONL file."""

import json
from typing import Optional
from .reformat_conversation import reformat_ai_conversation


def reformat_jsonl_file(input_file: str, output_file: Optional[str] = None) -> None:
    """
    Reformat all AI conversations in a JSONL file.
    
    Reads a JSONL file where each line is a JSON conversation object,
    reformats each conversation using reformat_ai_conversation,
    and writes the reformatted conversations to an output file.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file (if None, adds '_reformatted' suffix)
    """
    if output_file is None:
        # Generate output filename by adding suffix before extension
        base = input_file.rsplit('.', 1)[0] if '.' in input_file else input_file
        output_file = f"{base}_reformatted.jsonl"
    
    reformatted_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                # Skip empty lines
                continue
                
            try:
                # Parse JSON conversation
                conversation = json.loads(line)
                
                # Reformat the conversation
                reformatted = reformat_ai_conversation(conversation)
                
                # Write reformatted conversation as a single line
                json.dump(reformatted, outfile, ensure_ascii=False)
                outfile.write('\n')
                
                reformatted_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                error_count += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                error_count += 1
    
    print(f"Reformatting complete:")
    print(f"  - Successfully reformatted: {reformatted_count} conversations")
    print(f"  - Errors encountered: {error_count}")
    print(f"  - Output written to: {output_file}")


def reformat_jsonl_lines(jsonl_content: str) -> str:
    """
    Reformat all AI conversations in JSONL content string.
    
    Args:
        jsonl_content: String containing JSONL data (one JSON object per line)
        
    Returns:
        String containing reformatted JSONL data
    """
    lines = jsonl_content.strip().split('\n')
    reformatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        try:
            conversation = json.loads(line)
            reformatted = reformat_ai_conversation(conversation)
            reformatted_lines.append(json.dumps(reformatted, ensure_ascii=False))
        except Exception as e:
            print(f"Error processing line: {e}")
            # Optionally keep the original line on error
            reformatted_lines.append(line)
    
    return '\n'.join(reformatted_lines)


# Example usage
if __name__ == "__main__":
    # Example with file paths
    input_path = "conversations.jsonl"
    output_path = "conversations_reformatted.jsonl"
    
    # Uncomment to test with actual files
    # reformat_jsonl_file(input_path, output_path)
    
    # Example with string content
    example_jsonl = """{"messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"<DOCUMENT>Doc 1</DOCUMENT><DOCUMENT>Doc 2</DOCUMENT>Question?"},{"role":"assistant","content":"Answer with ##begin_quote##Doc 1##end_quote##."}]}
{"messages":[{"role":"user","content":"<DOCUMENT>Another doc</DOCUMENT>Another question?"},{"role":"assistant","content":"Response."}]}"""
    
    reformatted = reformat_jsonl_lines(example_jsonl)
    print("Reformatted JSONL:")
    print(reformatted)