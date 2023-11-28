import argparse
from pprint import pprint
from llm_annotator import LLMAnnotator
import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KE")


if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='GPT Annotation Job arguments')

    # Define the arguments you want to accept
    parser.add_argument('--version', type=float, help='Which template version to be used for annotation')
    parser.add_argument('--batch', nargs='+', type=int, help='Discribe the batch to be annotated')
    parser.add_argument('--device', type=str, help='The device where the matcher model will be allocated', default='cpu')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    version = args.version
    batch = args.batch
    annotator = LLMAnnotator(version = version, matcher_device=args.device)
    value = "\n".join([f"{k}: {v}" for k,v in annotator.card.items()])
    annotator.logger.info(f"Template Card:\n{value}")

    # Start annotating
    annotator.generate_labels(batch=batch)