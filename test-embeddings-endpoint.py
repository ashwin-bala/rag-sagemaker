import json
import boto3

text1 = "How cute your dog is!"
text2 = "Your dog is so cute."
text3 = "The mitochondria is the powerhouse of the cell."



newline, bold, unbold = '\n', '\033[1m', '\033[0m'
endpoint_name = 'jumpstart-dft-hf-textembedding-bloom-7b1-fp16'


def query_endpoint_with_json_payload(encoded_json):
    client = boto3.client('runtime.sagemaker')
    response = client.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=encoded_json)
    return response


def parse_response_multiple_texts(query_response):
    model_predictions = json.loads(query_response['Body'].read())
    embeddings = model_predictions['embedding']
    return embeddings


payload = {"text_inputs": [text1, text2, text3]}
query_response = query_endpoint_with_json_payload(json.dumps(payload).encode('utf-8'))
embeddings = parse_response_multiple_texts(query_response)
print(embeddings)