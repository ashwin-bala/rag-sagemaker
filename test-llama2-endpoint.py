import json
import boto3

dialogs = [
    [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
    [
        {"role": "user", "content": "I am going to Paris, what should I see?"},
        {
            "role": "assistant",
            "content": """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
        },
        {"role": "user", "content": "What is so great about #1?"},
    ],
    [
        {"role": "system", "content": "Always answer with Haiku"},
        {"role": "user", "content": "I am going to Paris, what should I see?"},
    ],
    [
        {
            "role": "system",
            "content": "Always answer with emojis",
        },
        {"role": "user", "content": "How to go from Beijing to NY?"},
    ],
]

endpoint_name = "jumpstart-dft-meta-textgeneration-llama-2-7b-f"


def query_endpoint(payload):
    client = boto3.client("sagemaker-runtime")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
        CustomAttributes="accept_eula=true",
    )
    response = response["Body"].read().decode("utf8")
    response = json.loads(response)
    return response

for dialog in dialogs:
    payload = {
        "inputs": [dialog], 
        "parameters": {"max_new_tokens": 256, "top_p": 0.9, "temperature": 0.6}
    }
    result = query_endpoint(payload)[0]
    for msg in dialog:
        print(f"{msg['role'].capitalize()}: {msg['content']}\n")
    print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
    print("\n==================================\n")