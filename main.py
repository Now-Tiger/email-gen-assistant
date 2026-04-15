import os
from dotenv import find_dotenv, load_dotenv
from src.generator.graph import generator_graph

_ = load_dotenv(find_dotenv())

result = generator_graph.invoke(
    {
        "intent": "Follow up after a job interview",
        "facts": [
            "Interview was Monday 14 April at 10am",
            "Role: Senior ML Engineer on Recommendations team",
            "Interviewer: Sarah Chen",
            "Team uses PyTorch",
        ],
        "tone": "Formal, grateful", # also mention how many other tones user can access
        "model_name": os.environ.get("ELEPHANT_ALPHA_LLM_MODEL", ""),
        "reasoning": None,
        "raw_output": None,
        "subject": "Interview follow up for ML Engineer role",
        "body": None,
        "error": None,
    }
)

print("Subject:", result["subject"])
print()
print(result["body"])