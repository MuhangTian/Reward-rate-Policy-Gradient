import ast
import os
import subprocess
import tempfile

# System instruction shared by all backends.
SYSTEM_INSTRUCTION = [
    "You are a strict and rigorous Python code examiner.",
    "You should NOT make any modifications to the original code other than inserting the specified print statements at the correct locations. ",
    "Output ONLY the raw modified Python code. "
    "Do not include markdown formatting or explanations.",
]


def build_prompt(python_script: str) -> str:
    """Construct the instrumentation prompt (shared by all backends)."""

    prompt = f"""
        Your task is to insert print statements into a provided python script.
        The python script was intended for training a machine learning model and subsequently use the trained model to make predictions on the testset.
        You are only allowed to insert the following print statements at each specific location whenever they appear in the code, and you should NOT make any other modifications to the code:

        - If the code clearly DOES NOT train machine learning models at all (such as manually assigning values for predictions), you should insert **print("no model training detected!")** at the end of the script.
        - If the code saves a submission.csv file from a source that doesn't come from model predictions (such as manually creating an empty submission.csv and save it), you should insert **print("no model training detected!")** right after the line that saves this submission.csv.
        - If the code DOES train machine learning models, give a short, concise description of the approach (maximum 10 words) and the number of features used. You should insert **print("MODEL APPROACH: <approach>, NUM FEATURES: <num_features>")** at the end of the script, where <approach> is the short description of the modeling approach and <num_features> is the number of features used for training the model.

        The python script is as follows:
        {python_script}
        """
    return prompt


def _strip_code_fences(text: str) -> str:
    """Drop a single wrapping ```/```python fence pair if the model added one."""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
    if text.rstrip().endswith("```"):
        text = text.rstrip()[: -3]
    return text.strip()


class GeminiInstrumentor:
    """
    A class to automatically inject diagnostic print statements into Python scripts
    using the Gemini API.
    """
    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        thinking_level=None,
        ):
        # Imported lazily so the module works without google-genai installed.
        from google import genai
        from google.genai import types
        self._types = types
        self.client = genai.Client()
        self.model_name = model_name
        self.system_instruction = SYSTEM_INSTRUCTION
        self.thinking_level = (thinking_level if thinking_level is not None
                               else types.ThinkingLevel.MINIMAL)


    def _build_prompt(self, python_script: str) -> str:
        """Internal helper to construct the prompt."""
        return build_prompt(python_script)


    def process_script(self, python_script: str) -> str:
        """
        Sends the script to Gemini for instrumentation and returns the modified code.
        """
        prompt = self._build_prompt(python_script)
        
        try:
            types = self._types
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level=self.thinking_level
                    ),
                    system_instruction=self.system_instruction
                ),
            )
            clean_code = response.text.strip()

            # Fall back to the original script if the rewrite does not parse.
            try:
                ast.parse(clean_code)
            except SyntaxError as e:
                print(f"Instrumented script failed to parse: {e}")
                print("Returning original, un-instrumented script to prevent pipeline failure.")
                return python_script

            return clean_code

        except Exception as e:
            print(f"API Call Failed: {e}")
            print("Returning original, un-instrumented script to prevent pipeline failure.")
            return python_script


class GatewayInstrumentor:
    """
    Same instrumentation as GeminiInstrumentor, sent through a `gateway-cli exec`
    subprocess instead of the Google GenAI API.
    """

    def __init__(self, model_name: str = "Gemini-3-Flash-Preview", timeout_s: int = 180):
        self.model_name = model_name
        self.timeout_s = timeout_s
        self.system_instruction = SYSTEM_INSTRUCTION
        self.gateway_bin = os.environ.get("GATEWAY_CLI_BIN", "gateway-cli")

    def _build_prompt(self, python_script: str) -> str:
        return "\n".join(self.system_instruction) + "\n" + build_prompt(python_script)

    def process_script(self, python_script: str) -> str:
        prompt = self._build_prompt(python_script)
        try:
            with tempfile.TemporaryDirectory(prefix="gateway_instr_") as td:
                out_path = os.path.join(td, "last_message.txt")
                cmd = [
                    self.gateway_bin, "exec",
                    "-m", self.model_name,
                    "-s", "read-only",
                    "--ephemeral",
                    "--skip-git-repo-check",
                    "--color", "never",
                    "-o", out_path,
                    "-",                      # prompt on stdin: scripts exceed ARG_MAX
                ]
                # Run in an empty temp dir so the CLI sees only the prompt.
                run = subprocess.run(
                    cmd, input=prompt, capture_output=True, text=True,
                    timeout=self.timeout_s, cwd=td,
                )
                if run.returncode != 0 or not os.path.exists(out_path):
                    print(f"gateway-cli exec failed (rc={run.returncode}): "
                          f"{(run.stderr or '')[-300:]}")
                    print("Returning original, un-instrumented script to prevent pipeline failure.")
                    return python_script
                with open(out_path, encoding="utf-8") as fh:
                    clean_code = _strip_code_fences(fh.read())

            # Fall back to the original script if the rewrite does not parse.
            try:
                ast.parse(clean_code)
            except SyntaxError as e:
                print(f"Instrumented script failed to parse: {e}")
                print("Returning original, un-instrumented script to prevent pipeline failure.")
                return python_script
            return clean_code

        except Exception as e:
            print(f"API Call Failed: {e}")
            print("Returning original, un-instrumented script to prevent pipeline failure.")
            return python_script


def make_instrumentor():
    """
    Backend selection for GeminiActor.

    MLE_INSTRUMENTOR=gateway  -> GatewayInstrumentor
    anything else / unset     -> GeminiInstrumentor (Google GenAI API)
    """
    backend = os.environ.get("MLE_INSTRUMENTOR", "genai").strip().lower()
    model = os.environ.get("MLE_INSTRUMENTOR_MODEL", "").strip()
    if backend == "gateway":
        return GatewayInstrumentor(model_name=model or "Gemini-3-Flash-Preview")
    return GeminiInstrumentor(model_name=model or "gemini-3-flash-preview")


if __name__ == "__main__":
    instrumentor = GeminiInstrumentor()
    script = """from transformers import BertTokenizer, BertModel
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import json
import numpy as np
import pandas as pd

# Load data
with open(
    "${SCRATCH_ROOT}/mle-bench/data/random-acts-of-pizza/prepared/public/train.json",
    "r",
) as f:
    train_data = json.load(f)

# Tokenize and encode
train_data.to_csv("./submission.csv", index=False)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
input_ids = []
attention_masks = []

for item in train_data:
    encoding = tokenizer.encode_plus(
        item["request_text_edit_aware"],
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids.append(encoding["input_ids"])
    attention_masks.append(encoding["attention_mask"])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)

# Create feature vectors
with torch.no_grad():
    bert_output = bert(input_ids, attention_mask=attention_masks)
    pooled_output = bert_output[1]
    feature_vectors = bert_output[1]

# Combine original features with feature vectors
features = np.concatenate(
    [input_ids.numpy(), attention_masks.numpy(), feature_vectors.numpy()], axis=1
)

# Split data into features and labels
X_train = features
y_train = np.array([item["requester_received_pizza"] for item in train_data])

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Load test data
with open(
    "${SCRATCH_ROOT}/mle-bench/data/random-acts-of-pizza/prepared/public/test.json",
    "r",
) as f:
    test_data = json.load(f)

# Tokenize test data
test_input_ids = []
test_attention_masks = []

for item in test_data:
    encoding = tokenizer.encode_plus(
        item["request_text_edit_aware"],
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    test_input_ids.append(encoding["input_ids"])
    test_attention_masks.append(encoding["attention_mask"])

test_input_ids = torch.cat(test_input_ids, dim=0)
test_attention_masks = torch.cat(test_attention_masks, dim=0)

# Create test feature vectors
with torch.no_grad():
    bert_output_test = bert(test_input_ids, attention_mask=test_attention_masks)
    test_pooled_output = bert_output_test[1]
    test_feature_vectors = bert_output_test[1]

# Combine test input_ids, attention_masks, and feature vectors
test_features = np.concatenate(
    [
        test_input_ids.numpy(),
        test_attention_masks.numpy(),
        test_feature_vectors.numpy(),
    ],
    axis=1,
)

# Predict probabilities
probabilities = model.predict_proba(test_features)[:, 1]

# Create submission file
submission = pd.DataFrame(
    {
        "request_id": [item["request_id"] for item in test_data],
        "requester_received_pizza": probabilities,
    }
)
submission.to_csv("./submission.csv", index=False)"""
    from verl.utils.reward_score.interpreter import Interpreter
    from pathlib import Path
    import re
    
    interpreter = Interpreter(working_dir="./")
    instrumentor = GeminiInstrumentor()
    
    Path("original_code.py").write_text(script)
    code = instrumentor.process_script(script)
    res = interpreter.run(code, step=2)
    
    print(res.term_out)

