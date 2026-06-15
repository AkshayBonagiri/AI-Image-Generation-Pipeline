from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from io import BytesIO
import base64
import os

load_dotenv()


# MODELS


model1 = ChatGroq(model="qwen/qwen3-32b")
model2 = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")


# STRUCTURED OUTPUT


class classify(BaseModel):
    score: int = Field(
        description="Score out of 100 representing image quality and prompt alignment."
    )
    feedback: str = Field(
        description="Feedback for improving the prompt if needed."
    )

model2_structured = model2.with_structured_output(classify)


# IMAGE MODEL


client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HF_TOKEN")
)

IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"


# CONFIG


MAX_RETRIES = 3
APPROVAL_THRESHOLD = 90


# STATE


class imageState(TypedDict):
    query: str

    # Per-slot optimized prompts
    optimized_query1: str
    optimized_query2: str
    optimized_query3: str

    # Per-slot images
    image1: object
    image2: object
    image3: object

    # Per-slot scores & feedback
    score1: int
    score2: int
    score3: int

    feedback1: str
    feedback2: str
    feedback3: str

    # Which slots still need work: e.g. [1, 2, 3] → [2, 3] → []
    pending_slots: List[int]

    retry_count: int
    category: str


# NODE 1: PROMPT OPTIMIZATION (runs for each pending slot)


def clear_instruction(state: imageState):

    query = state["query"]
    updates = {}

    for slot in state["pending_slots"]:

        feedback = state.get(f"feedback{slot}", "")                         # get is a safer way to get if there is no key then "" is returned
        previous_prompt = state.get(f"optimized_query{slot}", "")

        if feedback:
            user_prompt = f"""
            Original Query: {query}

            Previous Optimized Prompt: {previous_prompt}

            Feedback: {feedback}

            Improve the prompt using the feedback.
            Return ONLY the improved prompt.
        """
        else:
            user_prompt = f"""
                Optimize this image prompt:
                {query}

                Return ONLY the optimized prompt.
            """

        messages = [
            SystemMessage(content="""
                You are an expert prompt engineer specialized in text-to-image generation.

                Guidelines:
                - Add lighting details
                - Add atmosphere and mood
                - Add camera composition
                - Add color palette
                - Add textures
                - Add art style references
                - Add quality boosters

                Keep under 200 words. Return ONLY the optimized prompt.
                """),
            HumanMessage(content=user_prompt)
        ]

        result = model1.invoke(messages)
        updates[f"optimized_query{slot}"] = result.content

    return updates              # updates is a dict returning the optimized queries for the three images


# NODE 2: IMAGE GENERATION (runs for each pending slot)


def image_generation(state: imageState):

    updates = {}

    for slot in state["pending_slots"]:

        optimized_query = state[f"optimized_query{slot}"]

        print(f"\nGenerating image {slot}...")
        print("Prompt:", optimized_query)

        image = client.text_to_image(
            optimized_query,
            model=IMAGE_MODEL
        )

        updates[f"image{slot}"] = image

    return updates


# NODE 3: EVALUATION (runs for each pending slot)


def evaluate(state: imageState):

    query = state["query"]
    updates = {}
    still_pending = []

    for slot in state["pending_slots"]:

        image = state[f"image{slot}"]

        # Convert PIL image to Base64  => the evaluating model cant understand png
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        messages = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": f"""
                        You are an expert image evaluator.

                        Original Prompt: {query}

                        Evaluate the generated image based on:
                        1. Prompt alignment
                        2. Visual quality
                        3. Composition
                        4. Presence of key elements
                        5. Overall effectiveness

                        Return:
                        - score (0-100)
                        - feedback

                        A score above {APPROVAL_THRESHOLD} means approved.
                        """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}"
                        }
                    }
                ]
            )
        ]

        result = model2_structured.invoke(messages)

        category = "Approved" if result.score > APPROVAL_THRESHOLD else "Not Approved"

        print("\n" + "=" * 50)
        print(f"Slot {slot} | Attempt {state.get('retry_count', 0) + 1}/{MAX_RETRIES}")
        print(f"Score: {result.score} | Category: {category}")
        print(f"Feedback: {result.feedback}")
        print("=" * 50)

        updates[f"score{slot}"] = result.score
        updates[f"feedback{slot}"] = result.feedback

        if result.score <= APPROVAL_THRESHOLD:
            still_pending.append(slot)

    # Update pending_slots to only the ones that failed
    updates["pending_slots"] = still_pending
    updates["retry_count"] = state.get("retry_count", 0) + 1

    return updates


# ROUTER


def route_image(state: imageState):

    # All three approved
    if not state["pending_slots"]:
        print("\nAll 3 images approved!")
        return "approved"

    # Max retries hit
    if state["retry_count"] >= MAX_RETRIES:
        print(f"\nMax retries ({MAX_RETRIES}) reached.")
        return "max_retries"

    print(f"\nRetrying slots: {state['pending_slots']}")
    return "retry"


# GRAPH


graph = StateGraph(imageState)

graph.add_node("clear_instruction", clear_instruction)
graph.add_node("image_generation", image_generation)
graph.add_node("evaluate", evaluate)

graph.add_edge(START, "clear_instruction")
graph.add_edge("clear_instruction", "image_generation")
graph.add_edge("image_generation", "evaluate")

graph.add_conditional_edges(
    "evaluate",
    route_image,
    {
        "approved": END,
        "retry": "clear_instruction",
        "max_retries": END
    }
)

workflow = graph.compile()


# RUN


query = "A volcanic eruption"

initial_state = {
    "query": query,
    "retry_count": 0,
    "pending_slots": [1, 2, 3],   # all three start as pending
}

final_state = workflow.invoke(initial_state)


# RESULTS


print("\n")
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"Total Attempts: {final_state['retry_count']}")

for slot in [1, 2, 3]:
    score = final_state.get(f"score{slot}", "N/A")
    status = "✅ Approved" if isinstance(score, int) and score > APPROVAL_THRESHOLD else "❌ Not Approved"
    print(f"\nImage {slot}: Score={score} | {status}")
    print(f"  Prompt: {final_state.get(f'optimized_query{slot}', '')[:80]}...")
    final_state[f"image{slot}"].save(f"output_{slot}.png")
    print(f"  Saved as output_{slot}.png")