#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chainlit conversational UI for the Email Generation Assistant.

Non-technical users are guided through a multi-step conversation:
  intent → facts → tone → model → generate → (regenerate | new email)

Run with:
    chainlit run app.py
"""
from __future__ import annotations

import os
from typing import Optional

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# Import after load_dotenv so env vars are available for get_llm()
from src.generator.graph import generator_graph  # noqa: E402


# Constants
TONES: list[str] = [
    "Formal, grateful",
    "Apologetic, professional",
    "Persuasive, confident",
    "Warm, casual",
    "Urgent, technical",
    "Personal, sincere",
    "Formal, concise",
    "Neutral, clear",
    "Firm, professional",
    "Enthusiastic, strategic",
    "Custom…",
]


MODELS: dict[str, str] = {
    "Elephant Alpha":  "openrouter/elephant-alpha",
    "NVIDIA Nemotron": "nvidia/nemotron-3-super-120b-a12b:free",
    "MiniMax M2.5":    "minimax/minimax-m2.5:free",
    "GLM-4.5 Air":     "z-ai/glm-4.5-air:free",
}


_TIMEOUT_USER = 300    # seconds for text input steps
_TIMEOUT_CHOICE = 120  # seconds for button selection steps


# Persistence — DDL for SQLite tables required by SQLAlchemyDataLayer
_CREATE_TABLES_SQL = [
    """CREATE TABLE IF NOT EXISTS users (
        "id" TEXT PRIMARY KEY, "identifier" TEXT NOT NULL UNIQUE,
        "createdAt" TEXT, "metadata" TEXT NOT NULL DEFAULT '{}')""",
    """CREATE TABLE IF NOT EXISTS threads (
        "id" TEXT PRIMARY KEY, "createdAt" TEXT, "name" TEXT,
        "userId" TEXT, "userIdentifier" TEXT, "tags" TEXT,
        "metadata" TEXT NOT NULL DEFAULT '{}')""",
    """CREATE TABLE IF NOT EXISTS steps (
        "id" TEXT PRIMARY KEY, "name" TEXT NOT NULL, "type" TEXT NOT NULL,
        "threadId" TEXT NOT NULL, "parentId" TEXT, "streaming" INTEGER,
        "waitForAnswer" INTEGER, "isError" INTEGER, "metadata" TEXT DEFAULT '{}',
        "tags" TEXT, "input" TEXT, "output" TEXT, "createdAt" TEXT,
        "start" TEXT, "end" TEXT, "generation" TEXT DEFAULT '{}',
        "showInput" TEXT, "language" TEXT, "indent" INTEGER,
        "defaultOpen" INTEGER DEFAULT 0, "autoCollapse" INTEGER DEFAULT 0)""",
    """CREATE TABLE IF NOT EXISTS elements (
        "id" TEXT PRIMARY KEY, "threadId" TEXT, "type" TEXT,
        "chainlitKey" TEXT, "url" TEXT, "objectKey" TEXT, "name" TEXT NOT NULL,
        "props" TEXT DEFAULT '{}', "display" TEXT NOT NULL, "size" TEXT,
        "language" TEXT, "page" INTEGER, "autoPlay" INTEGER, "playerConfig" TEXT,
        "forId" TEXT, "mime" TEXT)""",
    """CREATE TABLE IF NOT EXISTS feedbacks (
        "id" TEXT PRIMARY KEY, "forId" TEXT NOT NULL,
        "value" INTEGER NOT NULL, "comment" TEXT)""",
]


# Persistence hooks
@cl.on_app_startup
async def on_startup() -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine
    db_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///chat_history.db")
    engine = create_async_engine(db_url)
    async with engine.begin() as conn:
        for stmt in _CREATE_TABLES_SQL:
            await conn.execute(text(stmt))
        # Migrate existing steps table — add columns introduced in Chainlit 2.x
        for col, default in [("defaultOpen", "0"), ("autoCollapse", "0")]:
            try:
                await conn.execute(text(f'ALTER TABLE steps ADD COLUMN "{col}" INTEGER DEFAULT {default}'))
            except Exception:
                pass  # column already exists
    await engine.dispose()


@cl.data_layer
def get_data_layer() -> SQLAlchemyDataLayer:
    return SQLAlchemyDataLayer(conninfo=os.getenv("DATABASE_URL", "sqlite+aiosqlite:///chat_history.db"))


@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> Optional[cl.User]:
    expected_user = os.getenv("CHAINLIT_DEFAULT_USER", "admin")
    expected_pass = os.getenv("CHAINLIT_DEFAULT_PASSWORD", "admin")
    if username == expected_user and password == expected_pass:
        return cl.User(identifier=username, metadata={"role": "user"})
    return None


@cl.on_chat_resume
async def on_resume(thread: dict) -> None:
    import json as _json

    # Try metadata first (written by new code on email completion)
    raw_meta = thread.get("metadata") or {}
    if isinstance(raw_meta, str):
        try:
            raw_meta = _json.loads(raw_meta)
        except Exception:
            raw_meta = {}

    subject = raw_meta.get("last_subject", "")
    body = raw_meta.get("last_body", "")
    intent = raw_meta.get("last_intent", "")
    tone = raw_meta.get("last_tone", "")

    # Fallback: scan steps for the last generated email
    if not (subject or body):
        for step in reversed(thread.get("steps", [])):
            output = (step.get("output") or "").strip()
            if output.startswith("### ✉️ Your Email"):
                lines = output.splitlines()
                for line in lines:
                    if line.startswith("**Subject:**"):
                        subject = line.replace("**Subject:**", "").strip()
                body_lines = output.split("---\n\n", 1)
                body = body_lines[1].strip() if len(body_lines) > 1 else ""
                break

    # Render last email
    if subject or body:
        history_content = (
            "**Here is the email you generated in this chat:**\n\n"
            + (f"*Purpose:* {intent}\n" if intent else "")
            + (f"*Tone:* {tone}\n\n" if tone else "\n")
            + f"**Subject:** {subject}\n\n---\n\n{body}"
        )
        await cl.Message(content=history_content).send()
    else:
        await cl.Message(content="Welcome back! No completed email found for this chat.").send()

    await cl.AskActionMessage(
        content="What would you like to do next?",
        actions=[cl.Action(name="new_email", payload={"v": "new"}, label="✉️ Write a new email")],
        timeout=_TIMEOUT_USER,
    ).send()
    await _run_conversation()


# Entry point
@cl.on_chat_start
async def start() -> None:
    await _run_conversation()


# Conversation flow — fully sequential, all in one async function
async def _run_conversation() -> None:
    """Guide the user through intent → facts → tone → model → generate."""

    # Step 1: Intent
    intent_res = await cl.AskUserMessage(
        content=(
            "👋 **What email do you need to write?**\n\n"
            "Describe the purpose in one sentence — for example:\n"
            "- *Follow up after a job interview*\n"
            "- *Request a project deadline extension*\n"
            "- *Pitch a product to a new client*"
        ),
        timeout=_TIMEOUT_USER,
    ).send()

    if not intent_res:
        await cl.Message(content="⏱️ Timed out. Click **New Chat** to start over.").send()
        return

    intent = intent_res["output"].strip()
    await cl.Message(content=f"Got it — **\"{intent}\"**\n\nNow let's add the key facts.").send()

    # Step 2: Facts
    facts: list[str] = []

    while True:
        prompt = (
            "Type your first key fact (a date, name, number, or detail to include):"
            if not facts
            else "Add the next fact:"
        )

        fact_res = await cl.AskUserMessage(content=prompt, timeout=_TIMEOUT_USER).send()

        if not fact_res:
            # timeout — proceed with whatever facts we have
            break

        facts.append(fact_res["output"].strip())

        facts_display = "\n".join(f"- {f}" for f in facts)
        more_res = await cl.AskActionMessage(
            content=f"**Facts added ({len(facts)}):**\n{facts_display}",
            actions=[
                cl.Action(name="add_more", payload={"v": "more"}, label="➕ Add another fact"),
                cl.Action(name="done_facts", payload={"v": "done"}, label="✅ Done — choose tone"),
            ],
            timeout=_TIMEOUT_USER,
        ).send()

        if not more_res or more_res["payload"]["v"] == "done":
            break

    if not facts:
        await cl.Message(content="⚠️ No facts added — I'll generate based on the intent alone.").send()

    # Step 3: Tone
    tone_actions = [
        cl.Action(name=f"tone_{i}", payload={"tone": t}, label=t)
        for i, t in enumerate(TONES)
    ]

    tone_res = await cl.AskActionMessage(
        content="**What tone should the email have?**",
        actions=tone_actions,
        timeout=_TIMEOUT_CHOICE,
    ).send()

    if not tone_res:
        tone = "Professional"

    elif tone_res["payload"]["tone"] == "Custom…":

        custom_res = await cl.AskUserMessage(
            content="Describe the tone (e.g. *Direct but empathetic*, *Enthusiastic and informal*):",
            timeout=_TIMEOUT_CHOICE,
        ).send()

        tone = custom_res["output"].strip() if custom_res else "Professional"
    else:
        tone = tone_res["payload"]["tone"]

    # Step 4: Model
    model_actions = [
        cl.Action(
            name=f"model_{i}",
            payload={"model": model_id, "label": label},
            label=label,
        )
        for i, (label, model_id) in enumerate(MODELS.items())
    ]

    model_res = await cl.AskActionMessage(
        content="**Which AI model should write the email?**\n*(All are free via OpenRouter)*",
        actions=model_actions,
        timeout=_TIMEOUT_CHOICE,
    ).send()

    if model_res:
        model_name = model_res["payload"]["model"]
        model_label = model_res["payload"]["label"]
    else:
        # Default to first model on timeout
        first_label, first_id = next(iter(MODELS.items()))
        model_label, model_name = first_label, first_id

    # Step 5: Generate & loop
    await _generate_and_prompt(intent, facts, tone, model_name, model_label)


async def _generate_and_prompt(intent: str, facts: list[str], tone: str, model_name: str, model_label: str) -> None:
    """Generate the email, display it, then offer Regenerate / New Email."""

    # Show spinner while generating
    thinking = cl.Message(content=f"✍️ Writing your email with **{model_label}**…")
    await thinking.send()

    result = await generator_graph.ainvoke({
        "intent": intent,
        "facts": facts,
        "tone": tone,
        "model_name": model_name,
        "reasoning": None,
        "raw_output": None,
        "subject": None,
        "body": None,
        "error": None,
    })

    # Replace spinner with blank to keep chat clean
    thinking.content = ""
    await thinking.update()

    # Error handling
    if result.get("error"):
        error_msg = result["error"]
        retry_res = await cl.AskActionMessage(
            content=(
                f"⚠️ **Generation failed.**\n\n"
                f"`{error_msg[:200]}`\n\n"
                "This usually means the model is temporarily rate-limited."
            ),
            actions=[
                cl.Action(name="retry", payload={"v": "retry"}, label="🔄 Try again"),
                cl.Action(name="new", payload={"v": "new"}, label="📧 Start over"),
            ],
            timeout=_TIMEOUT_CHOICE,
        ).send()

        if not retry_res or retry_res["payload"]["v"] == "new":
            await _run_conversation()
        else:
            await _generate_and_prompt(intent, facts, tone, model_name, model_label)
        return

    # Display result
    subject = result.get("subject") or ""
    body = result.get("body") or ""
    reasoning = result.get("reasoning") or ""

    result_content = (
        f"### ✉️ Your Email\n\n"
        f"**Subject:** {subject}\n\n"
        f"---\n\n"
        f"{body}"
    )

    elements: list[cl.Text] = []

    if reasoning.strip():
        elements.append(
            cl.Text(
                name="💭 How I approached this email",
                content=reasoning,
                display="side",
            )
        )

    await cl.Message(content=result_content, elements=elements).send()

    # Save the generated email to thread metadata so it can be shown on resume.
    if cl.context.session.thread_id:
        from chainlit.data import get_data_layer
        if dl := get_data_layer():
            await dl.update_thread(
                thread_id=cl.context.session.thread_id,
                metadata={
                    "last_intent": intent,
                    "last_tone": tone,
                    "last_subject": subject,
                    "last_body": body,
                },
            )

    # Post-generation actions
    next_res = await cl.AskActionMessage(
        content="What would you like to do next?",
        actions=[
            cl.Action(name="regenerate", payload={"v": "regen"}, label="🔄 Regenerate"),
            cl.Action(name="new_email", payload={"v": "new"}, label="📧 Write a new email"),
        ],
        timeout=_TIMEOUT_USER,
    ).send()

    if not next_res or next_res["payload"]["v"] == "new":
        await _run_conversation()
    else:
        await _generate_and_prompt(intent, facts, tone, model_name, model_label)
