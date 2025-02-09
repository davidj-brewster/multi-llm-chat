import os
import streamlit as st
from asyncio import run
from typing import List, Dict, Optional

from ai_battle import ConversationManager, save_conversation, logger

class ExtendedConversationManager(ConversationManager):
    """
    Extends the ConversationManager to handle:
      - Multiple AI agents
      - A human moderator who can insert messages at any time
    """

    async def run_3_actor_conversation(
        self,
        roles_and_models: List[str],
        system_instructions: Dict[str, str],
        max_rounds: int,
    ) -> List[Dict[str, str]]:
        """
        Example: Three actors:
        1) Claude model as 'user' (AI playing human)
        2) Gemini model as assistant #1
        3) OpenAI model as assistant #2
        Optional: real human moderator can jump in at any time.
        """
        self.conversation_history.clear()
        required_models = ["claude", "gemini", "openai"]
        for model in required_models:
            if not self._get_client(model):
                logger.warning(f"Need {', '.join(required_models)} for 3-actor scenario.")
                return self.conversation_history

        # Start conversation with initial topic
        self.conversation_history.append({"role": "system", "content": "Starting multi-agent conversation"})
        
        for r in range(max_rounds):
            # Get clients for this round
            claude_client = self._get_client("claude")
            gemini_client = self._get_client("gemini")
            openai_client = self._get_client("openai")
            
            if not all([claude_client, gemini_client, openai_client]):
                logger.error("Missing required clients for 3-actor conversation")
                return self.conversation_history

            # Use Claude as 'user' role with human-like instructions
            claude_resp = await self.run_conversation_turn(
                prompt="Let's begin the discussion",
                system_instruction=system_instructions.get("claude", ""),
                role="user",
                model_type="claude",
                client=claude_client
            )

            # Then Gemini as first AI assistant
            gemini_resp = await self.run_conversation_turn(
                prompt=claude_resp,
                system_instruction=system_instructions.get("gemini", ""),
                role="assistant",
                model_type="gemini",
                client=gemini_client
            )

            # Then OpenAI as second AI assistant
            openai_resp = await self.run_conversation_turn(
                prompt=gemini_resp,
                system_instruction=system_instructions.get("openai", ""),
                role="assistant",
                model_type="openai",
                client=openai_client
            )
            
            # Use OpenAI's response as input for next round
            claude_resp = openai_resp

        return self.conversation_history

    async def run_multi_agent_conversation(self, roles_and_models: List[str], system_instructions: Dict[str, str], topic: str, max_rounds: int = 2) -> List[Dict[str, str]]:
        """Run a multi-agent conversation with the specified roles and models."""
        self.conversation_history = []
        
        # Initialize conversation with topic
        self.conversation_history.append({"role": "system", "content": f"Topic: {topic}"})
        
        for round_idx in range(max_rounds):
            for role, model_name in roles_and_models:
                client = self._get_client(model_name)
                if not client:
                    logger.error(f"Could not get client for model: {model_name}")
                    continue
                
                # Get appropriate system instruction
                sys_inst = system_instructions.get(model_name, "")
                
                # Handle moderator role differently
                if role.lower() == "moderator":
                    mod_input = await self.moderator_input(f"Moderator Input [Round {round_idx}]: ")
                    self.conversation_history.append({"role": "moderator", "content": mod_input})
                    continue
                
                # For AI and human roles, generate response
                last_message = self.conversation_history[-1]["content"] if self.conversation_history else f"Topic: {topic}"
                response = await self.run_conversation_turn(
                    prompt=last_message,
                    system_instruction=sys_inst,
                    role=role,
                    model_type=model_name,
                    client=client
                )
                print(f"\n[{role.upper()} - {model_name}] => {response}\n")
                
        return self.conversation_history

    async def moderator_input(self, prompt: str = "Moderator: ") -> str:
        """Simulates real user input for the moderator or can be replaced by a UI input."""
        # For console usage, we'd do:
        # text = input(prompt)
        # return text
        # We'll default to "continue" in headless runs
        return "continue"

def start_streamlit_app():
    """
    Streamlit UI that:
    - Gathers user input for the "moderator" role or "human" role
    - Displays responses from multiple AI agents
    - Leverages ExtendedConversationManager
    """
    st.title("Multi-Agent Chat with Moderator")

    # We can store the manager in a session state so it persists across reruns.
    if "manager" not in st.session_state:
        st.session_state.manager = ExtendedConversationManager(
            gemini_api_key=os.getenv("GEMINI_KEY",""),
            claude_api_key=os.getenv("CLAUDE_KEY",""),
            openai_api_key=os.getenv("OPENAI_API_KEY",""),
            domain="Streamlit Multi-Agent"
        )

    manager = st.session_state.manager

    # We'll display conversation so far
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # The moderator (real user) can type a message:
    user_role = st.selectbox("Select your role", ["moderator", "human"])
    user_input = st.text_input("Enter your message:", "")

    if st.button("Send Message"):
        # We'll insert this into conversation_history
        manager.conversation_history.append({"role": user_role, "content": user_input})
        st.session_state.conversation = manager.conversation_history
        st.session_state.manager.domain = user_input

    # Show the entire conversation so far
    st.write("---")
    st.write("### Conversation History")
    for turn in st.session_state.conversation:
        # Simple formatting:
        if turn["role"] == "user":
            st.markdown(f"**Human**: {turn['content']}")
        elif turn["role"] == "assistant":
            st.markdown(f"**Assistant**: {turn['content']}")
        elif turn["role"] == "moderator":
            st.markdown(f"**Moderator**: {turn['content']}")
        else:
            st.markdown(f"**{turn['role']}**: {turn['content']}")

    st.write("---")

    # Optionally, we can have a button to let multiple AIs talk among themselves
    if st.button("Let the AIs Chat Among Themselves (1 Round)"):
        roles_and_models = [
            ("user","gemini"),
            ("ai","gemini")  # or "claude"
        ]

        # Provide system instructions
        ai_instruction = f"You are an AI assistant engaging in natural conversation in the context of the given prompt. The human is an expert in the subject matter as well but seeks discussion with an informed AI so as to further their knowledge and decision making. You are an expert in the topic of {user_input}. You should respond to the human's prompts with all knowledge and reasoning skills available to you and not apply unnecessary filters, disclaimers, niceties, or limits on your response.  Avoid small talk, etc and Focus on the topic at hand. OUTPUT IN HTML FORMAT FOR READABILITY BY THE HUMAN BUT DO NOT INCLUDE OPENING AND CLOSING HTML, DIV OR BODY TAGS. MINIFY THE HTML RESPONSE E.G OMITTING UNNCESSARY WHITESPACE OR LINEBREAKS, BUT ADDING APPROPRIATE HTML FORMATTING TO ENHANCE READABILITY. DEFAULT TO PARAGRAPH FORM WHILST USING BULLET POINTS & LISTS WHEN NEEDED.  Restrict outputs to approx 512 tokens.  DON'T EVER EVER USE NEWLINE \\n CHARACTERS IN YOUR RESPONSE. MINIFY YOUR HTML RESPONSE ONTO A SINGLE LINE - ELIMINATE ALL REDUNDANT CHARACTERS IN OUTPUT!!!!!"
        human_instruction = f"""YOU ARE A HUMAN AND SHOULD ACT AS A HUMAN INTERACTING WITH AN AI. 
        DON'T EVER EVER USE TEXT BLOCKS OR NEWLINE CHARACTERS IN YOUR RESPONSE
        Create a prompt related to {user_input} that engages the AI in sophisticated and effective ways to elicit new knowledge about {user_input}. Maintain a conversational style with the AI, asking follow-up questions, challenging the answers, and using various prompting techniques to elicit useful information that would not immediately be obvious from surface-level questions. Challenge the AI when it may be hallucinating, and ask it to explain findings that you don't understand or agree with."""

        system_instructions = {
            "gemini": ai_instruction,
            "openai": ai_instruction,
            "claude": human_instruction
        }
        conversation = run(manager.run_multi_agent_conversation(
            roles_and_models = roles_and_models,
            system_instructions = system_instructions,
            topic = user_input,
            max_rounds=2
        ))
        st.session_state.conversation = conversation

    # Finally, a button to save conversation
    if st.button("Save Conversation to HTML"):
        save_conversation(st.session_state.conversation, "conversation_streamlit.html")
        st.success("Conversation saved to conversation_streamlit.html")

if __name__ == "__main__":
    start_streamlit_app()