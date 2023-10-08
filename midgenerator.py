# Import necessary libraries
import streamlit as st
import os
from langchain import PromptTemplate
from langchain.llms import OpenAI
from trubrics.integrations.streamlit import FeedbackCollector

template = """
    The following text requires transformation into a detailed prompt for Midjourney, an AI-based text-to-image program. 
    Your tasks are:
    **Identify the Primary Subject or Theme**:
        Task: Extract the main subject or theme from the text.
        Example 1: A fierce dragon tamer adorned in fire-resistant armor.
        Example 2: A young elven mage with flowing silver hair.

    **Elaborate on the Subject with Intricate Details**:
        Task: Provide a detailed description of the subject.
        Example 1: The tamer's armor is studded with gemstones that glow in the dark, and they wield a staff made of ancient bone.
        Example 2: The mage has intricate tattoos glowing on their arms, signifying their mastery over elemental magic.

    **Environment & Atmosphere**:
        Task: Describe the environment and mood of the scene.
        Example 1: Perched on a cliff, with a backdrop of erupting volcanoes and a crimson sky, signifying an impending battle.
        Example 2: Standing at the edge of a serene forest glade, with magical creatures like phoenixes and unicorns grazing nearby.

    **Output Details & Inspiration**:
        Task: Specify the desired output details and inspiration.
        Example 1: rendered in a high-fantasy style, reminiscent of the works of artists like Yoshitaka Amano.
        Example 2: in vibrant anime style, drawing inspiration from renowned creators like Hayao Miyazaki.

    Construct the prompt using the following format. Only place where -- should be used is in front of --v 5 options:
    "/imagine: [subject with intricate details], [vivid environment description], [mood and atmosphere], [specific output details] --v 5"

    Provided Text:
    {user_text}

    YOUR ENHANCED MIDJOURNEY PROMPT SUGGESTION:
"""


prompt = PromptTemplate(
    input_variables=["user_text"],
    template=template,
)

def load_LLM(openai_api_key):
    """Logic for loading the chain"""
    llm = OpenAI(temperature=1.0, openai_api_key=openai_api_key)
    return llm

# Load API key from environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found in environment variables!")

# Set page configuration
st.set_page_config(
    page_title="Midjourney Prompt Generator",
    page_icon=":rocket:",
    layout="wide",
)

st.markdown("""
## Midjourney Prompt Generator with GPT

Welcome to the Midjourney Prompt Generator! This tool is designed to help you craft detailed and coherent prompts for the Midjourney text-to-image program. Here's how to use it:

1. **Enter Your Basic Idea**: In the textbox provided, type in the basic concept or subject you'd like to visualize. The current limit is 500 characters.
2. **Generate Detailed Prompt**: After entering your idea, click on the "Generate Prompt" button. The app will expand on your concept and suggest a detailed prompt that Midjourney can use to render a high-quality image.
3. **Review & Customize**: The generated prompt will be displayed below. Review it and make any necessary tweaks to ensure it captures your vision accurately.
4. **Feedback is Essential**: Your insights are crucial to our continuous improvement. By sharing your feedback, you aid us in refining and enhancing this tool. So, if you find this application beneficial or have ideas for upgrades, please share your thoughts!

Thank you for using the Midjourney Prompt Generator. Let's bring your ideas to life!
""")

            


# Textbox for users to input their idea with a character limit
user_text = st.text_area("Basic image idea:", max_chars=500)

# Button to trigger the GPT analysis
if st.button("Generate Prompt"):

    # Load the LLM with the provided API key and temperature
    llm = load_LLM(openai_api_key)
    
    # Use the LLM to generate the prompt
    prompt_with_text = prompt.format(user_text=user_text)
    response = llm(prompt_with_text)
    st.write(response)

# Trubrics Feedback Collector
collector = FeedbackCollector(
    project="midgenerator",
    email=os.environ.get("TRUBRICS_EMAIL"),
    password=os.environ.get("TRUBRICS_PASSWORD"),
)

collector.st_feedback(
    component="rating",
    feedback_type="faces",
    model="gpt-3.5-turbo",
    prompt_id=None,  # see prompts to log prompts and model generations
    open_feedback_label='[Optional] Provide additional feedback'
)


# Display the text description
st.write("If you like my work, consider supporting me:")

# Embed the custom "Buy Me A Coffee" button
st.markdown("""
    <a href="https://www.buymeacoffee.com/sherlockanddan" target="_blank">
        <img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;">
    </a>
""", unsafe_allow_html=True)
